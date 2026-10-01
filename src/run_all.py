"""Command-line entry point: ``python -m src.run_all [--stage data|tune|evaluate|all]``.

Stages:
    data      download the dataset (or synthesise one), load SQLite, run the SQL scripts
    tune      tune ALS (Optuna) and the content fallback (grid) on the validation window
    evaluate  refit on train+validation, score the test window, write metrics and chart
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.data import load_raw_transactions
from src.evaluation.evaluate import (
    EvalResult,
    bootstrap_lift_interval,
    evaluate_recommendations,
    relative_lift,
)
from src.evaluation.plots import plot_recall_by_segment
from src.evaluation.report import SEGMENT_LABELS, render_results, update_readme
from src.interactions import InteractionData, build_ground_truth, build_interaction_data, history_sizes
from src.models import (
    ALSRecommender,
    ContentRecommender,
    HybridRecommender,
    ItemEmbeddings,
    PopularityRecommender,
    build_item_embeddings,
)
from src.split import Split, chronological_split
from src.sql import build_database, connect
from src.tuning import build_validation_set, simulate_cold_start, tune_als, tune_content

logger = logging.getLogger("run_all")

BEST_PARAMS_PATH = config.RESULTS_DIR / "best_params.json"
METRICS_PATH = config.RESULTS_DIR / "metrics.json"
RECENT_WINDOW_DAYS = 28


def _write_json(path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _has_raw_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_transactions'"
    ).fetchone()
    return row is not None


def stage_data(force_synthetic: bool = False, refresh: bool = False) -> dict[str, Any]:
    """Build the SQLite database and return the cleaning report plus the data source."""
    conn = connect()
    try:
        if refresh or force_synthetic or not _has_raw_table(conn) or not config.SOURCE_PATH.exists():
            raw, source = load_raw_transactions(force_synthetic=force_synthetic)
            report = build_database(conn, raw)
            _write_json(config.SOURCE_PATH, {"source": source})
        else:
            logger.info("Raw table already loaded; re-running the SQL scripts only")
            report = build_database(conn)
            source = json.loads(config.SOURCE_PATH.read_text(encoding="utf-8"))["source"]
    finally:
        conn.close()
    report = {"source": source, **report}
    logger.info("Data ready: %s", report)
    return report


def _load_split_and_descriptions() -> tuple[Split, dict[str, str]]:
    conn = connect()
    try:
        split = chronological_split(conn)
        items = pd.read_sql_query("SELECT stock_code, description FROM items", conn)
    finally:
        conn.close()
    return split, dict(zip(items["stock_code"].astype(str), items["description"]))


def stage_tune(n_trials: int = config.N_TRIALS) -> dict[str, Any]:
    """Tune on the validation window (models are fitted on the training window only)."""
    split, descriptions = _load_split_and_descriptions()
    data = build_interaction_data(split.train)
    truth = build_ground_truth(split.val, data)
    embeddings = build_item_embeddings(data.item_ids, descriptions)
    validation = build_validation_set(data, truth, embeddings.vectors)
    logger.info("Tuning on %d validation users; popularity reference: %s",
                len(validation.customer_ids), validation.reference)

    als_params, trials = tune_als(validation, n_trials=n_trials)
    content_params, grid = tune_content(validation, embeddings)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials.to_csv(config.RESULTS_DIR / "optuna_trials.csv", index=False, float_format="%.6g")
    grid.to_csv(config.RESULTS_DIR / "content_tuning.csv", index=False, float_format="%.6g")
    best_trial = trials.loc[trials["objective"].idxmax()]
    payload = {
        "accuracy_weight": config.ACCURACY_WEIGHT,
        "n_trials": int(len(trials)),
        "n_validation_users": int(len(validation.customer_ids)),
        "validation_window": [split.bounds.as_dict()["val_start"],
                              split.bounds.as_dict()["test_start"]],
        "popularity_reference": validation.reference,
        "als_best_params": als_params,
        "als_best_objective": float(best_trial["objective"]),
        "als_best_validation_metrics": {
            name: float(best_trial[name]) for name in
            (f"recall@{config.K}", f"precision@{config.K}", f"ndcg@{config.K}", "diversity",
             "coverage")
        },
        "content_best_params": content_params,
    }
    _write_json(BEST_PARAMS_PATH, payload)
    logger.info("Best ALS params: %s (objective %.4f)", als_params, payload["als_best_objective"])
    logger.info("Best content params: %s", content_params)
    return payload


def _segment_masks(data: InteractionData, customers: np.ndarray) -> dict[str, np.ndarray]:
    sizes = history_sizes(data, customers)
    return {
        "all": np.ones(len(customers), dtype=bool),
        "established": sizes >= config.MIN_HISTORY,
        "low_history": (sizes > 0) & (sizes < config.MIN_HISTORY),
        "new": sizes == 0,
    }


def _summarise(result: EvalResult, masks: dict[str, np.ndarray]) -> dict[str, dict]:
    return {segment: result.summary(mask) for segment, mask in masks.items()}


def stage_evaluate(data_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Refit on train+validation with the tuned parameters and score the test window."""
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError("results/best_params.json not found; run --stage tune first")
    tuning = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    if data_report is None:
        data_report = stage_data()

    split, descriptions = _load_split_and_descriptions()
    data = build_interaction_data(split.train_val)
    truth = build_ground_truth(split.test, data)
    customers = np.array(sorted(truth))
    embeddings: ItemEmbeddings = build_item_embeddings(data.item_ids, descriptions)
    masks = _segment_masks(data, customers)
    k = config.K

    popularity = PopularityRecommender().fit(data)
    recent = PopularityRecommender(window_days=RECENT_WINDOW_DAYS).fit(data)
    als = ALSRecommender(**tuning["als_best_params"]).fit(data)
    content = ContentRecommender(embeddings, **tuning["content_best_params"]).fit(data)
    models = {
        "popularity": popularity,
        "recent_popularity": recent,
        # ALS for every user it has factors for; brand-new users can only get popularity.
        "als": HybridRecommender(data, als, content, popularity, min_history=1),
        "hybrid": HybridRecommender(data, als, content, popularity, min_history=config.MIN_HISTORY),
    }
    results = {
        name: evaluate_recommendations(model.recommend(customers, k), customers, truth, data,
                                       embeddings.vectors, k)
        for name, model in models.items()
    }
    test_metrics = {name: _summarise(result, masks) for name, result in results.items()}

    # Simulated cold-start on established users: only their most recent items are visible.
    established = customers[masks["established"]]
    simulated: dict[str, dict] = {}
    for n_visible in (1, 2):
        visible = simulate_cold_start(data, established, n_visible)
        candidates = {
            "popularity": popularity.recommend(established, k),
            "content": content.recommend_truncated(established, visible, k),
            "als_fold_in": als.recommend_truncated(established, visible, k),
        }
        simulated[str(n_visible)] = {
            name: evaluate_recommendations(recs, established, truth, data, embeddings.vectors,
                                           k).summary()
            for name, recs in candidates.items()
        }

    n_users = len(customers)
    train_sizes = np.diff(data.matrix.indptr)
    below = masks["low_history"] | masks["new"]
    hybrid_recall, popularity_recall = results["hybrid"].recall, results["popularity"].recall
    metrics = {
        "data": data_report,
        "split": {
            "bounds": split.bounds.as_dict(),
            "interactions": {"train": len(split.train), "validation": len(split.val),
                             "train_plus_validation": len(split.train_val),
                             "test": len(split.test)},
            "catalogue_size": data.n_items,
        },
        "protocol": {
            "k": k,
            "min_history_for_als": config.MIN_HISTORY,
            "seed": config.SEED,
            "ground_truth": "items bought in the test window that the user had not bought "
                            "before it; users with no such item are not evaluated",
            "final_models_fitted_on": "train + validation",
        },
        "user_segments": {
            "test": {
                "n_users": n_users,
                **{segment: {"n_users": int(mask.sum()), "share": float(mask.mean())}
                   for segment, mask in masks.items() if segment != "all"},
                "below_threshold_share": float(below.mean()),
            },
            "train_customers": {
                "n_users": int(data.n_users),
                "below_threshold": int((train_sizes < config.MIN_HISTORY).sum()),
                "below_threshold_share": float((train_sizes < config.MIN_HISTORY).mean()),
            },
        },
        "test_metrics": test_metrics,
        "headline": {
            "hybrid_recall": float(hybrid_recall.mean()),
            "popularity_recall": float(popularity_recall.mean()),
            "recall_lift_vs_popularity": relative_lift(hybrid_recall, popularity_recall),
            "recall_lift_ci95": list(bootstrap_lift_interval(hybrid_recall, popularity_recall)),
            "recall_lift_vs_recent_popularity": relative_lift(
                hybrid_recall, results["recent_popularity"].recall),
            "als_recall_lift_vs_popularity": relative_lift(results["als"].recall,
                                                           popularity_recall),
        },
        "simulated_cold_start": simulated,
        "tuning": tuning,
    }
    _write_json(METRICS_PATH, metrics)

    chart_models = {"Popularity": "popularity", "ALS": "als", "Hybrid": "hybrid"}
    recall_by_segment = {
        label: {SEGMENT_LABELS[s]: test_metrics[name][s].get(f"recall@{k}", 0.0)
                for s in SEGMENT_LABELS}
        for label, name in chart_models.items()
    }
    sizes = {SEGMENT_LABELS[s]: int(mask.sum()) for s, mask in masks.items()}
    plot_recall_by_segment(recall_by_segment, sizes,
                           config.RESULTS_DIR / f"recall_at_{k}_by_segment.png", k)

    results_markdown = render_results(metrics)
    (config.RESULTS_DIR / "results.md").write_text(results_markdown + "\n", encoding="utf-8")
    if update_readme(config.ROOT / "README.md", results_markdown):
        logger.info("README results section updated")
    logger.info("Headline: %s", metrics["headline"])
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["data", "tune", "evaluate", "all"], default="all")
    parser.add_argument("--trials", type=int, default=config.N_TRIALS,
                        help="number of Optuna trials for ALS")
    parser.add_argument("--synthetic", action="store_true",
                        help="skip the download and use synthetic transactions")
    parser.add_argument("--refresh-data", action="store_true",
                        help="reload the raw table even if the database already has it")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")

    data_report = None
    if args.stage in ("data", "all"):
        data_report = stage_data(force_synthetic=args.synthetic, refresh=args.refresh_data)
    if args.stage in ("tune", "all"):
        stage_tune(n_trials=args.trials)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(data_report)


if __name__ == "__main__":
    main()
