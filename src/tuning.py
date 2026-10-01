"""Hyperparameter tuning on the validation window.

Purchase logs only contain positives, so optimising accuracy alone drifts towards
recommending the same safe best-sellers to everyone. The objective therefore rewards
both accuracy and variety:

    objective = w * (precision@k / precision@k of popularity)
              + (1 - w) * (diversity / diversity of popularity)

Both terms are ratios to the popularity baseline on the same users, which puts them on
a comparable scale (1.0 = "as good as popularity"). ``w`` is ``config.ACCURACY_WEIGHT``
and is fixed up front, never tuned.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import optuna
import pandas as pd

from src import config
from src.evaluation.evaluate import EvalResult, evaluate_recommendations, most_recent_items
from src.interactions import InteractionData, history_sizes
from src.models.als import ALSRecommender
from src.models.content import ContentRecommender, ItemEmbeddings
from src.models.popularity import PopularityRecommender

logger = logging.getLogger(__name__)

#: Grid searched for the content model (see :func:`tune_content`).
CONTENT_GRID: dict[str, tuple[float, ...]] = {
    "category_weight": (0.0, 0.5, 1.0, 2.0),
    "popularity_weight": (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
}


def balanced_objective(precision: float, diversity: float, reference_precision: float,
                       reference_diversity: float,
                       accuracy_weight: float = config.ACCURACY_WEIGHT) -> float:
    """Weighted sum of precision and diversity, each relative to a reference model."""
    if not 0.0 <= accuracy_weight <= 1.0:
        raise ValueError("accuracy_weight must be in [0, 1]")
    return (accuracy_weight * precision / reference_precision
            + (1.0 - accuracy_weight) * diversity / reference_diversity)


@dataclass(frozen=True)
class ValidationSet:
    """Everything needed to score a model on validation users with enough history for ALS."""

    data: InteractionData
    truth: Mapping[int, set[str]]
    customer_ids: np.ndarray
    item_vectors: np.ndarray
    reference: dict[str, float]
    k: int

    def evaluate(self, recommendations: np.ndarray) -> EvalResult:
        return evaluate_recommendations(recommendations, self.customer_ids, self.truth,
                                        self.data, self.item_vectors, self.k)

    def objective(self, summary: Mapping[str, float]) -> float:
        k = self.k
        return balanced_objective(summary[f"precision@{k}"], summary["diversity"],
                                  self.reference[f"precision@{k}"], self.reference["diversity"])


def build_validation_set(data: InteractionData, truth: Mapping[int, set[str]],
                         item_vectors: np.ndarray, k: int = config.K,
                         min_history: int = config.MIN_HISTORY) -> ValidationSet:
    """Select the validation users ALS would serve and score the popularity reference."""
    customers = np.array(sorted(truth))
    customers = customers[history_sizes(data, customers) >= min_history]
    popularity = PopularityRecommender().fit(data).recommend(customers, k)
    reference = evaluate_recommendations(popularity, customers, truth, data, item_vectors,
                                         k).summary()
    return ValidationSet(data, truth, customers, item_vectors, reference, k)


def tune_als(validation: ValidationSet, n_trials: int = config.N_TRIALS,
             seed: int = config.SEED) -> tuple[dict, pd.DataFrame]:
    """Search ALS hyperparameters with Optuna's TPE sampler.

    Returns:
        ``(best_params, trials)`` where ``trials`` has one row per trial with its
        hyperparameters, every validation metric and the objective value.
    """
    k = validation.k
    records: list[dict] = []

    def run_trial(trial: optuna.Trial) -> float:
        params = {
            "factors": trial.suggest_int("factors", 16, 256, log=True),
            "regularization": trial.suggest_float("regularization", 1e-3, 10.0, log=True),
            "iterations": trial.suggest_int("iterations", 10, 40),
            "alpha": trial.suggest_float("alpha", 0.1, 50.0, log=True),
            "weighting": trial.suggest_categorical("weighting", ["count", "log", "bm25"]),
        }
        started = time.perf_counter()
        model = ALSRecommender(seed=seed, **params).fit(validation.data)
        summary = validation.evaluate(model.recommend(validation.customer_ids, k)).summary()
        value = validation.objective(summary)
        records.append({"trial": trial.number, **params, **summary, "objective": value,
                        "fit_seconds": round(time.perf_counter() - started, 2)})
        logger.info("trial %2d  objective=%.4f  precision@%d=%.4f  diversity=%.4f  %s",
                    trial.number, value, k, summary[f"precision@{k}"], summary["diversity"],
                    params)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(run_trial, n_trials=n_trials)
    trials = pd.DataFrame(records).sort_values("trial").reset_index(drop=True)
    return dict(study.best_params), trials


def simulate_cold_start(data: InteractionData, customer_ids: np.ndarray, n_visible: int,
                        seed: int = config.SEED) -> list[np.ndarray]:
    """For each customer, pick the ``n_visible`` most recent training items to keep visible."""
    rng = np.random.default_rng(seed)
    return [most_recent_items(data, customer, n_visible, rng) for customer in customer_ids]


def tune_content(validation: ValidationSet, embeddings: ItemEmbeddings,
                 seed: int = config.SEED) -> tuple[dict, pd.DataFrame]:
    """Grid-search the content model on *simulated* cold-start validation users.

    Too few real validation users have 1-2 training items to tune on, so cold-start is
    simulated: each established user's profile is built from only their 1 or 2 most
    recent purchases, and the same balanced objective is averaged over both cases.

    Returns:
        ``(best_params, grid)`` with one row per grid point and history size.
    """
    k = validation.k
    visible = {n: simulate_cold_start(validation.data, validation.customer_ids, n, seed)
               for n in (1, 2)}
    records: list[dict] = []
    for category_weight in CONTENT_GRID["category_weight"]:
        for popularity_weight in CONTENT_GRID["popularity_weight"]:
            model = ContentRecommender(embeddings, category_weight,
                                       popularity_weight).fit(validation.data)
            for n, items in visible.items():
                recommendations = model.recommend_truncated(validation.customer_ids, items, k)
                summary = validation.evaluate(recommendations).summary()
                records.append({"category_weight": category_weight,
                                "popularity_weight": popularity_weight, "n_visible": n,
                                **summary, "objective": validation.objective(summary)})
    grid = pd.DataFrame(records)
    mean_objective = grid.groupby(["category_weight", "popularity_weight"])["objective"].mean()
    best_category, best_popularity = mean_objective.idxmax()
    return ({"category_weight": float(best_category),
             "popularity_weight": float(best_popularity)}, grid)
