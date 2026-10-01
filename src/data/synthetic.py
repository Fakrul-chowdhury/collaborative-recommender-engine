"""Synthetic retail transactions, used only when the real dataset cannot be downloaded.

The generator mimics the properties that matter to a recommender: power-law item
popularity, users whose taste concentrates on a few product categories, a long tail of
one-off shoppers, and the kinds of dirty rows (cancellations, postage lines, anonymous
checkouts) that the SQL cleaning layer has to remove.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_CATEGORIES: dict[str, list[str]] = {
    "MUG": ["CERAMIC", "ENAMEL", "RETRO", "SPOTTY", "BONE CHINA"],
    "CUSHION COVER": ["VELVET", "PATCHWORK", "FLORAL", "VINTAGE", "EMBROIDERED"],
    "T-LIGHT HOLDER": ["HANGING HEART", "GLASS STAR", "ZINC", "FILIGREE", "ANTIQUE"],
    "LUNCH BAG": ["WOODLAND", "POLKADOT", "SKULL", "PAISLEY", "CARS"],
    "CAKE TIN": ["REGENCY", "ROSES", "SWEETHEART", "ROUND", "NESTING SET"],
    "CHRISTMAS DECORATION": ["WOODEN STAR", "FELT TREE", "PAPER CHAIN", "BAUBLE", "REINDEER"],
    "WALL CLOCK": ["BAKELIKE", "IVORY", "DINER", "KITCHEN", "ALARM"],
    "STORAGE JAR": ["GLASS", "CLIP TOP", "HERB", "SPICE", "KILNER STYLE"],
    "GIFT WRAP": ["RIBBON REEL", "TISSUE", "KRAFT PAPER", "GIFT TAG", "BIRTHDAY"],
    "GARDEN TOOL": ["TROWEL", "KNEELING PAD", "PLANT LABEL", "WATERING CAN", "GLOVES"],
    "NOTEBOOK": ["A5 LINED", "POCKET", "SKETCH", "RECIPE", "ADDRESS"],
    "KIDS TOY": ["SPACEBOY", "DOLLY GIRL", "WOODEN BLOCKS", "SKIPPING ROPE", "JIGSAW"],
}
_COLOURS = ["RED", "BLUE", "PINK", "WHITE", "GREEN", "IVORY", "BLACK", "CREAM", "YELLOW", "LILAC"]
_COUNTRIES = ["United Kingdom"] * 9 + ["Germany", "France", "EIRE", "Netherlands"]


def _build_catalogue(rng: np.random.Generator, n_items: int) -> pd.DataFrame:
    """Create items with a category, a category-flavoured description and a Zipf weight."""
    names = list(_CATEGORIES)
    category = rng.integers(0, len(names), size=n_items)
    descriptions = [
        f"{rng.choice(_COLOURS)} {rng.choice(_CATEGORIES[names[c]])} {names[c]}" for c in category
    ]
    codes = rng.choice(np.arange(10000, 99999), size=n_items, replace=False).astype(str)
    suffix = rng.choice(["", "", "", "A", "B", "L"], size=n_items)
    # Zipf-like popularity over a random permutation of the catalogue.
    popularity = 1.0 / np.arange(1, n_items + 1) ** 0.9
    return pd.DataFrame(
        {
            "stock_code": np.char.add(codes, suffix),
            "description": descriptions,
            "category": category,
            "popularity": rng.permutation(popularity),
            "price": np.round(np.exp(rng.normal(0.9, 0.7, size=n_items)), 2).clip(0.19),
        }
    )


def generate_synthetic_transactions(n_users: int = 3000, n_items: int = 1500, months: int = 12,
                                    seed: int = 42) -> pd.DataFrame:
    """Generate raw transaction lines with the same columns as Online Retail II.

    Args:
        n_users: Number of distinct customers.
        n_items: Size of the product catalogue.
        months: Length of the simulated period (the period ends on 2011-12-09).
        seed: Seed for the random generator; identical seeds give identical data.

    Returns:
        One row per invoice line, with the raw column names
        (see ``src.data.download.RAW_COLUMNS``), including deliberately dirty rows.
    """
    rng = np.random.default_rng(seed)
    items = _build_catalogue(rng, n_items)
    n_categories = len(_CATEGORIES)
    category_items = [np.flatnonzero(items["category"].to_numpy() == c) for c in range(n_categories)]
    category_probs = [items["popularity"].to_numpy()[idx] / items["popularity"].to_numpy()[idx].sum()
                      for idx in category_items]

    end = pd.Timestamp("2011-12-09 18:00:00")
    start = end - pd.DateOffset(months=months)
    span_seconds = int((end - start).total_seconds())

    # Taste: each user concentrates on a few categories (sparse Dirichlet).
    tastes = rng.dirichlet(np.full(n_categories, 0.15), size=n_users)
    # Activity: heavy-tailed number of invoices, so many users only ever shop once.
    n_invoices = np.minimum(rng.geometric(0.28, size=n_users), 40)
    first_purchase = rng.integers(0, span_seconds, size=n_users)

    rows: list[tuple] = []
    invoice_number = 500000
    for user in range(n_users):
        customer_id = 12000 + user
        country = _COUNTRIES[rng.integers(len(_COUNTRIES))]
        offsets = np.sort(rng.integers(first_purchase[user], span_seconds, size=n_invoices[user]))
        offsets[0] = first_purchase[user]
        for offset in offsets:
            invoice_number += 1
            basket_size = 1 + rng.poisson(7)
            categories = rng.choice(n_categories, size=basket_size, p=tastes[user])
            chosen = {int(rng.choice(category_items[c], p=category_probs[c])) for c in categories}
            timestamp = start + pd.Timedelta(seconds=int(offset))
            for item in chosen:
                rows.append((
                    str(invoice_number),
                    items.at[item, "stock_code"],
                    items.at[item, "description"],
                    int(rng.geometric(0.25)),
                    timestamp,
                    float(items.at[item, "price"]),
                    customer_id,
                    country,
                ))
            if rng.random() < 0.15:  # postage line: a non-product stock code
                rows.append((str(invoice_number), "POST", "POSTAGE", 1, timestamp, 18.0,
                             customer_id, country))

    frame = pd.DataFrame(rows, columns=["invoice", "stock_code", "description", "quantity",
                                        "invoice_date", "price", "customer_id", "country"])
    frame["customer_id"] = frame["customer_id"].astype("float64")

    # Dirty rows the cleaning SQL must drop.
    n = len(frame)
    cancelled = frame.sample(frac=0.02, random_state=seed).copy()
    cancelled["invoice"] = "C" + cancelled["invoice"]
    cancelled["quantity"] = -cancelled["quantity"]
    anonymous = frame.sample(frac=0.08, random_state=seed + 1).copy()
    anonymous["customer_id"] = np.nan
    anonymous["invoice"] = (anonymous["invoice"].astype(int) + 400000).astype(str)
    free = frame.sample(n=max(n // 500, 1), random_state=seed + 2).copy()
    free["price"] = 0.0
    free["invoice"] = (free["invoice"].astype(int) + 800000).astype(str)

    out = pd.concat([frame, cancelled, anonymous, free], ignore_index=True)
    return out.sort_values(["invoice_date", "invoice"], kind="stable").reset_index(drop=True)
