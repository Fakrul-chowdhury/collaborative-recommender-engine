"""Content-based fallback: embeddings, categories and nearest-item recommendations."""

from __future__ import annotations

import numpy as np

from src.interactions import build_interaction_data
from src.models.base import PAD
from src.models.content import ContentRecommender, build_item_embeddings
from tests.helpers import interaction_frame, no_seen_items

_DESCRIPTIONS = {
    "10001": "RED CERAMIC MUG", "10002": "BLUE CERAMIC MUG", "10003": "RED ENAMEL MUG",
    "10004": "WOODEN GARDEN TROWEL", "10005": "STEEL GARDEN TROWEL",
    "10006": "RED PAPER NAPKINS", "10007": "BLUE PAPER NAPKINS",
}


def _catalogue():
    return build_interaction_data(interaction_frame([
        (1, "10001", 1, "2011-01-01"),
        (2, "10002", 1, "2011-01-01"), (2, "10003", 1, "2011-01-01"),
        (2, "10004", 1, "2011-01-01"), (2, "10005", 1, "2011-01-01"),
        (2, "10006", 1, "2011-01-01"), (2, "10007", 1, "2011-01-01"),
    ]))


def test_embeddings_are_unit_norm_and_categories_group_similar_products():
    data = _catalogue()
    embeddings = build_item_embeddings(data.item_ids, _DESCRIPTIONS, n_components=5,
                                       n_categories=3)
    assert embeddings.vectors.shape[0] == 7
    np.testing.assert_allclose(np.linalg.norm(embeddings.vectors, axis=1), 1.0, atol=1e-9)
    assert len(set(embeddings.categories)) == 3

    def categories(*codes):
        return {int(embeddings.categories[data.item_index[code]]) for code in codes}

    mugs, trowels, napkins = (categories("10001", "10002"), categories("10004", "10005"),
                              categories("10006", "10007"))
    assert len(mugs) == len(trowels) == len(napkins) == 1
    assert len(mugs | trowels | napkins) == 3


def test_category_blending_pulls_same_category_items_together():
    data = _catalogue()
    embeddings = build_item_embeddings(data.item_ids, _DESCRIPTIONS, n_components=5,
                                       n_categories=3)
    a, b = data.item_index["10001"], data.item_index["10002"]
    plain = embeddings.category_aware(0.0)
    blended = embeddings.category_aware(2.0)
    assert plain is embeddings.vectors
    assert blended[a] @ blended[b] > plain[a] @ plain[b]
    np.testing.assert_allclose(np.linalg.norm(blended, axis=1), 1.0, atol=1e-9)


def test_content_recommends_items_with_similar_descriptions():
    data = _catalogue()
    embeddings = build_item_embeddings(data.item_ids, _DESCRIPTIONS, n_components=5,
                                       n_categories=3)
    model = ContentRecommender(embeddings, category_weight=0.5, popularity_weight=0.0).fit(data)
    # Customer 1 bought a mug: the two other mugs come first.
    assert set(data.item_ids[model.recommend([1], 2)[0]]) == {"10002", "10003"}
    # No history -> no profile -> padding (the hybrid falls back to popularity).
    assert (model.recommend([42], 2)[0] == PAD).all()


def test_missing_descriptions_are_tolerated():
    data = _catalogue()
    partial = {**_DESCRIPTIONS, "10006": None}
    del partial["10005"]
    embeddings = build_item_embeddings(data.item_ids, partial, n_components=3, n_categories=2)
    assert np.isfinite(embeddings.vectors).all()
    # Products without usable text get an all-zero vector instead of a made-up position.
    for code in ("10005", "10006"):
        assert not embeddings.vectors[data.item_index[code]].any()
    model = ContentRecommender(embeddings).fit(data)
    assert model.recommend([1], 3).shape == (1, 3)


def test_content_never_recommends_purchased_items(train_data):
    groups = ["MUG", "CUSHION", "CLOCK", "LANTERN", "NOTEBOOK", "TROWEL", "RIBBON"]
    styles = ["RED", "BLUE", "IVORY"]
    descriptions = {item: f"{styles[int(item[:5]) % 3]} {groups[int(item[:5]) % 7]}"
                    for item in train_data.item_ids}
    embeddings = build_item_embeddings(train_data.item_ids, descriptions, n_components=8,
                                       n_categories=5)
    model = ContentRecommender(embeddings, popularity_weight=1.0).fit(train_data)
    customers = train_data.user_ids[:40].tolist()
    assert no_seen_items(train_data, customers, model.recommend(customers, 10))
