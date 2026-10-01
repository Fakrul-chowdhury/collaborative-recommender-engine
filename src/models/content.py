"""Content-based recommender built on product-category embeddings.

Item descriptions ("WHITE HANGING HEART T-LIGHT HOLDER") are the only item metadata in
the data. They are embedded with TF-IDF followed by TruncatedSVD (latent semantic
analysis), and product categories are derived by clustering those embeddings with
KMeans. The recommender represents each product by its description embedding pulled
towards its category centroid, and each low-history user by the mean of the few
products they bought.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src import config
from src.interactions import InteractionData
from src.models.base import PAD, top_k_excluding


@dataclass(frozen=True)
class ItemEmbeddings:
    """Description embeddings and derived categories for the catalogue.

    Attributes:
        vectors: ``(n_items, dim)`` unit-norm description embeddings (all-zero rows for
            products without a usable description). Also the space diversity is measured in.
        categories: KMeans cluster label per item.
        centroids: ``(n_categories, dim)`` unit-norm category centroids.
    """

    vectors: np.ndarray
    categories: np.ndarray
    centroids: np.ndarray

    def category_aware(self, category_weight: float) -> np.ndarray:
        """Blend each item with its category centroid: ``normalize(v + w * centroid)``.

        ``category_weight = 0`` returns the plain description embeddings; larger values
        make products of the same category increasingly similar to one another.
        """
        if category_weight == 0:
            return self.vectors
        return normalize(self.vectors + category_weight * self.centroids[self.categories])


def build_item_embeddings(item_ids: Sequence[str], descriptions: Mapping[str, str | None],
                          n_components: int = config.EMBEDDING_DIM,
                          n_categories: int = config.N_CATEGORIES,
                          seed: int = config.SEED) -> ItemEmbeddings:
    """Embed product descriptions and cluster them into categories.

    Args:
        item_ids: Stock codes in catalogue order.
        descriptions: Stock code -> description; missing or empty descriptions are allowed.
        n_components: Target embedding dimensionality (capped by the vocabulary size).
        n_categories: Number of KMeans clusters (capped by the number of items).
        seed: Seed for SVD and KMeans.
    """
    texts = [(descriptions.get(str(item)) or "").lower() for item in item_ids]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 stop_words="english", token_pattern=r"[a-z]{2,}")
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:  # tiny catalogues: no term appears twice
        vectorizer.set_params(min_df=1)
        tfidf = vectorizer.fit_transform(texts)

    n_components = max(1, min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    vectors = normalize(svd.fit_transform(tfidf))

    n_categories = max(1, min(n_categories, len(texts)))
    kmeans = KMeans(n_clusters=n_categories, n_init=10, random_state=seed)
    categories = kmeans.fit_predict(vectors)
    return ItemEmbeddings(vectors=vectors, categories=categories,
                          centroids=normalize(kmeans.cluster_centers_))


class ContentRecommender:
    """Nearest-neighbour recommender over category-aware product embeddings.

    A user's profile is the mean embedding of the items they bought; candidates are
    scored by cosine similarity to it, plus an optional popularity prior:
    ``score = cosine + popularity_weight * log1p(buyers) / log1p(max_buyers)``.

    Args:
        embeddings: Output of :func:`build_item_embeddings`.
        category_weight: Strength of the pull towards the category centroid.
        popularity_weight: Weight of the popularity prior (0 = pure similarity).
    """

    def __init__(self, embeddings: ItemEmbeddings, category_weight: float = 0.5,
                 popularity_weight: float = 0.0) -> None:
        self.embeddings = embeddings
        self.category_weight = category_weight
        self.popularity_weight = popularity_weight
        self._vectors = embeddings.category_aware(category_weight)
        self._prior: np.ndarray | None = None
        self._data: InteractionData | None = None

    def fit(self, data: InteractionData) -> "ContentRecommender":
        """Compute the popularity prior from training interactions."""
        if data.n_items != len(self._vectors):
            raise ValueError("Embeddings and interaction data cover different catalogues")
        buyers = np.bincount(data.matrix.indices, minlength=data.n_items).astype(np.float64)
        self._prior = np.log1p(buyers) / max(np.log1p(buyers.max()), 1e-12)
        self._data = data
        return self

    def _recommend_profile(self, items: np.ndarray, exclude: np.ndarray, k: int) -> np.ndarray:
        profile = self._vectors[items].mean(axis=0)
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        scores = self._vectors @ profile + self.popularity_weight * self._prior
        return top_k_excluding(scores, exclude, k)

    def recommend(self, customer_ids: Sequence[int], k: int) -> np.ndarray:
        """Top-k most similar unseen items per customer.

        Customers with no training history have no profile and get an all-``PAD`` row.
        """
        if self._data is None:
            raise RuntimeError("Call fit() before recommend()")
        out = np.full((len(customer_ids), k), PAD, dtype=np.int64)
        for row, customer_id in enumerate(customer_ids):
            seen = self._data.seen_items(customer_id)
            if len(seen):
                out[row] = self._recommend_profile(seen, seen, k)
        return out

    def recommend_truncated(self, customer_ids: Sequence[int], visible: Sequence[np.ndarray],
                            k: int) -> np.ndarray:
        """Recommend from a profile built only on each customer's ``visible`` items.

        Everything the customer really bought is still excluded from the output.
        """
        if self._data is None:
            raise RuntimeError("Call fit() before recommend()")
        out = np.full((len(customer_ids), k), PAD, dtype=np.int64)
        for row, (customer_id, items) in enumerate(zip(customer_ids, visible)):
            out[row] = self._recommend_profile(np.asarray(items),
                                               self._data.seen_items(customer_id), k)
        return out
