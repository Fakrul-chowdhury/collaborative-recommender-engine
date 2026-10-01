"""Recommenders: popularity baseline, ALS, content-based fallback and the hybrid router."""

from src.models.als import ALSRecommender, confidence_matrix
from src.models.base import PAD, Recommender
from src.models.content import ContentRecommender, ItemEmbeddings, build_item_embeddings
from src.models.hybrid import SEGMENTS, HybridRecommender, route_user
from src.models.popularity import PopularityRecommender

__all__ = [
    "ALSRecommender",
    "ContentRecommender",
    "HybridRecommender",
    "ItemEmbeddings",
    "PAD",
    "PopularityRecommender",
    "Recommender",
    "SEGMENTS",
    "build_item_embeddings",
    "confidence_matrix",
    "route_user",
]
