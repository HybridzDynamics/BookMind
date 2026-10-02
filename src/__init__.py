"""
BookMind Core Recommendation Source Package.

This package contains data preparation routines, recommendation algorithms,
evaluation metrics, graph generators, and virtual user simulators:
- `data_cleaning`: Cleans raw Goodbooks-10k metadata.
- `prepare_ratings`: Filters and deduplicates user ratings.
- `popularity_model`: Bayesian weighted popularity scoring.
- `content_model`: TF-IDF and NearestNeighbors content filtering.
- `collaborative_model`: TruncatedSVD matrix factorization.
- `hybrid_model`: Multi-signal weighted ensemble.
- `evaluation`: Precision@K, Recall@K, and NDCG@K ranking metrics.
- `generate_report_graphs`: Matplotlib visualization generation.
- `virtual_simulation`: Realistic user search/session simulator.
"""

from src.popularity_model import get_popular_books, train_popularity_model
from src.content_model import recommend_similar_books, train_content_model
from src.collaborative_model import recommend_for_user as collaborative_recommend, train_collaborative_model
from src.hybrid_model import recommend_for_user as hybrid_recommend, train_hybrid_model

__all__ = [
    "get_popular_books",
    "train_popularity_model",
    "recommend_similar_books",
    "train_content_model",
    "collaborative_recommend",
    "train_collaborative_model",
    "hybrid_recommend",
    "train_hybrid_model",
]
