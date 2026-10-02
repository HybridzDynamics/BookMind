"""
BookMind Recommendation Evaluation Module.

This module evaluates all four recommendation models:
1. Popularity-Based
2. Content-Based
3. Collaborative Filtering
4. Hybrid Ensemble

It calculates ranking and retrieval metrics:
- Precision@K: Fraction of top-K recommended items that are relevant.
- Recall@K: Fraction of user's relevant items captured in top-K.
- NDCG@K: Normalized Discounted Cumulative Gain accounting for rank positions.
- Hit Rate@K: Proportion of users for whom at least one recommended item was relevant.
- Catalog Coverage: Proportion of total catalog books recommended across all users.

Includes real-time progress logging, elapsed time tracking, user evaluation speed,
and estimated time of arrival (ETA).

Note: Strictly functional implementation using only `def` functions (no `class`).
"""

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set
import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
import sys
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

USER_RATINGS_PATH = PROJECT_ROOT / "data" / "processed" / "user_ratings.csv"
CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"
POPULARITY_MODEL_PATH = PROJECT_ROOT / "models" / "popularity_model.pkl"
CONTENT_MODEL_PATH = PROJECT_ROOT / "models" / "content_model.pkl"
COLLABORATIVE_MODEL_PATH = PROJECT_ROOT / "models" / "collaborative_model.pkl"
HYBRID_MODEL_PATH = PROJECT_ROOT / "models" / "hybrid_model.pkl"
EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "evaluation_results.csv"
TRAINING_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "training.log"


def log_evaluation_message(
    message_content: str,
    log_file_path: Path = TRAINING_LOG_PATH,
    log_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Format a timestamped evaluation log entry, print to stdout, append to log file,
    and forward to an optional streaming UI callback.

    Parameters
    ----------
    message_content : str
        Message text to log.
    log_file_path : Path
        Filesystem path to the log file.
    log_callback : Optional[Callable[[str], None]]
        Optional streaming callback function.

    Returns
    -------
    str
        Formatted log message with timestamp.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp_str}] [Evaluation] {message_content}"

    print(formatted_message)

    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file_path, "a", encoding="utf-8") as file_handle:
            file_handle.write(formatted_message + "\n")
    except Exception:
        pass

    if log_callback is not None:
        try:
            log_callback(formatted_message)
        except Exception:
            pass

    return formatted_message


def compute_precision_at_k(
    recommended_items: Sequence[str],
    relevant_items: Sequence[str],
    k: int = 10
) -> float:
    """
    Calculate Precision@K.

    Parameters
    ----------
    recommended_items : Sequence[str]
        Ranked list of recommended book IDs.
    relevant_items : Sequence[str]
        Set or list of user-relevant book IDs.
    k : int, default=10
        Cutoff rank threshold.

    Returns
    -------
    float
        Precision@K score.
    """
    if k <= 0:
        return 0.0

    top_k_recommendations = recommended_items[:k]
    relevant_items_set = set(relevant_items)

    if len(top_k_recommendations) == 0:
        return 0.0

    hit_count = 0
    for item in top_k_recommendations:
        if item in relevant_items_set:
            hit_count += 1

    denominator = len(top_k_recommendations)
    if denominator == 0:
        return 0.0
    else:
        return float(hit_count / denominator)


def compute_recall_at_k(
    recommended_items: Sequence[str],
    relevant_items: Sequence[str],
    k: int = 10
) -> float:
    """
    Calculate Recall@K.

    Parameters
    ----------
    recommended_items : Sequence[str]
        Ranked list of recommended book IDs.
    relevant_items : Sequence[str]
        Set or list of user-relevant book IDs.
    k : int, default=10
        Cutoff rank threshold.

    Returns
    -------
    float
        Recall@K score.
    """
    if k <= 0:
        return 0.0

    top_k_recommendations = recommended_items[:k]
    relevant_items_set = set(relevant_items)
    total_relevant_count = len(relevant_items_set)

    if total_relevant_count == 0:
        return 0.0

    hit_count = 0
    for item in top_k_recommendations:
        if item in relevant_items_set:
            hit_count += 1

    return float(hit_count / total_relevant_count)


def compute_ndcg_at_k(
    recommended_items: Sequence[str],
    relevant_items: Sequence[str],
    k: int = 10
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Parameters
    ----------
    recommended_items : Sequence[str]
        Ranked list of recommended book IDs.
    relevant_items : Sequence[str]
        Set or list of user-relevant book IDs.
    k : int, default=10
        Cutoff rank threshold.

    Returns
    -------
    float
        NDCG@K score.
    """
    if k <= 0:
        return 0.0

    top_k_recommendations = recommended_items[:k]
    relevant_items_set = set(relevant_items)

    if len(relevant_items_set) == 0:
        return 0.0

    dcg_score = 0.0
    for rank_position, item in enumerate(top_k_recommendations):
        if item in relevant_items_set:
            discount = float(np.log2(rank_position + 2))
            dcg_score += (1.0 / discount)

    ideal_hits_count = min(k, len(relevant_items_set))
    idcg_score = 0.0
    for rank_position in range(ideal_hits_count):
        discount = float(np.log2(rank_position + 2))
        idcg_score += (1.0 / discount)

    if idcg_score == 0.0:
        return 0.0
    else:
        return float(dcg_score / idcg_score)


def compute_hit_rate_at_k(
    recommended_items: Sequence[str],
    relevant_items: Sequence[str],
    k: int = 10
) -> float:
    """
    Calculate Hit Rate@K (1.0 if at least one hit in top K, else 0.0).

    Parameters
    ----------
    recommended_items : Sequence[str]
        Ranked list of recommended book IDs.
    relevant_items : Sequence[str]
        Set or list of user-relevant book IDs.
    k : int, default=10
        Cutoff rank threshold.

    Returns
    -------
    float
        1.0 if at least one recommended book was relevant, 0.0 otherwise.
    """
    top_k_recommendations = recommended_items[:k]
    relevant_items_set = set(relevant_items)

    for item in top_k_recommendations:
        if item in relevant_items_set:
            return 1.0

    return 0.0


def evaluate_all_models(
    ratings_path: Path = USER_RATINGS_PATH,
    books_path: Path = CLEAN_BOOKS_PATH,
    output_path: Path = EVALUATION_RESULTS_PATH,
    top_k: int = 10,
    max_evaluation_users: int = 500,
    log_callback: Optional[Callable[[str], None]] = None
) -> pd.DataFrame:
    """
    Evaluate Popularity, Content-Based, Collaborative Filtering, and Hybrid models
    with real-time progress logging and timing metrics.

    Parameters
    ----------
    ratings_path : Path
        Path to `user_ratings.csv`.
    books_path : Path
        Path to `books_clean.csv`.
    output_path : Path
        Path where evaluation table will be saved.
    top_k : int, default=10
        Top K cutoff.
    max_evaluation_users : int, default=500
        Maximum test users evaluated.
    log_callback : Optional[Callable[[str], None]]
        Optional streaming callback function for GUI logging.

    Returns
    -------
    pd.DataFrame
        Table with model, precision@10, recall@10, ndcg@10, hit_rate@10, coverage, users_evaluated.
    """
    evaluation_start_timestamp = time.time()

    log_evaluation_message("Starting offline evaluation session...", log_callback=log_callback)

    if not ratings_path.exists():
        error_message = f"Ratings file missing: {ratings_path}"
        log_evaluation_message(f"ERROR: {error_message}", log_callback=log_callback)
        raise FileNotFoundError(error_message)

    if not books_path.exists():
        error_message = f"Clean books file missing: {books_path}"
        log_evaluation_message(f"ERROR: {error_message}", log_callback=log_callback)
        raise FileNotFoundError(error_message)

    log_evaluation_message(f"Loading interaction dataset from: {ratings_path}", log_callback=log_callback)
    ratings_df = pd.read_csv(ratings_path)
    ratings_df["user_id"] = ratings_df["user_id"].astype(str)
    ratings_df["book_id"] = ratings_df["book_id"].astype(str)
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")
    ratings_df = ratings_df.dropna()

    total_ratings_count = len(ratings_df)
    total_unique_users = ratings_df["user_id"].nunique()
    log_evaluation_message(
        f"Loaded {total_ratings_count:,} ratings across {total_unique_users:,} unique users.",
        log_callback=log_callback
    )

    books_df = pd.read_csv(books_path)
    total_catalog_books = len(books_df)
    log_evaluation_message(f"Catalog book universe: {total_catalog_books:,} clean books.", log_callback=log_callback)

    # Load model artifacts
    log_evaluation_message("Loading trained model artifacts (Popularity, Collaborative, Content, Hybrid)...", log_callback=log_callback)
    pop_model = joblib.load(POPULARITY_MODEL_PATH)
    collab_model = joblib.load(COLLABORATIVE_MODEL_PATH)

    from src.content_model import recommend_similar_books
    from src.hybrid_model import recommend_for_user as get_hybrid_rec

    # Popularity top K
    pop_recommendations = []
    for bid in pop_model["bookId"].head(top_k):
        pop_recommendations.append(str(bid))

    # Holdout last interaction per user
    log_evaluation_message("Splitting per-user holdout test sets to prevent data leakage...", log_callback=log_callback)
    test_ratings = ratings_df.groupby("user_id").tail(1)

    model_names = ["Popularity-Based", "Content-Based", "Collaborative Filtering", "Hybrid"]
    metric_accumulators = {}
    coverage_sets = {}

    for name in model_names:
        metric_accumulators[name] = {
            "precision": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "hit_rate": 0.0
        }
        coverage_sets[name] = set()

    evaluated_users_count = 0
    grouped_test = test_ratings.groupby("user_id")

    log_evaluation_message(
        f"Commencing ranking evaluations for up to {max_evaluation_users:,} eligible test users (Top-K={top_k})...",
        log_callback=log_callback
    )

    loop_start_time = time.time()

    for uid_val, u_df in grouped_test:
        if evaluated_users_count >= max_evaluation_users:
            break

        pos_ratings = u_df[u_df["rating"] >= 4.0]
        if pos_ratings.empty:
            continue

        relevant_bids = []
        for b_val in pos_ratings["book_id"]:
            relevant_bids.append(str(b_val))

        target_user = str(uid_val)
        user_to_idx = collab_model["user_to_index"]
        if target_user not in user_to_idx:
            continue

        user_history = ratings_df[ratings_df["user_id"] == target_user]
        rated_set = set(user_history["book_id"].unique())

        # 1. Collaborative recommendations
        user_idx = user_to_idx[target_user]
        user_factor = collab_model["user_factors"][user_idx]
        book_factors = collab_model["book_factors"]
        user_scores = np.dot(user_factor, book_factors.T)
        sorted_indices = np.argsort(user_scores)[::-1]

        collab_recs = []
        for idx in sorted_indices:
            candidate_id = str(collab_model["book_ids"][idx])
            if candidate_id not in rated_set:
                collab_recs.append(candidate_id)
            if len(collab_recs) >= top_k:
                break

        # 2. Content recommendations
        content_recs = []
        train_ratings = user_history[~user_history.index.isin(u_df.index)]
        if not train_ratings.empty:
            top_train_book = str(train_ratings.sort_values("rating", ascending=False).iloc[0]["book_id"])
            try:
                content_results = recommend_similar_books(top_train_book, number_of_books=top_k)
                for rec_item in content_results:
                    content_recs.append(str(rec_item["bookId"]))
            except Exception:
                content_recs = []

        # 3. Hybrid recommendations
        hybrid_recs = []
        try:
            hybrid_results = get_hybrid_rec(target_user, number_of_books=top_k)
            for rec_item in hybrid_results:
                hybrid_recs.append(str(rec_item["bookId"]))
        except Exception:
            hybrid_recs = []

        recommendations_map = {
            "Popularity-Based": pop_recommendations,
            "Content-Based": content_recs,
            "Collaborative Filtering": collab_recs,
            "Hybrid": hybrid_recs
        }

        for model_name, rec_list in recommendations_map.items():
            p_val = compute_precision_at_k(rec_list, relevant_bids, top_k)
            r_val = compute_recall_at_k(rec_list, relevant_bids, top_k)
            n_val = compute_ndcg_at_k(rec_list, relevant_bids, top_k)
            h_val = compute_hit_rate_at_k(rec_list, relevant_bids, top_k)

            metric_accumulators[model_name]["precision"] += p_val
            metric_accumulators[model_name]["recall"] += r_val
            metric_accumulators[model_name]["ndcg"] += n_val
            metric_accumulators[model_name]["hit_rate"] += h_val

            for item_id in rec_list:
                coverage_sets[model_name].add(item_id)

        evaluated_users_count += 1

        # Real-time progress and timing logs
        should_log_progress = False
        if evaluated_users_count == 1:
            should_log_progress = True
        elif evaluated_users_count % 25 == 0:
            should_log_progress = True
        elif evaluated_users_count == max_evaluation_users:
            should_log_progress = True

        if should_log_progress:
            elapsed_seconds = time.time() - loop_start_time
            if elapsed_seconds > 0:
                users_per_sec = evaluated_users_count / elapsed_seconds
                remaining_users = max_evaluation_users - evaluated_users_count
                eta_seconds = remaining_users / users_per_sec
            else:
                users_per_sec = 0.0
                eta_seconds = 0.0

            percent_done = (evaluated_users_count / max_evaluation_users) * 100.0
            hybrid_running_hit_rate = metric_accumulators["Hybrid"]["hit_rate"] / evaluated_users_count

            log_evaluation_message(
                f"Progress: {evaluated_users_count}/{max_evaluation_users} users ({percent_done:.1f}%) | "
                f"Elapsed: {elapsed_seconds:.1f}s | Speed: {users_per_sec:.1f} users/s | "
                f"ETA: {eta_seconds:.1f}s | Hybrid HitRate@10: {hybrid_running_hit_rate:.3f}",
                log_callback=log_callback
            )

    if evaluated_users_count == 0:
        log_evaluation_message("Warning: No test users were evaluated.", log_callback=log_callback)
        return pd.DataFrame()

    total_evaluation_time = time.time() - evaluation_start_timestamp

    results_rows = []
    for model_name in model_names:
        avg_precision = metric_accumulators[model_name]["precision"] / evaluated_users_count
        avg_recall = metric_accumulators[model_name]["recall"] / evaluated_users_count
        avg_ndcg = metric_accumulators[model_name]["ndcg"] / evaluated_users_count
        avg_hit_rate = metric_accumulators[model_name]["hit_rate"] / evaluated_users_count

        if total_catalog_books > 0:
            coverage_score = len(coverage_sets[model_name]) / total_catalog_books
        else:
            coverage_score = 0.0

        results_rows.append({
            "model": model_name,
            "precision@10": round(float(avg_precision), 4),
            "recall@10": round(float(avg_recall), 4),
            "ndcg@10": round(float(avg_ndcg), 4),
            "hit_rate@10": round(float(avg_hit_rate), 4),
            "coverage": round(float(coverage_score), 4),
            "users_evaluated": evaluated_users_count
        })

    summary_df = pd.DataFrame(results_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    log_evaluation_message(
        f"Evaluation finished successfully in {total_evaluation_time:.2f} seconds "
        f"({evaluated_users_count} users evaluated, avg {total_evaluation_time / evaluated_users_count:.3f}s per user).",
        log_callback=log_callback
    )
    log_evaluation_message(f"Results saved to: {output_path}", log_callback=log_callback)

    print("\n" + "=" * 65)
    print("FINAL MODEL EVALUATION RESULTS (Top-10 Cutoff)")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65 + "\n")

    return summary_df


if __name__ == "__main__":
    evaluate_all_models()
