import sys
from datetime import datetime
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.hybrid_model import (
    DEFAULT_COLLABORATIVE_WEIGHT,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_POPULARITY_WEIGHT,
    DEFAULT_READABILITY_WEIGHT,
)

USER_RATINGS_PATH = PROJECT_ROOT / "data" / "processed" / "user_ratings.csv"
CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"
POPULARITY_MODEL_PATH = PROJECT_ROOT / "models" / "popularity_model.pkl"
CONTENT_MODEL_PATH = PROJECT_ROOT / "models" / "content_model.pkl"
EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "evaluation_results.csv"
TRAINING_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "training.log"

SVD_COMPONENTS = 50
RELEVANT_RATING_THRESHOLD = 4.0


def log_evaluation_message(message_content, log_file_path=TRAINING_LOG_PATH, log_callback=None):
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


def compute_precision_at_k(recommended_items, relevant_items, k=10):
    if k <= 0:
        return 0.0
    top_k_recommendations = recommended_items[:k]
    relevant_items_set = set(relevant_items)
    hit_count = 0
    for item in top_k_recommendations:
        if item in relevant_items_set:
            hit_count += 1
    return float(hit_count / k)


def compute_recall_at_k(recommended_items, relevant_items, k=10):
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


def compute_ndcg_at_k(recommended_items, relevant_items, k=10):
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


def compute_hit_rate_at_k(recommended_items, relevant_items, k=10):
    top_k_recommendations = recommended_items[:k]
    relevant_items_set = set(relevant_items)
    for item in top_k_recommendations:
        if item in relevant_items_set:
            return 1.0
    return 0.0


def build_train_interaction_model(train_ratings, n_components=SVD_COMPONENTS):
    train_grouped = train_ratings.groupby(["user_id", "book_id"], as_index=False)["rating"].mean()
    unique_users = train_grouped["user_id"].unique().tolist()
    unique_books = train_grouped["book_id"].unique().tolist()

    user_to_index = {}
    for user_index, user_id in enumerate(unique_users):
        user_to_index[user_id] = user_index

    book_to_index = {}
    for book_index, book_id in enumerate(unique_books):
        book_to_index[book_id] = book_index

    row_indices = train_grouped["user_id"].map(user_to_index)
    col_indices = train_grouped["book_id"].map(book_to_index)
    rating_values = train_grouped["rating"].astype(np.float32)

    interaction_matrix = csr_matrix(
        (rating_values, (row_indices, col_indices)),
        shape=(len(unique_users), len(unique_books))
    )

    minimum_dim = min(len(unique_users), len(unique_books))
    components_count = n_components
    if components_count >= minimum_dim:
        components_count = minimum_dim - 1

    svd_model = TruncatedSVD(n_components=components_count, random_state=42)
    user_factors = svd_model.fit_transform(interaction_matrix).astype(np.float32)
    book_factors = svd_model.components_.T.astype(np.float32)

    return {
        "user_factors": user_factors,
        "book_factors": book_factors,
        "user_to_index": user_to_index,
        "book_ids": unique_books,
        "user_item_matrix": interaction_matrix.astype(np.float32),
    }


def evaluate_all_models(
    ratings_path=USER_RATINGS_PATH,
    books_path=CLEAN_BOOKS_PATH,
    output_path=EVALUATION_RESULTS_PATH,
    top_k=10,
    max_evaluation_users=500,
    log_callback=None
):
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

    test_ratings = ratings_df.groupby("user_id").tail(1)
    train_ratings = ratings_df.drop(index=test_ratings.index)
    log_evaluation_message(
        f"Holdout split: {len(test_ratings):,} test interactions, {len(train_ratings):,} training interactions.",
        log_callback=log_callback
    )

    log_evaluation_message("Training evaluation-only SVD model on training split (no leakage)...", log_callback=log_callback)
    train_model = build_train_interaction_model(train_ratings)
    train_user_to_index = train_model["user_to_index"]
    train_book_ids = train_model["book_ids"]
    log_evaluation_message(
        f"Train matrix: {len(train_user_to_index):,} users x {len(train_book_ids):,} books.",
        log_callback=log_callback
    )

    log_evaluation_message("Loading popularity and content model artifacts...", log_callback=log_callback)
    pop_model = joblib.load(POPULARITY_MODEL_PATH)
    content_model = joblib.load(CONTENT_MODEL_PATH)

    book_position = {}
    for pos, bid in enumerate(books_df["bookId"].astype(str)):
        book_position[bid] = pos

    difficulty_lookup = {}
    if "difficulty" in books_df.columns:
        for bid, diff_val in zip(books_df["bookId"].astype(str), books_df["difficulty"]):
            difficulty_lookup[str(bid)] = float(diff_val)

    pop_lookup = {}
    for bid, score_val in zip(pop_model["bookId"].astype(str), pop_model["popularity_score"]):
        pop_lookup[str(bid)] = float(score_val)
    raw_pop_array = np.array([pop_lookup.get(str(b), 0.0) for b in books_df["bookId"].astype(str)], dtype=float)
    pop_min = float(raw_pop_array.min())
    pop_max = float(raw_pop_array.max())
    if pop_max > pop_min:
        normalized_popularity = (raw_pop_array - pop_min) / (pop_max - pop_min)
    else:
        normalized_popularity = np.zeros_like(raw_pop_array)

    content_books = content_model["books"]
    content_book_position = {}
    for pos, bid in enumerate(content_books["bookId"].astype(str)):
        content_book_position[str(bid)] = pos
    tfidf_matrix = content_model["tfidf_matrix"]
    nn_model = content_model["nearest_neighbors"]

    user_histories = {}
    for uid_val, grp in train_ratings.groupby("user_id", sort=False):
        grp_bids = grp["book_id"].astype(str)
        rated_set = set(grp_bids)
        liked_set = set(grp.loc[grp["rating"] >= RELEVANT_RATING_THRESHOLD, "book_id"].astype(str))
        top_book_id = str(grp.loc[grp["rating"].idxmax(), "book_id"])
        user_histories[str(uid_val)] = {
            "rated": rated_set,
            "liked": liked_set,
            "top_book": top_book_id,
        }

    model_names = ["Popularity-Based", "Content-Based", "Collaborative Filtering", "Hybrid"]
    metric_accumulators = {}
    coverage_sets = {}
    for name in model_names:
        metric_accumulators[name] = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0}
        coverage_sets[name] = set()

    evaluated_users_count = 0
    grouped_test = test_ratings.groupby("user_id")

    log_evaluation_message(
        f"Commencing ranking evaluations for up to {max_evaluation_users:,} eligible test users (Top-K={top_k})...",
        log_callback=log_callback
    )

    loop_start_time = time.time()
    weight_total = (
        DEFAULT_COLLABORATIVE_WEIGHT
        + DEFAULT_CONTENT_WEIGHT
        + DEFAULT_READABILITY_WEIGHT
        + DEFAULT_POPULARITY_WEIGHT
    )
    if weight_total <= 0:
        weight_total = 1.0

    for uid_val, u_df in grouped_test:
        if evaluated_users_count >= max_evaluation_users:
            break

        target_user = str(uid_val)
        holdout_row = u_df.iloc[0]
        holdout_book_id = str(holdout_row["book_id"])
        holdout_rating = float(holdout_row["rating"])

        if holdout_rating < RELEVANT_RATING_THRESHOLD:
            continue

        if target_user not in train_user_to_index:
            continue

        history = user_histories.get(target_user)
        if history is None:
            continue

        rated_set = history["rated"]
        liked_set = history["liked"]
        relevant_items = [holdout_book_id]

        pop_recs = []
        for bid in pop_model["bookId"].astype(str):
            if str(bid) in rated_set:
                continue
            pop_recs.append(str(bid))
            if len(pop_recs) >= top_k:
                break

        collab_recs = []
        user_idx = train_user_to_index[target_user]
        user_factor = train_model["user_factors"][user_idx]
        book_factors = train_model["book_factors"]
        collab_scores = np.dot(user_factor, book_factors.T)
        collab_order = np.argsort(collab_scores)[::-1]
        for idx in collab_order:
            candidate_id = str(train_book_ids[idx])
            if candidate_id in rated_set:
                continue
            collab_recs.append(candidate_id)
            if len(collab_recs) >= top_k:
                break

        content_recs = []
        top_train_book = history["top_book"]
        if top_train_book in content_book_position:
            target_pos = content_book_position[top_train_book]
            query_k = min(top_k * 4 + 1, len(content_books))
            distances, indices = nn_model.kneighbors(tfidf_matrix[target_pos], n_neighbors=query_k)
            for dist_val, n_idx in zip(distances[0], indices[0]):
                candidate_id = str(content_books.iloc[n_idx]["bookId"])
                if candidate_id == top_train_book or candidate_id in rated_set:
                    continue
                content_recs.append(candidate_id)
                if len(content_recs) >= top_k:
                    break

        hybrid_recs = []
        collab_lookup = {}
        for idx, bid in enumerate(train_book_ids):
            collab_lookup[str(bid)] = float(collab_scores[idx])
        collab_aligned = np.array([collab_lookup.get(str(b), 0.0) for b in books_df["bookId"].astype(str)], dtype=float)
        collab_min = float(collab_aligned.min())
        collab_max = float(collab_aligned.max())
        if collab_max > collab_min:
            collab_norm = (collab_aligned - collab_min) / (collab_max - collab_min)
        else:
            collab_norm = np.zeros_like(collab_aligned)

        content_norm = np.zeros(total_catalog_books, dtype=float)
        if len(liked_set) > 0:
            liked_positions = [content_book_position[b] for b in liked_set if b in content_book_position]
            if len(liked_positions) > 0:
                profile_sparse = tfidf_matrix[liked_positions].mean(axis=0)
                profile_vector = np.asarray(profile_sparse)
                raw_sims = cosine_similarity(profile_vector, tfidf_matrix).ravel()
                content_aligned = np.zeros(total_catalog_books, dtype=float)
                content_books_ids = content_books["bookId"].astype(str).tolist()
                for c_pos, bid in enumerate(content_books_ids):
                    bpos = book_position.get(bid)
                    if bpos is not None:
                        content_aligned[bpos] = float(raw_sims[c_pos])
                c_min = float(content_aligned.min())
                c_max = float(content_aligned.max())
                if c_max > c_min:
                    content_norm = (content_aligned - c_min) / (c_max - c_min)

        liked_diffs = [difficulty_lookup[b] for b in liked_set if b in difficulty_lookup]
        if len(liked_diffs) > 0:
            preferred_diff = float(np.mean(liked_diffs))
        else:
            all_diffs = [difficulty_lookup[b] for b in rated_set if b in difficulty_lookup]
            preferred_diff = float(np.mean(all_diffs)) if len(all_diffs) > 0 else 0.5

        if "difficulty" in books_df.columns:
            diff_array = pd.to_numeric(books_df["difficulty"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
        else:
            diff_array = np.full(total_catalog_books, 0.5, dtype=float)
        readability = np.clip(1.0 - np.abs(diff_array - preferred_diff), 0.0, 1.0)

        combined = (
            (DEFAULT_COLLABORATIVE_WEIGHT / weight_total) * collab_norm
            + (DEFAULT_CONTENT_WEIGHT / weight_total) * content_norm
            + (DEFAULT_READABILITY_WEIGHT / weight_total) * readability
            + (DEFAULT_POPULARITY_WEIGHT / weight_total) * normalized_popularity
        )
        hybrid_order = np.argsort(combined)[::-1]
        books_ids_str = books_df["bookId"].astype(str).tolist()
        for idx in hybrid_order:
            candidate_id = books_ids_str[idx]
            if candidate_id in rated_set:
                continue
            hybrid_recs.append(candidate_id)
            if len(hybrid_recs) >= top_k:
                break

        recommendations_map = {
            "Popularity-Based": pop_recs,
            "Content-Based": content_recs,
            "Collaborative Filtering": collab_recs,
            "Hybrid": hybrid_recs
        }

        for model_name, rec_list in recommendations_map.items():
            p_val = compute_precision_at_k(rec_list, relevant_items, top_k)
            r_val = compute_recall_at_k(rec_list, relevant_items, top_k)
            n_val = compute_ndcg_at_k(rec_list, relevant_items, top_k)
            h_val = compute_hit_rate_at_k(rec_list, relevant_items, top_k)

            metric_accumulators[model_name]["precision"] += p_val
            metric_accumulators[model_name]["recall"] += r_val
            metric_accumulators[model_name]["ndcg"] += n_val
            metric_accumulators[model_name]["hit_rate"] += h_val

            for item_id in rec_list:
                coverage_sets[model_name].add(item_id)

        evaluated_users_count += 1

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
