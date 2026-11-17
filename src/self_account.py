import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"
CONTENT_MODEL_PATH = PROJECT_ROOT / "models" / "content_model.pkl"
COLLABORATIVE_MODEL_PATH = PROJECT_ROOT / "models" / "collaborative_model.pkl"
POPULARITY_MODEL_PATH = PROJECT_ROOT / "models" / "popularity_model.pkl"
ACCOUNTS_PATH = PROJECT_ROOT / "data" / "processed" / "self_accounts.json"


def load_accounts() -> Dict[str, Dict[str, Any]]:
    if not ACCOUNTS_PATH.exists():
        return {}
    with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_accounts(accounts: Dict[str, Dict[str, Any]]) -> None:
    ACCOUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCOUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=2, ensure_ascii=False)


def search_books(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    books_df = pd.read_csv(CLEAN_BOOKS_PATH)
    books_df["bookId"] = books_df["bookId"].astype(str)

    clean_query = query.strip().lower()
    title_mask = books_df["title"].str.lower().str.contains(clean_query, regex=False, na=False)
    author_mask = books_df["author"].str.lower().str.contains(clean_query, regex=False, na=False)
    content_mask = books_df["content"].str.lower().str.contains(clean_query, regex=False, na=False)

    matches = books_df[title_mask | author_mask | content_mask].head(limit)

    results = []
    for _, row in matches.iterrows():
        results.append({
            "bookId": str(row["bookId"]),
            "title": str(row["title"]),
            "author": str(row["author"]),
            "rating": float(row.get("rating", 0.0)),
            "numRatings": int(row.get("numRatings", 0)),
            "genres": str(row.get("genres", "")),
            "image_url": str(row.get("image_url", ""))
        })
    return results


def create_account(account_id: str, name: str) -> Dict[str, Any]:
    accounts = load_accounts()
    if account_id in accounts:
        raise ValueError(f"Account '{account_id}' already exists")

    account = {
        "account_id": account_id,
        "name": name,
        "ratings": {},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    accounts[account_id] = account
    save_accounts(accounts)
    return account


def add_rating(account_id: str, book_id: str, rating: float) -> Dict[str, Any]:
    accounts = load_accounts()
    if account_id not in accounts:
        raise ValueError(f"Account '{account_id}' not found")

    books_df = pd.read_csv(CLEAN_BOOKS_PATH)
    books_df["bookId"] = books_df["bookId"].astype(str)
    if book_id not in books_df["bookId"].values:
        raise ValueError(f"Book ID '{book_id}' not found in catalog")

    accounts[account_id]["ratings"][book_id] = float(rating)
    accounts[account_id]["updated_at"] = datetime.now().isoformat()
    save_accounts(accounts)
    return accounts[account_id]


def remove_rating(account_id: str, book_id: str) -> Dict[str, Any]:
    accounts = load_accounts()
    if account_id not in accounts:
        raise ValueError(f"Account '{account_id}' not found")

    if book_id in accounts[account_id]["ratings"]:
        del accounts[account_id]["ratings"][book_id]
        accounts[account_id]["updated_at"] = datetime.now().isoformat()
        save_accounts(accounts)
    return accounts[account_id]


def get_account(account_id: str) -> Optional[Dict[str, Any]]:
    accounts = load_accounts()
    return accounts.get(account_id)


def list_accounts() -> List[Dict[str, Any]]:
    accounts = load_accounts()
    result = []
    for acc_id, acc_data in accounts.items():
        result.append({
            "account_id": acc_id,
            "name": acc_data["name"],
            "ratings_count": len(acc_data["ratings"]),
            "created_at": acc_data["created_at"],
            "updated_at": acc_data["updated_at"]
        })
    return result


def delete_account(account_id: str) -> None:
    accounts = load_accounts()
    if account_id in accounts:
        del accounts[account_id]
        save_accounts(accounts)


def normalize_scores_array(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    min_val = float(scores.min())
    max_val = float(scores.max())
    if max_val == min_val:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_val) / (max_val - min_val)


def get_collaborative_scores_for_account(account_id: str) -> Dict[str, float]:
    accounts = load_accounts()
    if account_id not in accounts:
        raise ValueError(f"Account '{account_id}' not found")

    ratings = accounts[account_id]["ratings"]
    if not ratings:
        return {}

    collab_model = joblib.load(COLLABORATIVE_MODEL_PATH)
    book_ids = collab_model["book_ids"]
    book_factors = collab_model["book_factors"]
    book_to_index = collab_model["book_to_index"]

    user_vector = np.zeros_like(book_factors[0], dtype=float)
    valid_rated_count = 0

    for book_id, rating in ratings.items():
        if book_id in book_to_index:
            book_idx = book_to_index[book_id]
            user_vector += book_factors[book_idx] * rating
            valid_rated_count += 1

    if valid_rated_count > 0:
        user_vector = user_vector / valid_rated_count
    else:
        return {}

    scores = np.dot(user_vector, book_factors.T)
    result = {}
    for idx, bid in enumerate(book_ids):
        result[str(bid)] = float(scores[idx])
    return result


def get_content_scores_for_account(account_id: str) -> Dict[str, float]:
    accounts = load_accounts()
    if account_id not in accounts:
        raise ValueError(f"Account '{account_id}' not found")

    ratings = accounts[account_id]["ratings"]
    liked_book_ids = [bid for bid, r in ratings.items() if r >= 4.0]

    if not liked_book_ids:
        return {}

    content_model = joblib.load(CONTENT_MODEL_PATH)
    content_books = content_model["books"]
    tfidf_matrix = content_model["tfidf_matrix"]

    liked_indices = []
    for idx, bid in enumerate(content_books["bookId"]):
        if str(bid) in liked_book_ids:
            liked_indices.append(idx)

    if not liked_indices:
        return {}

    user_profile = tfidf_matrix[liked_indices].mean(axis=0)
    user_profile = np.asarray(user_profile)

    similarities = cosine_similarity(user_profile, tfidf_matrix).ravel()

    result = {}
    for idx, bid in enumerate(content_books["bookId"]):
        result[str(bid)] = float(similarities[idx])
    return result


def get_popularity_scores() -> Dict[str, float]:
    pop_df = joblib.load(POPULARITY_MODEL_PATH)
    result = {}
    for _, row in pop_df.iterrows():
        result[str(row["bookId"])] = float(row.get("popularity_score", 0.0))
    return result


def recommend_for_account(
    account_id: str,
    number_of_books: int = 10,
    max_books_per_author: int = 2,
    collaborative_weight: float = 0.35,
    content_weight: float = 0.30,
    readability_weight: float = 0.15,
    popularity_weight: float = 0.20
) -> List[Dict[str, Any]]:
    accounts = load_accounts()
    if account_id not in accounts:
        raise ValueError(f"Account '{account_id}' not found")

    ratings = accounts[account_id]["ratings"]
    rated_book_ids = set(ratings.keys())

    collab_scores = get_collaborative_scores_for_account(account_id)
    content_scores = get_content_scores_for_account(account_id)
    pop_scores = get_popularity_scores()

    books_df = pd.read_csv(CLEAN_BOOKS_PATH)
    books_df["bookId"] = books_df["bookId"].astype(str)

    aligned_collab = []
    aligned_content = []
    aligned_pop = []
    aligned_difficulty = []

    for _, row in books_df.iterrows():
        bid = str(row["bookId"])
        aligned_collab.append(collab_scores.get(bid, 0.0))
        aligned_content.append(content_scores.get(bid, 0.0))
        aligned_pop.append(pop_scores.get(bid, 0.0))
        diff = float(row.get("difficulty", 0.5))
        aligned_difficulty.append(diff)

    collab_arr = normalize_scores_array(np.array(aligned_collab))
    content_arr = normalize_scores_array(np.array(aligned_content))
    pop_arr = normalize_scores_array(np.array(aligned_pop))
    diff_arr = np.array(aligned_difficulty)

    liked_difficulties = []
    for bid, r in ratings.items():
        row = books_df[books_df["bookId"] == bid]
        if not row.empty:
            diff = float(row.iloc[0].get("difficulty", 0.5))
            liked_difficulties.append(diff)

    if liked_difficulties:
        preferred_diff = float(np.mean(liked_difficulties))
    else:
        preferred_diff = 0.5

    readability_arr = 1.0 - np.abs(diff_arr - preferred_diff)
    readability_arr = np.clip(readability_arr, 0.0, 1.0)

    total_weight = collaborative_weight + content_weight + readability_weight + popularity_weight
    if total_weight <= 0:
        total_weight = 1.0

    hybrid_scores = (
        (collaborative_weight / total_weight) * collab_arr
        + (content_weight / total_weight) * content_arr
        + (readability_weight / total_weight) * readability_arr
        + (popularity_weight / total_weight) * pop_arr
    )

    sorted_indices = np.argsort(hybrid_scores)[::-1]

    liked_authors = set()
    for bid, r in ratings.items():
        if r >= 4.0:
            row = books_df[books_df["bookId"] == bid]
            if not row.empty:
                liked_authors.add(str(row.iloc[0]["author"]).strip())

    results = []
    author_counts = {}

    for idx in sorted_indices:
        row = books_df.iloc[idx]
        bid = str(row["bookId"])

        if bid in rated_book_ids:
            continue

        author = str(row["author"]).strip()
        if author_counts.get(author, 0) >= max_books_per_author:
            continue

        matched_author = author in liked_authors
        c_score = float(collab_arr[idx])
        t_score = float(content_arr[idx])
        r_score = float(readability_arr[idx])
        p_score = float(pop_arr[idx])
        h_score = float(hybrid_scores[idx])

        if matched_author:
            reason = f"Matches your interest in books by {author}"
        elif c_score > 0.6:
            reason = "Similar to your reading patterns"
        elif t_score > 0.5:
            reason = "Similar themes to books you rated 4+ stars"
        elif r_score > 0.8:
            reason = "Reading difficulty matches your preferences"
        else:
            reason = "Popular title aligned with your tastes"

        results.append({
            "bookId": bid,
            "title": str(row["title"]),
            "author": author,
            "hybrid_score": round(h_score, 4),
            "collaborative_score": round(c_score, 4),
            "content_score": round(t_score, 4),
            "readability_score": round(r_score, 4),
            "popularity_score": round(p_score, 4),
            "reason": reason,
            "matched_author": matched_author,
            "image_url": str(row.get("image_url", ""))
        })
        author_counts[author] = author_counts.get(author, 0) + 1

        if len(results) >= number_of_books:
            break

    return results


if __name__ == "__main__":
    demo_acc = create_account("demo_user", "Demo User")
    print(f"Created account: {demo_acc}")
