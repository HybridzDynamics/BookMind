
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List
import json
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.popularity_model import get_popular_books
from src.content_model import recommend_similar_books
from src.collaborative_model import recommend_for_user as get_collaborative_recommendations
from src.hybrid_model import recommend_for_user as get_hybrid_recommendations
from src.self_account import (
    search_books,
    create_account,
    add_rating,
    remove_rating,
    get_account,
    list_accounts,
    delete_account,
    recommend_for_account as get_self_account_recommendations
)

METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DEMO_DIR = PROJECT_ROOT / "demo"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_popular_books(number_of_books=1)
        recommend_similar_books(book_id="1", number_of_books=1)
        get_collaborative_recommendations(user_id="1", number_of_books=1)
        get_hybrid_recommendations(user_id="1", number_of_books=1)
    except Exception as warmup_error:
        print(f"Model warmup skipped: {warmup_error}")
    yield


app = FastAPI(
    title="BookMind Recommendation API",
    description="Machine Learning Book Recommendation API supporting Popularity, Content, Collaborative, and Hybrid models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


def validate_limit_parameter(limit_value: int) -> None:
    if limit_value < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'limit' must be greater than or equal to 1."
        )
    elif limit_value > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'limit' must be less than or equal to 100."
        )


@app.get("/", tags=["General"], summary="Service overview")
def get_service_root() -> Dict[str, str]:
    result = {
        "service": "BookMind",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "demo": "/demo/"
    }
    return result


@app.get("/health", tags=["General"], summary="Health check probe")
def get_health_status() -> Dict[str, str]:
    result = {
        "status": "healthy"
    }
    return result


@app.get("/stats", tags=["General"], summary="Dataset and model statistics")
def get_statistics() -> Dict[str, Any]:
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            return {
                "books": metadata.get("books_used", 0),
                "users": metadata.get("users_used", 0),
                "ratings": metadata.get("ratings_used", 0),
                "models": 4,
                "hybrid_weights": metadata.get("hybrid_weights", {}),
                "trained_at": metadata.get("trained_at", "unknown"),
                "model_version": metadata.get("model_version", "1.0.0")
            }
        except Exception:
            pass

    return {
        "books": 0,
        "users": 0,
        "ratings": 0,
        "models": 4,
        "hybrid_weights": {
            "collaborative": 0.35,
            "content": 0.30,
            "readability": 0.15,
            "popularity": 0.20
        },
        "trained_at": "unknown",
        "model_version": "1.0.0"
    }


@app.get("/recommendations/popular", tags=["Recommendations"], summary="Get popular books")
def get_popular_recommendations(limit: int = 10) -> List[Dict[str, Any]]:
    validate_limit_parameter(limit)

    try:
        popular_dataframe = get_popular_books(number_of_books=limit)
        results_list = []
        for _, row in popular_dataframe.iterrows():
            record = {
                "bookId": str(row["bookId"]),
                "title": str(row["title"]),
                "author": str(row["author"]),
                "rating": float(row.get("rating", 0.0)),
                "numRatings": int(row.get("numRatings", 0)),
                "popularity_score": float(row.get("popularity_score", 0.0)),
                "reason": "Highly rated and widely read by BookMind readers"
            }
            results_list.append(record)
        return results_list
    except FileNotFoundError as file_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Popularity model file missing: {str(file_error)}"
        )
    except Exception as general_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(general_error)}"
        )


@app.get("/recommendations/content/{book_id}", tags=["Recommendations"], summary="Get content-similar books")
def get_content_recommendations(book_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    validate_limit_parameter(limit)

    try:
        similar_books = recommend_similar_books(book_id=book_id, number_of_books=limit)
        return similar_books
    except ValueError as not_found_error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(not_found_error)
        )
    except FileNotFoundError as file_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content model file missing: {str(file_error)}"
        )
    except Exception as general_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(general_error)}"
        )


@app.get("/recommendations/collaborative/{user_id}", tags=["Recommendations"], summary="Get collaborative recommendations")
def get_collaborative_recommendations_endpoint(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    validate_limit_parameter(limit)

    try:
        recommendations = get_collaborative_recommendations(user_id=user_id, number_of_books=limit)
        return recommendations
    except FileNotFoundError as file_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Collaborative model file missing: {str(file_error)}"
        )
    except Exception as general_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(general_error)}"
        )


@app.get("/recommendations/hybrid/{user_id}", tags=["Recommendations"], summary="Get hybrid recommendations")
def get_hybrid_recommendations_endpoint(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    validate_limit_parameter(limit)

    try:
        recommendations = get_hybrid_recommendations(user_id=user_id, number_of_books=limit)
        return recommendations
    except FileNotFoundError as file_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid model file missing: {str(file_error)}"
        )
    except Exception as general_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(general_error)}"
        )


@app.get("/evaluation", tags=["General"], summary="Get model evaluation metrics")
def get_evaluation_results() -> List[Dict[str, Any]]:
    import csv as _csv
    eval_csv_path = PROJECT_ROOT / "data" / "processed" / "evaluation_results.csv"
    if not eval_csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation results not available. Run the evaluation pipeline first."
        )
    try:
        with open(eval_csv_path, "r", encoding="utf-8", newline="") as csv_file:
            reader = _csv.DictReader(csv_file)
            rows = [dict(row) for row in reader]
        return rows
    except Exception as eval_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read evaluation results: {str(eval_error)}"
        )


@app.get("/self-accounts", tags=["Self Account"], summary="List all self accounts")
def get_self_accounts() -> List[Dict[str, Any]]:
    try:
        return list_accounts()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list accounts: {str(e)}"
        )


@app.post("/self-accounts", tags=["Self Account"], summary="Create a new self account")
def create_self_account(account_id: str, name: str) -> Dict[str, Any]:
    try:
        return create_account(account_id, name)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create account: {str(e)}"
        )


@app.get("/self-accounts/{account_id}", tags=["Self Account"], summary="Get a self account by ID")
def get_self_account(account_id: str) -> Dict[str, Any]:
    try:
        account = get_account(account_id)
        if account is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account '{account_id}' not found"
            )
        return account
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get account: {str(e)}"
        )


@app.delete("/self-accounts/{account_id}", tags=["Self Account"], summary="Delete a self account")
def delete_self_account(account_id: str) -> Dict[str, str]:
    try:
        delete_account(account_id)
        return {"message": f"Account '{account_id}' deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete account: {str(e)}"
        )


@app.post("/self-accounts/{account_id}/ratings", tags=["Self Account"], summary="Add a rating to a self account")
def add_self_account_rating(account_id: str, book_id: str, rating: float) -> Dict[str, Any]:
    if not (1.0 <= rating <= 5.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rating must be between 1.0 and 5.0"
        )
    try:
        return add_rating(account_id, book_id, rating)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add rating: {str(e)}"
        )


@app.delete("/self-accounts/{account_id}/ratings/{book_id}", tags=["Self Account"], summary="Remove a rating from a self account")
def remove_self_account_rating(account_id: str, book_id: str) -> Dict[str, Any]:
    try:
        return remove_rating(account_id, book_id)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove rating: {str(e)}"
        )


@app.get("/self-accounts/{account_id}/recommendations", tags=["Self Account"], summary="Get recommendations for a self account")
def get_self_account_recommendations_endpoint(
    account_id: str,
    limit: int = 10,
    collaborative_weight: float = 0.35,
    content_weight: float = 0.30,
    readability_weight: float = 0.15,
    popularity_weight: float = 0.20
) -> List[Dict[str, Any]]:
    validate_limit_parameter(limit)
    try:
        return get_self_account_recommendations(
            account_id=account_id,
            number_of_books=limit,
            collaborative_weight=collaborative_weight,
            content_weight=content_weight,
            readability_weight=readability_weight,
            popularity_weight=popularity_weight
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.get("/books/search", tags=["Self Account"], summary="Search books by title, author, or keywords")
def search_books_endpoint(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    if not query or len(query.strip()) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be at least 2 characters"
        )
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 100"
        )
    try:
        return search_books(query, limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(DEMO_DIR), html=True), name="demo")
