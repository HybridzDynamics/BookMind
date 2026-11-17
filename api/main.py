
import sys
from pathlib import Path
from typing import Any, Dict, List
import json
from fastapi import FastAPI, HTTPException, Query, status

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.popularity_model import get_popular_books
from src.content_model import recommend_similar_books
from src.collaborative_model import recommend_for_user as get_collaborative_recommendations
from src.hybrid_model import recommend_for_user as get_hybrid_recommendations

METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

app = FastAPI(
    title="BookMind Recommendation API",
    description="Machine Learning Book Recommendation API supporting Popularity, Content, Collaborative, and Hybrid models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
        "docs": "/docs"
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
