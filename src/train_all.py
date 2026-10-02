"""
BookMind End-to-End Training Pipeline Orchestrator.

This script runs the entire BookMind preparation, training, evaluation, and reporting pipeline:
1. data_cleaning.py
2. prepare_ratings.py
3. popularity_model.py
4. content_model.py
5. collaborative_model.py
6. hybrid_model.py
7. evaluation.py
8. generate_report_graphs.py

It halts immediately if any stage raises an exception, writes timestamped progress logs to
`data/processed/training.log`, supports real-time logging callbacks for web dashboard panels,
and saves model metadata to `models/model_metadata.json`.

Note: Strictly functional implementation using only `def` functions (no `class`).
"""

import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Callable, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data_cleaning import clean_books_data
from src.prepare_ratings import prepare_ratings_data
from src.popularity_model import train_popularity_model
from src.content_model import train_content_model
from src.collaborative_model import train_collaborative_model
from src.hybrid_model import train_hybrid_model
from src.evaluation import evaluate_all_models
from src.generate_report_graphs import generate_all_report_graphs

METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"
USER_RATINGS_PATH = PROJECT_ROOT / "data" / "processed" / "user_ratings.csv"
TRAINING_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "training.log"



def append_training_log(
    message_text: str,
    log_file_path: Path = TRAINING_LOG_PATH,
    log_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Format a timestamped log entry, write it to file, print to stdout,
    and forward it to an optional UI callback.

    Parameters
    ----------
    message_text : str
        The status or error text to record.
    log_file_path : Path
        Target log file path.
    log_callback : Optional[Callable[[str], None]]
        Optional streaming callback for live UI dashboards.

    Returns
    -------
    str
        Formatted log message with timestamp.
    """
    timestamp_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_entry = f"[{timestamp_string}] {message_text}"

    print(formatted_entry)

    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file_path, "a", encoding="utf-8") as file_handle:
            file_handle.write(formatted_entry + "\n")
    except Exception:
        pass

    if log_callback is not None:
        try:
            log_callback(formatted_entry)
        except Exception:
            pass

    return formatted_entry


def save_model_metadata(
    metadata_file_path: Path = METADATA_PATH,
    books_file_path: Path = CLEAN_BOOKS_PATH,
    ratings_file_path: Path = USER_RATINGS_PATH,
    log_callback: Optional[Callable[[str], None]] = None
) -> None:
    """
    Generate and save JSON metadata describing the trained model bundle.

    Parameters
    ----------
    metadata_file_path : Path
        Destination path for `model_metadata.json`.
    books_file_path : Path
        Path to `books_clean.csv`.
    ratings_file_path : Path
        Path to `user_ratings.csv`.
    log_callback : Optional[Callable[[str], None]]
        Optional UI logging callback.
    """
    books_count = 0
    if books_file_path.exists():
        books_df = pd.read_csv(books_file_path)
        books_count = len(books_df)

    users_count = 0
    ratings_count = 0
    if ratings_file_path.exists():
        ratings_df = pd.read_csv(ratings_file_path)
        ratings_count = len(ratings_df)
        users_count = ratings_df["user_id"].nunique()

    metadata_payload = {
        "model_version": "1.0.0",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "books_used": int(books_count),
        "users_used": int(users_count),
        "ratings_used": int(ratings_count),
        "hybrid_weights": {
            "collaborative": 0.40,
            "content": 0.35,
            "popularity": 0.25
        }
    }

    metadata_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file_path, "w", encoding="utf-8") as file_handle:
        json.dump(metadata_payload, file_handle, indent=2)

    append_training_log(
        f"Model metadata saved to: {metadata_file_path} (Books: {books_count:,}, Users: {users_count:,}, Ratings: {ratings_count:,})",
        log_callback=log_callback
    )


def execute_single_stage(
    stage_identifier: str,
    log_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Execute an individual stage of the BookMind training pipeline.

    Parameters
    ----------
    stage_identifier : str
        Stage name (e.g. 'clean', 'ratings', 'popularity', 'content', 'collaborative', 'hybrid', 'eval', 'graphs').
    log_callback : Optional[Callable[[str], None]]
        Optional UI callback for streaming progress logs.

    Returns
    -------
    bool
        True if the stage completed successfully, False otherwise.
    """
    stage_key = stage_identifier.strip().lower()

    if stage_key in ["1", "clean", "data_cleaning"]:
        append_training_log(">>> Starting Stage 1: Data Cleaning (data_cleaning.py)...", log_callback=log_callback)
        clean_books_data()
        append_training_log("<<< Stage 1 Complete: books_clean.csv successfully generated.", log_callback=log_callback)
        return True

    elif stage_key in ["2", "ratings", "prepare_ratings"]:
        append_training_log(">>> Starting Stage 2: Ratings Preparation (prepare_ratings.py)...", log_callback=log_callback)
        prepare_ratings_data()
        append_training_log("<<< Stage 2 Complete: user_ratings.csv successfully generated.", log_callback=log_callback)
        return True

    elif stage_key in ["3", "popularity", "popularity_model"]:
        append_training_log(">>> Starting Stage 3: Training Popularity Model (popularity_model.py)...", log_callback=log_callback)
        train_popularity_model()
        append_training_log("<<< Stage 3 Complete: popularity_model.pkl saved.", log_callback=log_callback)
        return True

    elif stage_key in ["4", "content", "content_model"]:
        append_training_log(">>> Starting Stage 4: Training Content-Based Model (content_model.py)...", log_callback=log_callback)
        train_content_model()
        append_training_log("<<< Stage 4 Complete: content_model.pkl saved.", log_callback=log_callback)
        return True

    elif stage_key in ["5", "collaborative", "collaborative_model"]:
        append_training_log(">>> Starting Stage 5: Training Collaborative Model (collaborative_model.py)...", log_callback=log_callback)
        train_collaborative_model()
        append_training_log("<<< Stage 5 Complete: collaborative_model.pkl saved.", log_callback=log_callback)
        return True

    elif stage_key in ["6", "hybrid", "hybrid_model"]:
        append_training_log(">>> Starting Stage 6: Training Hybrid Ensemble Model (hybrid_model.py)...", log_callback=log_callback)
        train_hybrid_model()
        append_training_log("<<< Stage 6 Complete: hybrid_model.pkl saved.", log_callback=log_callback)
        return True

    elif stage_key in ["7", "eval", "evaluation"]:
        append_training_log(">>> Starting Stage 7: Running Offline Evaluation (evaluation.py)...", log_callback=log_callback)
        evaluate_all_models(log_callback=log_callback)
        append_training_log("<<< Stage 7 Complete: evaluation_results.csv saved.", log_callback=log_callback)
        return True


    elif stage_key in ["8", "graphs", "generate_report_graphs"]:
        append_training_log(">>> Starting Stage 8: Generating Report Visualizations (generate_report_graphs.py)...", log_callback=log_callback)
        generate_all_report_graphs()
        append_training_log("<<< Stage 8 Complete: 9 report figures generated in reports/.", log_callback=log_callback)
        return True

    else:
        append_training_log(f"Unknown pipeline stage: '{stage_identifier}'", log_callback=log_callback)
        return False


def run_full_training_pipeline(
    log_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Execute all 8 pipeline stages sequentially and stop immediately on failure.

    Parameters
    ----------
    log_callback : Optional[Callable[[str], None]]
        Optional streaming log callback function.

    Returns
    -------
    bool
        True if all stages completed successfully, False otherwise.
    """
    append_training_log("=" * 65, log_callback=log_callback)
    append_training_log("BookMind Pipeline: Training started", log_callback=log_callback)
    append_training_log("=" * 65, log_callback=log_callback)

    stages_plan = [
        ("1", "Data Cleaning"),
        ("2", "Ratings Preparation"),
        ("3", "Popularity Model"),
        ("4", "Content-Based Model"),
        ("5", "Collaborative Filtering"),
        ("6", "Hybrid Ensemble"),
        ("7", "Offline Evaluation"),
        ("8", "Report Graphs Generation")
    ]

    for stage_index, (stage_code, stage_label) in enumerate(stages_plan, start=1):
        append_training_log(f"\n[Step {stage_index}/8] Running {stage_label}...", log_callback=log_callback)
        try:
            success = execute_single_stage(stage_code, log_callback=log_callback)
            if not success:
                append_training_log(f"ERROR: Stage {stage_label} failed.", log_callback=log_callback)
                return False
        except Exception as error_exception:
            append_training_log(f"FATAL ERROR in {stage_label}: {str(error_exception)}", log_callback=log_callback)
            return False

    # Save metadata JSON
    save_model_metadata(log_callback=log_callback)

    append_training_log("\n" + "=" * 65, log_callback=log_callback)
    append_training_log("Training progress: 100% Completed", log_callback=log_callback)
    append_training_log("Training completed successfully!", log_callback=log_callback)
    append_training_log("All models saved in models/", log_callback=log_callback)
    append_training_log("Evaluation metrics saved in data/processed/evaluation_results.csv", log_callback=log_callback)
    append_training_log("Report charts generated in reports/", log_callback=log_callback)
    append_training_log("=" * 65, log_callback=log_callback)

    return True


if __name__ == "__main__":
    pipeline_success = run_full_training_pipeline()
    if not pipeline_success:
        sys.exit(1)
