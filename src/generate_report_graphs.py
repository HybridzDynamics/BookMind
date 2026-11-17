from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_RATINGS_PATH = PROJECT_ROOT / "data" / "processed" / "user_ratings.csv"
POPULARITY_MODEL_PATH = PROJECT_ROOT / "models" / "popularity_model.pkl"
EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "evaluation_results.csv"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"


def save_active_plot(figure_filename: str, destination_folder: Path = REPORTS_DIRECTORY) -> Path:
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_path = destination_folder / figure_filename

    plt.tight_layout()
    plt.savefig(destination_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated report figure: {destination_path}")
    return destination_path


def plot_rating_distribution(
    ratings_path: Path = USER_RATINGS_PATH,
    reports_folder: Path = REPORTS_DIRECTORY
) -> None:
    if not ratings_path.exists():
        return

    ratings_df = pd.read_csv(ratings_path)
    ratings_series = pd.to_numeric(ratings_df["rating"], errors="coerce").dropna()
    rating_counts = ratings_series.value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    rating_counts.plot.bar(color="#2B6CB0", edgecolor="#1A365D", width=0.6)
    plt.title("Distribution of User Ratings (1 to 5 Stars)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Rating (Stars)", fontsize=12)
    plt.ylabel("Rating Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_active_plot("rating_distribution.png", reports_folder)


def plot_user_activity_distribution(
    ratings_path: Path = USER_RATINGS_PATH,
    reports_folder: Path = REPORTS_DIRECTORY
) -> None:
    if not ratings_path.exists():
        return

    ratings_df = pd.read_csv(ratings_path)
    user_counts = ratings_df["user_id"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.hist(user_counts, bins=40, color="#4299E1", edgecolor="#2B6CB0", log=True)
    plt.title("User Activity Distribution (Ratings per User)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Number of Ratings Given", fontsize=12)
    plt.ylabel("Number of Users (Log Scale)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_active_plot("user_activity_distribution.png", reports_folder)


def plot_book_activity_distribution(
    ratings_path: Path = USER_RATINGS_PATH,
    reports_folder: Path = REPORTS_DIRECTORY
) -> None:
    if not ratings_path.exists():
        return

    ratings_df = pd.read_csv(ratings_path)
    book_counts = ratings_df["book_id"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.hist(book_counts, bins=40, color="#ED8936", edgecolor="#C05621", log=True)
    plt.title("Book Activity Distribution (Ratings per Book)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Number of Ratings Received", fontsize=12)
    plt.ylabel("Number of Books (Log Scale)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_active_plot("book_activity_distribution.png", reports_folder)


def plot_top_popular_books(
    popularity_model_path: Path = POPULARITY_MODEL_PATH,
    reports_folder: Path = REPORTS_DIRECTORY,
    top_n: int = 10
) -> None:
    if not popularity_model_path.exists():
        return

    pop_df = joblib.load(popularity_model_path)
    top_subset = pop_df.head(top_n).sort_values("popularity_score", ascending=True)

    truncated_titles = []
    for title in top_subset["title"]:
        title_text = str(title)
        if len(title_text) > 35:
            truncated_titles.append(title_text[:32] + "...")
        else:
            truncated_titles.append(title_text)

    plt.figure(figsize=(10, 6))
    plt.barh(truncated_titles, top_subset["popularity_score"], color="#DD6B20", edgecolor="#7B341E")
    plt.title(f"Top {top_n} Popular Books by Bayesian Score", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Bayesian Popularity Score", fontsize=12)
    plt.ylabel("Book Title", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    save_active_plot("top_popular_books.png", reports_folder)


def plot_hybrid_weights(reports_folder: Path = REPORTS_DIRECTORY) -> None:
    labels = [
        "Collaborative Filtering (35%)",
        "Content-Based (30%)",
        "Readability Match (15%)",
        "Popularity (20%)"
    ]
    weights = [0.35, 0.30, 0.15, 0.20]
    colors = ["#3182CE", "#38A169", "#DD6B20", "#805AD5"]

    plt.figure(figsize=(7, 7))
    plt.pie(
        weights,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.4}
    )
    plt.title("Hybrid Recommendation Model Component Weights", fontsize=14, fontweight="bold", pad=15)

    save_active_plot("hybrid_weights.png", reports_folder)


def plot_model_evaluation_metrics(
    evaluation_path: Path = EVALUATION_RESULTS_PATH,
    reports_folder: Path = REPORTS_DIRECTORY
) -> None:
    if not evaluation_path.exists():
        return

    results_df = pd.read_csv(evaluation_path)
    if "model" not in results_df.columns:
        return

    indexed_results = results_df.set_index("model")
    ranking_metric_cols = ["precision@10", "recall@10", "ndcg@10", "hit_rate@10"]
    metric_cols = ranking_metric_cols + ["coverage"]

    available_ranking_metrics = [
        col for col in ranking_metric_cols if col in indexed_results.columns
    ]
    available_metrics = [col for col in metric_cols if col in indexed_results.columns]

    if len(available_metrics) == 0:
        return

    if len(available_ranking_metrics) > 0:
        ranking_max = float(indexed_results[available_ranking_metrics].to_numpy().max())
        plt.figure(figsize=(10, 6))
        indexed_results[available_ranking_metrics].plot.bar(edgecolor="black", width=0.7)
        plt.title("Recommendation Models Performance Comparison", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Recommendation Model", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0.0, max(0.01, ranking_max * 1.15))
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend([metric.upper().replace("@", " @ ") for metric in available_ranking_metrics])
        save_active_plot("model_performance_comparison.png", reports_folder)

    metric_colors = {
        "precision@10": "#2C7A7B",
        "recall@10": "#805AD5",
        "ndcg@10": "#C53030",
        "hit_rate@10": "#2F855A",
        "coverage": "#B7791F"
    }

    for metric_name in available_metrics:
        plt.figure(figsize=(8, 5))
        plot_color = metric_colors.get(metric_name, "#4A5568")
        metric_max = float(indexed_results[metric_name].max())

        indexed_results[metric_name].plot.bar(color=plot_color, edgecolor="black", width=0.5)
        clean_title = metric_name.upper().replace("@", " @ ")
        plt.title(f"Model Comparison: {clean_title}", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Model Architecture", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0.0, max(0.01, metric_max * 1.15))
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        safe_filename = metric_name.replace("@", "_") + ".png"
        save_active_plot(safe_filename, reports_folder)


def generate_all_report_graphs(reports_folder: Path = REPORTS_DIRECTORY) -> None:
    reports_folder.mkdir(parents=True, exist_ok=True)
    print(f"Generating all 11 report visualization figures into: {reports_folder}")

    plot_rating_distribution(reports_folder=reports_folder)
    plot_user_activity_distribution(reports_folder=reports_folder)
    plot_book_activity_distribution(reports_folder=reports_folder)
    plot_top_popular_books(reports_folder=reports_folder)
    plot_hybrid_weights(reports_folder=reports_folder)
    plot_model_evaluation_metrics(reports_folder=reports_folder)

    print("All report figures successfully generated.")


if __name__ == "__main__":
    generate_all_report_graphs()
