
from datetime import datetime
from pathlib import Path
import random
from typing import Any, Dict, List, Optional
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"
USER_RATINGS_PATH = PROJECT_ROOT / "data" / "processed" / "user_ratings.csv"
POPULARITY_MODEL_PATH = PROJECT_ROOT / "models" / "popularity_model.pkl"
CONTENT_MODEL_PATH = PROJECT_ROOT / "models" / "content_model.pkl"
COLLABORATIVE_MODEL_PATH = PROJECT_ROOT / "models" / "collaborative_model.pkl"
HYBRID_MODEL_PATH = PROJECT_ROOT / "models" / "hybrid_model.pkl"
EVALUATION_PATH = PROJECT_ROOT / "data" / "processed" / "evaluation_results.csv"
SIMULATION_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "virtual_user_activity.csv"
TRAINING_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "training.log"
REPORTS_DIR = PROJECT_ROOT / "reports"


st.set_page_config(
    page_title="BookMind — Recommendation Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_csv_data(file_path: Path) -> pd.DataFrame:
    if file_path.exists():
        loaded_df = pd.read_csv(file_path)
        return loaded_df
    else:
        return pd.DataFrame()


def count_trained_models() -> int:
    model_paths = [
        POPULARITY_MODEL_PATH,
        CONTENT_MODEL_PATH,
        COLLABORATIVE_MODEL_PATH,
        HYBRID_MODEL_PATH
    ]
    trained_count = 0
    for path in model_paths:
        if path.exists():
            trained_count += 1
    return trained_count


def render_dashboard_panel(books_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
    st.header("📊 System Overview & Metrics")

    total_books = len(books_df)
    total_ratings = len(ratings_df)
    total_models = count_trained_models()

    if ratings_df.empty:
        total_users = 0
    else:
        total_users = ratings_df["user_id"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Books in Catalog", f"{total_books:,}")
    col2.metric("Unique Users", f"{total_users:,}")
    col3.metric("Ratings Collected", f"{total_ratings:,}")
    col4.metric("Trained Models", f"{total_models} / 4")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Top Trending Books (Bayesian Popularity)")
        if POPULARITY_MODEL_PATH.exists():
            from src.popularity_model import get_popular_books
            popular_df = get_popular_books(number_of_books=10)

            display_cols = ["bookId", "title", "author", "rating", "numRatings", "popularity_score"]
            valid_cols = []
            for col in display_cols:
                if col in popular_df.columns:
                    valid_cols.append(col)

            st.dataframe(popular_df[valid_cols], use_container_width=True, hide_index=True)
        else:
            st.info("Popularity model not yet trained. Run `python src/popularity_model.py` or use the Training Panel.")

    with col_right:
        st.subheader("Dataset Summary")
        if not books_df.empty:
            summary_info = {
                "Metric": [
                    "Average Catalog Rating",
                    "Total Ratings Available",
                    "Unique Authors",
                    "SVD Factors Target",
                    "Recommendation Ensembles"
                ],
                "Value": [
                    f"{books_df['rating'].mean():.2f} / 5.0",
                    f"{total_ratings:,}",
                    f"{books_df['author'].nunique():,}",
                    "50 latent components",
                    "Collaborative (35%) + Content (30%) + Readability (15%) + Popularity (20%)"
                ]
            }
            st.table(pd.DataFrame(summary_info))
        else:
            st.warning("Books dataset not processed yet.")


def render_book_search_panel(books_df: pd.DataFrame) -> None:
    st.header("🔍 Book Search & Content Discovery")

    if books_df.empty:
        st.warning("Cleaned books dataset missing. Run `python src/data_cleaning.py` first.")
        return

    search_query = st.text_input(
        "Search catalog by Title, Author, Genre, or Keywords:",
        placeholder="e.g. Harry Potter, Stephen King, sci-fi, thriller..."
    )

    if search_query:
        clean_query = search_query.strip().lower()
        title_mask = books_df["title"].str.lower().str.contains(clean_query, regex=False)
        author_mask = books_df["author"].str.lower().str.contains(clean_query, regex=False)
        content_mask = books_df["content"].str.lower().str.contains(clean_query, regex=False)

        matches_df = books_df[title_mask | author_mask | content_mask].head(25)
        st.write(f"Displaying **{len(matches_df)}** matching titles:")

        if not matches_df.empty:
            show_columns = ["bookId", "title", "author", "rating", "numRatings", "genres"]
            valid_show = []
            for col in show_columns:
                if col in matches_df.columns:
                    valid_show.append(col)

            st.dataframe(matches_df[valid_show], use_container_width=True, hide_index=True)

            st.markdown("### Discover Content-Similar Books")
            book_choices = {}
            for _, row in matches_df.iterrows():
                book_label = f"{row['title']} by {row['author']} (ID: {row['bookId']})"
                book_choices[book_label] = str(row["bookId"])

            selected_book_label = st.selectbox("Select a book from search results:", list(book_choices.keys()))
            selected_book_id = book_choices[selected_book_label]
            recommendation_limit = st.slider("Number of similar books to discover:", 3, 20, 5)

            if st.button("Find Similar Books", type="primary"):
                if not CONTENT_MODEL_PATH.exists():
                    st.error("Content model is not yet trained. Run `python src/content_model.py` first.")
                else:
                    from src.content_model import recommend_similar_books
                    try:
                        similar_results = recommend_similar_books(selected_book_id, number_of_books=recommendation_limit)
                        results_df = pd.DataFrame(similar_results)
                        st.success(f"Discovered {len(similar_results)} similar books!")

                        display_cols = ["bookId", "title", "author", "similarity_score", "reason"]
                        valid_display = []
                        for col in display_cols:
                            if col in results_df.columns:
                                valid_display.append(col)

                        st.dataframe(results_df[valid_display], use_container_width=True, hide_index=True)
                    except Exception as error_exception:
                        st.error(f"Failed to retrieve similar books: {error_exception}")


def render_user_recommendations_panel(ratings_df: pd.DataFrame) -> None:
    st.header("👤 Personalized User Recommendations")

    if ratings_df.empty:
        st.warning("User ratings missing. Run `python src/prepare_ratings.py` first.")
        return

    unique_users = sorted(ratings_df["user_id"].astype(str).unique().tolist())
    sample_users = unique_users[:100]

    col_u, col_m, col_n = st.columns([2, 2, 1])

    with col_u:
        target_user = st.selectbox("Choose Target User ID:", sample_users)

    with col_m:
        model_choice = st.selectbox(
            "Recommendation Engine:",
            ["Hybrid Ensemble (35% Collab + 30% Content + 15% Readability + 20% Popularity)", "Collaborative Filtering (SVD)"]
        )

    with col_n:
        rec_count = st.slider("Recommendations Count:", 5, 25, 10)

    with st.expander("User's Past Ratings History"):
        user_history = ratings_df[ratings_df["user_id"].astype(str) == str(target_user)]
        st.dataframe(user_history.sort_values("rating", ascending=False), use_container_width=True, hide_index=True)

    if st.button("Generate Recommendations", type="primary"):
        if model_choice.startswith("Hybrid"):
            if not HYBRID_MODEL_PATH.exists():
                st.error("Hybrid model artifact missing. Run `python src/hybrid_model.py` first.")
            else:
                from src.hybrid_model import recommend_for_user as get_hybrid_rec
                try:
                    recs = get_hybrid_rec(target_user, number_of_books=rec_count)
                    recs_df = pd.DataFrame(recs)
                    st.success(f"Generated {len(recs)} Hybrid Recommendations!")

                    preferred_cols = [
                        "bookId", "title", "author", "hybrid_score",
                        "collaborative_score", "content_score", "readability_score", "popularity_score", "reason"
                    ]
                    available_preferred = []
                    for col in preferred_cols:
                        if col in recs_df.columns:
                            available_preferred.append(col)

                    st.dataframe(recs_df[available_preferred], use_container_width=True, hide_index=True)
                except Exception as error_exception:
                    st.error(f"Error generating hybrid recommendations: {error_exception}")

        else:
            if not COLLABORATIVE_MODEL_PATH.exists():
                st.error("Collaborative model artifact missing. Run `python src/collaborative_model.py` first.")
            else:
                from src.collaborative_model import recommend_for_user as get_collab_rec
                try:
                    recs = get_collab_rec(target_user, number_of_books=rec_count)
                    recs_df = pd.DataFrame(recs)
                    st.success(f"Generated {len(recs)} Collaborative Filtering Recommendations!")
                    st.dataframe(recs_df, use_container_width=True, hide_index=True)
                except Exception as error_exception:
                    st.error(f"Error generating collaborative recommendations: {error_exception}")


def render_virtual_simulation_panel(books_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
    st.header("🤖 Virtual User Activity Simulation")
    st.caption("Simulates real-world browsing sessions without modifying ground-truth ratings.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sim_users_count = st.number_input("Number of Virtual Users:", min_value=1, max_value=25, value=5)
    with col2:
        sim_searches_per_user = st.number_input("Searches per User:", min_value=1, max_value=10, value=3)
    with col3:
        sim_model_choice = st.selectbox("Model to Simulate:", ["hybrid", "collaborative", "content"])
    with col4:
        sim_seed = st.number_input("Random Seed:", min_value=1, max_value=9999, value=42)

    if st.button("Start Live Simulation", type="primary"):
        if books_df.empty or ratings_df.empty:
            st.error("Data missing. Ensure data_cleaning.py and prepare_ratings.py have run.")
            return

        random.seed(sim_seed)
        unique_users = ratings_df["user_id"].astype(str).unique().tolist()
        sample_count = min(sim_users_count, len(unique_users))
        selected_users = random.sample(unique_users, sample_count)

        from src.content_model import recommend_similar_books
        from src.collaborative_model import recommend_for_user as get_collab_rec
        from src.hybrid_model import recommend_for_user as get_hybrid_rec

        total_actions = sample_count * sim_searches_per_user
        progress_bar = st.progress(0)
        status_text = st.empty()

        sim_records = []
        action_counter = 0
        success_counter = 0
        books_searched_list = []

        for u_idx, real_user_id in enumerate(selected_users, start=1):
            virtual_user_id = f"virtual_user_{u_idx}"

            for s_num in range(1, sim_searches_per_user + 1):
                action_counter += 1
                sample_book = books_df.sample(1).iloc[0]
                target_book_id = str(sample_book["bookId"])
                target_book_title = str(sample_book["title"])
                books_searched_list.append(target_book_title)

                title_words = target_book_title.split()
                if len(title_words) > 0:
                    query_word = title_words[0]
                else:
                    query_word = "book"

                status_msg = "success"
                recs_found = 0

                try:
                    if sim_model_choice == "content":
                        recs = recommend_similar_books(target_book_id, number_of_books=5)
                    elif sim_model_choice == "collaborative":
                        recs = get_collab_rec(real_user_id, number_of_books=5)
                    else:
                        recs = get_hybrid_rec(real_user_id, number_of_books=5)

                    recs_found = len(recs)
                    success_counter += 1
                except Exception as ex:
                    status_msg = f"error: {str(ex)}"
                    recs_found = 0

                sim_records.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "virtual_user": virtual_user_id,
                    "real_user_id": real_user_id,
                    "search_number": s_num,
                    "query": query_word,
                    "selected_book_id": target_book_id,
                    "selected_book_title": target_book_title,
                    "model": sim_model_choice,
                    "recommendation_count": recs_found,
                    "status": status_msg
                })

                progress_val = action_counter / total_actions
                progress_bar.progress(progress_val)
                status_text.text(f"Simulating action {action_counter}/{total_actions}: {virtual_user_id} querying '{query_word}'...")

        progress_bar.progress(1.0)
        status_text.success("Simulation finished successfully!")

        sim_df = pd.DataFrame(sim_records)
        SIMULATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        sim_df.to_csv(SIMULATION_LOG_PATH, index=False)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Active Virtual Users", f"{sample_count}")
        m2.metric("Total Searches", f"{action_counter}")
        m3.metric("Successful Queries", f"{success_counter}")
        m4.metric("Books Searched", f"{len(set(books_searched_list))}")
        m5.metric("Total Recs Delivered", f"{sim_df['recommendation_count'].sum()}")

        st.subheader("Live Activity Log")
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

        csv_bytes = sim_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Simulation Activity CSV",
            data=csv_bytes,
            file_name="virtual_user_activity.csv",
            mime="text/csv"
        )


def render_report_graphs_panel() -> None:
    st.header("📈 Report Visualizations & Model Evaluation")

    if EVALUATION_PATH.exists():
        st.subheader("Information Retrieval Model Performance")
        eval_df = pd.read_csv(EVALUATION_PATH)
        st.dataframe(eval_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Visual Analytics Gallery (reports/)")

    report_figures = [
        ("rating_distribution.png", "Distribution of User Ratings (1-5 Stars)"),
        ("user_activity_distribution.png", "Ratings Distribution per User"),
        ("book_activity_distribution.png", "Ratings Distribution per Book"),
        ("top_popular_books.png", "Top 10 Popular Books by Bayesian Score"),
        ("hybrid_weights.png", "Hybrid Recommendation Component Weights"),
        ("model_performance_comparison.png", "Overall Model Performance Comparison"),
        ("precision_10.png", "Precision@10 Metric Comparison"),
        ("recall_10.png", "Recall@10 Metric Comparison"),
        ("ndcg_10.png", "NDCG@10 Metric Comparison"),
        ("hit_rate_10.png", "Hit Rate@10 Metric Comparison"),
        ("coverage.png", "Catalog Coverage Comparison")
    ]

    for index in range(0, len(report_figures), 2):
        col_a, col_b = st.columns(2)
        fig_a_name, fig_a_title = report_figures[index]
        path_a = REPORTS_DIR / fig_a_name

        with col_a:
            if path_a.exists():
                st.image(str(path_a), caption=fig_a_title, use_container_width=True)
            else:
                st.info(f"Report image '{fig_a_name}' not generated yet. Run `python src/generate_report_graphs.py`.")

        if index + 1 < len(report_figures):
            fig_b_name, fig_b_title = report_figures[index + 1]
            path_b = REPORTS_DIR / fig_b_name
            with col_b:
                if path_b.exists():
                    st.image(str(path_b), caption=fig_b_title, use_container_width=True)
                else:
                    st.info(f"Report image '{fig_b_name}' not generated yet. Run `python src/generate_report_graphs.py`.")


def render_training_logs_panel() -> None:
    st.header("⚙️ Model Training & Live Execution Logs")
    st.caption("Trigger data processing, train recommendation models, and monitor real-time execution logs.")

    from src.train_all import execute_single_stage, run_full_training_pipeline

    col_ctrl1, col_ctrl2 = st.columns([3, 1])

    with col_ctrl1:
        training_target = st.selectbox(
            "Select Training Scope:",
            [
                "All Stages (Full End-to-End Pipeline)",
                "Stage 1: Data Cleaning (books_clean.csv)",
                "Stage 2: Ratings Preparation (user_ratings.csv)",
                "Stage 3: Popularity Model (popularity_model.pkl)",
                "Stage 4: Content-Based Model (content_model.pkl)",
                "Stage 5: Collaborative Filtering (collaborative_model.pkl)",
                "Stage 6: Hybrid Ensemble (hybrid_model.pkl)",
                "Stage 7: Offline Evaluation (evaluation_results.csv)",
                "Stage 8: Report Graphs Generation (reports/*.png)"
            ]
        )

    with col_ctrl2:
        st.write("")
        st.write("")
        start_training_clicked = st.button("▶ Start Training", type="primary")

    if start_training_clicked:
        st.markdown("### 🖥️ Live Training Console")
        progress_bar = st.progress(0.0)
        status_banner = st.empty()
        log_box = st.empty()

        captured_logs = []

        def handle_live_log(log_line: str) -> None:
            captured_logs.append(log_line)
            combined_text = "\n".join(captured_logs)
            log_box.code(combined_text, language="text")

        status_banner.info("Initializing selected training task...")

        if training_target.startswith("All"):
            stages_list = [
                ("1", "Data Cleaning", 0.12),
                ("2", "Ratings Preparation", 0.25),
                ("3", "Popularity Model", 0.38),
                ("4", "Content-Based Model", 0.50),
                ("5", "Collaborative Filtering", 0.65),
                ("6", "Hybrid Ensemble", 0.78),
                ("7", "Offline Evaluation", 0.90),
                ("8", "Report Graphs Generation", 1.0)
            ]

            overall_success = True
            for stage_code, stage_title, target_progress in stages_list:
                status_banner.info(f"Executing {stage_title}...")
                success = execute_single_stage(stage_code, log_callback=handle_live_log)
                if not success:
                    overall_success = False
                    status_banner.error(f"Stage '{stage_title}' failed! Check console output below.")
                    break
                progress_bar.progress(target_progress)

            if overall_success:
                from src.train_all import save_model_metadata
                save_model_metadata(log_callback=handle_live_log)
                progress_bar.progress(1.0)
                status_banner.success("✅ Full pipeline completed successfully! All models and reports are ready.")
        else:
            stage_number = training_target.split(":")[0].replace("Stage", "").strip()
            status_banner.info(f"Executing {training_target}...")
            single_success = execute_single_stage(stage_number, log_callback=handle_live_log)
            progress_bar.progress(1.0)
            if single_success:
                status_banner.success(f"✅ {training_target} completed successfully!")
            else:
                status_banner.error(f"❌ {training_target} encountered an error.")

    st.markdown("---")
    st.subheader("📜 Historical Training Log (`data/processed/training.log`)")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        reload_logs = st.button("🔄 Refresh Log File")
    with col_btn2:
        clear_logs = st.button("🗑️ Clear Log History")

    if clear_logs:
        if TRAINING_LOG_PATH.exists():
            TRAINING_LOG_PATH.unlink()
            st.success("Log file cleared.")

    if TRAINING_LOG_PATH.exists():
        with open(TRAINING_LOG_PATH, "r", encoding="utf-8") as fh:
            file_log_content = fh.read()

        line_count = len(file_log_content.splitlines())
        st.caption(f"Displaying {line_count:,} recorded log lines from `{TRAINING_LOG_PATH}`:")
        st.code(file_log_content, language="text")
    else:
        st.info("No training log file found yet. Click 'Start Training' above to generate execution logs.")


def main() -> None:
    books_df = load_csv_data(CLEAN_BOOKS_PATH)
    ratings_df = load_csv_data(USER_RATINGS_PATH)

    st.sidebar.title("Navigation")
    active_panel = st.sidebar.radio(
        "Choose Panel:",
        [
            "Dashboard",
            "Book Search",
            "User Recommendations",
            "Virtual User Simulation",
            "Report Graphs",
            "Model Training & Logs"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Artifact Status")
    status_map = {
        "Popularity": POPULARITY_MODEL_PATH.exists(),
        "Content-Based": CONTENT_MODEL_PATH.exists(),
        "Collaborative": COLLABORATIVE_MODEL_PATH.exists(),
        "Hybrid": HYBRID_MODEL_PATH.exists()
    }
    for model_name, is_ready in status_map.items():
        if is_ready:
            st.sidebar.write(f"✅ **{model_name}**: Ready")
        else:
            st.sidebar.write(f"⚠️ **{model_name}**: Not Trained")

    if active_panel == "Dashboard":
        render_dashboard_panel(books_df, ratings_df)
    elif active_panel == "Book Search":
        render_book_search_panel(books_df)
    elif active_panel == "User Recommendations":
        render_user_recommendations_panel(ratings_df)
    elif active_panel == "Virtual User Simulation":
        render_virtual_simulation_panel(books_df, ratings_df)
    elif active_panel == "Report Graphs":
        render_report_graphs_panel()
    elif active_panel == "Model Training & Logs":
        render_training_logs_panel()


if __name__ == "__main__":
    main()
