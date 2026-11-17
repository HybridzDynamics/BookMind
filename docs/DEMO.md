# Demo Guide

## 1. Start the API

From the project root, run:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

## 2. Open Swagger

Visit:

- http://127.0.0.1:8000/docs

Use the interface to test:

- /health
- /recommendations/popular
- /recommendations/content/{book_id}
- /recommendations/collaborative/{user_id}
- /recommendations/hybrid/{user_id}

## 3. Open the live demo

With the API running, open:

- http://127.0.0.1:8000/demo/

On Windows you can start the API and open that page with `start_demo.bat`.

The page calls the live recommendation endpoints. Opening `demo/index.html` as a file also works, because the API allows browser requests from another origin.

## 4. Expected demonstration sequence

1. Select a model from the dropdown.
2. Enter a valid book or user ID.
3. Choose a recommendation count such as 5, 10, or 20.
4. Click Generate Recommendations.
5. Review the card list and model explanation.
6. Explain how the output is produced using the model documentation.

## 5. Self Account demonstration (new)

1. Click the **Self Account** tab.
2. Click **+ New Account**, enter ID `demo` and name `Demo User`.
3. Search for "Harry Potter" and rate the first result 5 stars.
4. Search for "Hunger Games" and rate the first result 4 stars.
5. Observe the personalized recommendations appear automatically.
6. Explain SVD fold-in: your ratings are folded into the pre-trained SVD latent space without retraining.

## 6. Viva-friendly explanation points

- Popularity is the baseline cold-start model.
- Content-Based uses metadata similarity around a chosen book.
- Collaborative uses interaction history and latent factors.
- Hybrid combines the three signals and is the strongest general recommender.
- Self Account uses SVD fold-in: personal recommendations from your ratings without full model retrain.

## 7. Local evaluation charts

After generating report graphs, open the images in [../reports](../reports) to show model metrics and chart outputs during the viva.

### Available Visualizations (13 total)

| Graph | Description |
|---|---|
| `rating_distribution.png` | Distribution of 1-5 star ratings |
| `user_activity_distribution.png` | Ratings count per user (log scale) |
| `book_activity_distribution.png` | Ratings count per book (log scale) |
| `top_popular_books.png` | Top 10 books by Bayesian popularity score |
| `hybrid_weights.png` | Hybrid model component weights pie chart |
| `rating_vs_popularity_scatter.png` | Scatter: rating vs rating count, colored by popularity |
| `user_book_interaction_matrix.png` | Heatmap: top 100 users × top 100 books interaction matrix |
| `model_performance_comparison.png` | All model metrics side-by-side |
| `precision_10.png` | Precision@10 per model |
| `recall_10.png` | Recall@10 per model |
| `ndcg_10.png` | NDCG@10 per model |
| `hit_rate_10.png` | Hit Rate@10 per model |
| `coverage.png` | Catalog coverage per model |
