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

## 3. Run the static demo

Open the demo page directly in a browser:

- [../demo/index.html](../demo/index.html)

If the API is running, the demo will call the live endpoints automatically.

## 4. Expected demonstration sequence

1. Select a model from the dropdown.
2. Enter a valid book or user ID.
3. Choose a recommendation count such as 5, 10, or 20.
4. Click Generate Recommendations.
5. Review the card list and model explanation.
6. Explain how the output is produced using the model documentation.

## 5. Viva-friendly explanation points

- Popularity is the baseline cold-start model.
- Content-Based uses metadata similarity around a chosen book.
- Collaborative uses interaction history and latent factors.
- Hybrid combines the three signals and is the strongest general recommender.

## 6. Local evaluation charts

After generating report graphs, open the images in [../reports](../reports) to show model metrics and chart outputs during the viva.
