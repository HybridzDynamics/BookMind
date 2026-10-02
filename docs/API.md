# API Reference

BookMind exposes a FastAPI service for model access and demonstration. The recommended startup command is:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

After startup, open the Swagger UI at:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/redoc

## Endpoints

### GET /
Returns service metadata and documentation links.

### GET /health
Returns service health status.

### GET /recommendations/popular
Returns the top popular books.

Example:

```bash
curl "http://127.0.0.1:8000/recommendations/popular?limit=5"
```

### GET /recommendations/content/{book_id}
Returns content-based recommendations for a selected book.

Example:

```bash
curl "http://127.0.0.1:8000/recommendations/content/1?limit=5"
```

### GET /recommendations/collaborative/{user_id}
Returns collaborative filtering recommendations for a user.

### GET /recommendations/hybrid/{user_id}
Returns hybrid recommendations for a user and combines the component scores.

## Validation and error handling

The API validates the `limit` parameter to keep values in the range 1–100. Unknown books return a 404 status, and model file errors are converted into clear API-friendly messages rather than raw tracebacks.

## Local demo

The static demo in [../demo/index.html](../demo/index.html) calls these endpoints directly.
