# Architecture Overview

BookMind follows a straightforward recommendation pipeline from raw dataset to API and demonstration interface.

```text
Dataset
  ↓
Data Cleaning
  ↓
Feature Preparation
  ↓
Popularity ─┐
Content ───┼→ Hybrid → Evaluation
Collaborative ┘
  ↓
Saved Models
  ↓
FastAPI Service
  ↓
Demo Interface
```

## Components

### Data layer
- [../data/raw](../data/raw): raw source dataset files
- [../data/processed](../data/processed): cleaned and prepared datasets

### Model layer
- [../src/popularity_model.py](../src/popularity_model.py)
- [../src/content_model.py](../src/content_model.py)
- [../src/collaborative_model.py](../src/collaborative_model.py)
- [../src/hybrid_model.py](../src/hybrid_model.py)

### Evaluation and reports
- [../src/evaluation.py](../src/evaluation.py)
- [../src/generate_report_graphs.py](../src/generate_report_graphs.py)
- [../reports](../reports)

### Runtime API
- [../api/main.py](../api/main.py)

### Demo UI
- [../demo/index.html](../demo/index.html)
- [../demo/script.js](../demo/script.js)

## Pipeline details

1. Load the raw Goodreads-style data.
2. Clean and validate metadata and ratings.
3. Build a processed catalog and filtered ratings table.
4. Train popularity, content, and collaborative models.
5. Combine them into the hybrid model.
6. Save the results and metadata.
7. Expose recommendations via FastAPI.
8. Showcase results through the lightweight HTML/JS demo.

## Deployment note

This project is designed as a local academic ML application, not as a cloud-hosted product. It is meant to run on a local machine for viva, demonstration, and GitHub showcase purposes.
