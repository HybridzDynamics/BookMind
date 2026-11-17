<div align="center">

# 📚 BookMind

**An intelligent book recommendation system built with Python, FastAPI, and four ML models.**

<img src="https://img.shields.io/badge/Python-3.10--3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10-3.13" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
<img src="https://img.shields.io/badge/License-MIT-8b5cf6?style=for-the-badge" alt="MIT License" />

</div>

BookMind demonstrates how a complete recommender pipeline works, from raw data cleaning and feature extraction through model training, evaluation, a REST API, and a live demo interface.

**Jump to:** [What it does](#what-it-does) · [Quick start](#quick-start) · [Models](#models) · [Evaluation](#evaluation) · [Testing](#testing) · [Limitations](#limitations)

---

## What it does

| Model | Input | Signal | Use case |
|---|---|---|---|
| Popularity | — | Rating volume + average | Cold-start / trending books |
| Content-Based | Book ID | TF-IDF metadata similarity | "More like this" |
| Collaborative | User ID | SVD latent factors | Personalized recommendations |
| Hybrid | User ID | 40% Collab + 35% Content + 25% Popularity | Best overall accuracy |

## Dataset

- **10,000 books** · **53,424 users** · **5,976,479 ratings**
- Source: Goodreads-style public dataset (see [docs/DATASET.md](docs/DATASET.md) for attribution and license)

## Key features

- Popularity baseline with Bayesian weighted scoring
- Content-based recommendations using TF-IDF + cosine NearestNeighbors
- Collaborative filtering with TruncatedSVD latent factors
- Hybrid ensemble with configurable per-component weights
- FastAPI backend with full Swagger/OpenAPI UI at `/docs`
- Interactive browser demo (no Node.js required)
- Streamlit training dashboard with live pipeline logs
- Evaluation metrics: Precision@10, Recall@10, NDCG@10, Hit Rate@10
- Author diversity enforcement in all recommendation outputs
- Graceful cold-start fallback to popularity for unknown users

## Documentation

| File | Contents |
|---|---|
| [docs/DEMO.md](docs/DEMO.md) | Step-by-step demo guide |
| [docs/API.md](docs/API.md) | API endpoint reference |
| [docs/MODELS.md](docs/MODELS.md) | Model explanations |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Metrics and results |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture |
| [docs/DATASET.md](docs/DATASET.md) | Dataset details and attribution |

## Project structure

```text
bookmind/
├── api/          FastAPI backend
├── data/         Processed datasets
├── demo/         Static HTML/CSS/JS demo interface
├── docs/         All project documentation
├── models/       Trained model pickle files
├── reports/      Evaluation charts and graphs
├── src/          ML model source code
├── tests/        Pytest test suite
├── app.py        Streamlit dashboard
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Quick start

> **Requires Python 3.10–3.13.**

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models (optional)

Trained model files are already included in `models/`. To retrain from scratch:

```bash
python src/train_all.py
```

This runs all 8 pipeline stages: data cleaning → ratings prep → popularity → content → collaborative → hybrid → evaluation → report graphs.

### 4. Start the API server

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Swagger UI is then available at **http://127.0.0.1:8000/docs**.

### 5. Open the demo

**Browser demo:** with the API running, open `demo/index.html` in a browser.

**Streamlit dashboard:**

```bash
streamlit run app.py
```

## Models

See [docs/MODELS.md](docs/MODELS.md) for full per-model documentation.

| Model | Algorithm | Key parameters |
|---|---|---|
| Popularity | Bayesian weighted score | Minimum 8 ratings |
| Content-Based | TF-IDF + NearestNeighbors (cosine) | 50,000 vocabulary, `n_neighbors=50` |
| Collaborative | TruncatedSVD | 50 latent components |
| Hybrid | Weighted combination | 40% + 35% + 25% |

## Evaluation

Metrics are reported in [docs/EVALUATION.md](docs/EVALUATION.md) and stored in [data/processed/evaluation_results.csv](data/processed/evaluation_results.csv).

Evaluation uses a per-user holdout (each user's last rated item) to avoid data leakage. Absolute scores are modest, which is typical for sparse ratings data with a single held-out item per user, so compare models against each other rather than against an ideal.

## Architecture

![BookMind architecture diagram](docs/images/architecture.png)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for pipeline details.

## Testing

```bash
pytest
```

The suite has 15 tests covering model behavior, API endpoints, data validation, cold-start handling, and recommendation output.

## Limitations

- Collaborative recommendations depend on the density of user interaction data.
- Cold-start users (not in training) receive popularity-based fallback.
- Evaluation scores are low in absolute terms because the dataset is sparse.
- Recommendation quality is limited by metadata completeness in the source dataset.

## License

MIT License. See [LICENSE](LICENSE).

> **Note**: The source dataset has its own attribution files in the `dataset/` directory. See [docs/DATASET.md](docs/DATASET.md).
