# Evaluation Documentation

## Methodology

The project uses a simple offline evaluation setup on the processed user rating data. The recommendation models are assessed using a subset of user interactions and ranking metrics. The actual evaluation output stored in [../data/processed/evaluation_results.csv](../data/processed/evaluation_results.csv) is the source of truth.

## Metrics

The repository calculates and records:

- Precision@10
- Recall@10
- NDCG@10
- Hit Rate@10
- Coverage

## Actual results

The live evaluation file currently reports:

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity-Based | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |
| Content-Based | 0.0012 | 0.0120 | 0.0064 | 0.0120 | 0.1440 |
| Collaborative Filtering | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0673 |
| Hybrid | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0878 |

## Interpretation

These values are real, and they show that the current evaluation setup is conservative and not optimized for a high-score narrative. The content-based model is the strongest of the current baselines in this configuration, while the collaborative and hybrid models are limited by the evaluation design and user sparsity.

## Limitations

- The evaluation uses rankings on a sparse dataset and does not enforce a highly tuned recommender objective.
- Some models can return zero metrics under the selected evaluation settings.
- This is acceptable for an academic project, but it should not be described as a production-grade recommendation engine.

## Graphs

The evaluation charts are stored in [../reports](../reports) and can be regenerated with:

```bash
python src/generate_report_graphs.py
```
