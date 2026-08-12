# Phase 11 — ML Benchmarking

Phase 11 benchmarks reproducible baseline classifiers on the Phase 10 feature
vectors and an existing chronological train/validation/test split.

## Models

- DummyClassifier (prior)
- Logistic Regression
- Random Forest
- Gradient Boosting

## Evaluation

For every model, validation and test metrics are recorded:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1
- ROC-AUC when binary probabilities are available
- PR-AUC when binary probabilities are available
- Confusion matrix

Model selection uses **validation F1 by default**. The test set is never used
for selecting the winning model.

## Reproducibility

- deterministic model seeds
- deterministic model ordering
- serialized model parameters
- serialized feature names
- explicit split sample sizes and class distributions

## Scope boundary

Phase 11 does not:
- refit Phase 10 features on validation/test
- create a new random split
- inspect future return/impact fields as features
- tune hyperparameters against the test set
- perform backtesting or paper trading

## Run

```bash
python -m unittest tests/test_ml_benchmark.py -v
python -m unittest discover -s tests -v
git diff --check
```

Do not commit/tag/push until review and full regression pass.
