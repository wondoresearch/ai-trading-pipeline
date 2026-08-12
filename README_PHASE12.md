# Phase 12 — Model Training & Selection

Phase 12 performs controlled, deterministic tuning of the model family chosen
from Phase 11. Candidate selection is based only on the validation partition.

## Scope

- small deterministic hyperparameter grids
- validation-only candidate selection
- optional validation-only classification-threshold selection
- one final evaluation on the untouched test partition
- reproducible model parameters and artifact metadata

## Model families

- Logistic Regression
- Random Forest
- Gradient Boosting
- Dummy classifier

## Data boundary

Phase 10 feature vectors are supplied directly. Phase 12 never refits or
changes feature engineering and never creates a new split.

The test partition is not used for:
- hyperparameter selection
- model selection
- threshold selection

## Run

```bash
python -m unittest tests/test_model_training.py -v
python -m unittest discover -s tests -v
git diff --check
```

Do not commit/tag/push until Phase 12 review and full regression pass.
