# Phase 9 — Dataset Validation & ML Readiness

Phase 9 validates the Phase 8 canonical research dataset before downstream
feature engineering or model training.

## Scope

- leakage detection
- duplicate event×ticker detection
- required feature/time validation
- missing-data analysis
- target availability
- class-balance warnings
- conservative numeric outlier review
- chronological train/validation/test split

## Explicit non-goals

Phase 9 does not train models, engineer technical indicators, fetch market
data, recalculate returns/statistics, or mutate the Phase 8 dataset.

## API

```python
from app.dataset_validator import DatasetValidator

report = DatasetValidator().validate(dataset, target="impact_label")
print(report.to_json())
```

The split is chronological and never shuffled.

## Run

```bash
python -m unittest tests/test_dataset_validator.py -v
python -m unittest discover -s tests -v
git diff --check
```

Do not commit/tag/push until Phase 9 review and full regression pass.
