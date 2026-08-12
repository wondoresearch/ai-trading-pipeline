# Phase 10 — Leakage-Safe Feature Engineering

Phase 10 converts Phase 8 event-level features into deterministic numeric
vectors suitable for downstream ML.

## Scope

- numeric sentiment features with training-only scaling
- deterministic one-hot encoding for categorical event/sentiment fields
- cyclical encoding of effective hour and weekday
- explicit rejection of target/outcome fields in the feature layer
- deterministic feature ordering and JSON serialization
- unseen categories map to all-zero category indicators

## Leakage contract

`FeatureEngineer.fit()` MUST receive the training partition only.

It learns:
- numeric scaling statistics
- categorical vocabularies

Validation/test records are passed only to `transform()`.

Targets and post-event outcomes are never read and are rejected if they appear
inside `features`.

Missing numeric values are represented as `0.0` after transformation. This is
an explicit model-input encoding, not a financial return substitution.

## Current feature families

Numeric:
- `sentiment_score`
- `sentiment_magnitude` when supplied

Categorical:
- `event_type`
- `sentiment_label`

Time:
- effective hour sine/cosine
- effective weekday sine/cosine

## Run

```bash
python -m unittest tests/test_feature_engineering.py -v
python -m unittest discover -s tests -v
git diff --check
```

Do not commit/tag/push until Phase 10 review and full regression pass.
