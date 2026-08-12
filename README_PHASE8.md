# Phase 8 — Research Dataset & Event Feature Pipeline

Phase 8 creates the canonical event×ticker research dataset from Phase 1–7
outputs without recalculating upstream results.

## Contract

- Feature and target layers are physically separated.
- Future outcomes never enter `features()`.
- Phase 5 returns, Phase 6 CAR/statistics, and Phase 7 labels are consumed as-is.
- Missing values remain `None`; no zero substitution.
- `event_id + ticker` is the analytical primary key.
- Duplicate keys are explicit and never silently overwritten.
- Cross-phase identity mismatches are rejected.
- Output ordering and JSON serialization are deterministic.
- No market-data provider, NLP model, return calculation, or statistical
  recalculation is used.

## Files

- `app/research_dataset.py`
- `tests/test_research_dataset.py`

## Run

```bash
python -m unittest tests/test_research_dataset.py -v
python -m unittest discover -s tests -v
git diff --check
```

Do not commit/tag/push until Phase 8 design review and full regression pass.
