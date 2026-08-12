# Phase 7 — Event Impact Signal & Labeling (Correction)

This correction hardens the Phase 7 contract without modifying Phase 1-6.

Changes:
- explicit INVALID_INPUT path;
- strict boolean handling for Phase 6 `significant`;
- deterministic p-value fallback when significance is absent;
- explicit Phase-6 CAR window selection;
- expanded boundary/data-quality tests;
- no market-data, event-time, return, or statistical recalculation.

Apply from repository root after removing the current untracked Phase 7 files:

```bash
rm -f app/event_impact.py tests/test_event_impact.py README_PHASE7.md
git apply --check --recount /tmp/phase7-fix/phase7-fix.patch
git apply --recount /tmp/phase7-fix/phase7-fix.patch
```

Then:

```bash
python -m unittest tests/test_event_impact.py -v
python -m unittest discover -s tests -v
git diff --check
git status --short
```

Do not commit/tag/push until final review.
