import unittest
from datetime import date
from app.final_opportunity.backtest.pit_v2 import (
    FinancialHistory,
    build_pit_state,
    audit_pit_states,
)


class TestPITReconstructionV2(unittest.TestCase):
    def evidenced(self, period, pub, score):
        return FinancialHistory(
            "BBRI",
            date.fromisoformat(period),
            date.fromisoformat(pub),
            score,
            "stockanalysis_web",
            "listing_timestamp",
            pub + "T10:00:00",
            "https://idx.sahamidx.com/lk/?y=2026",
        )

    def test_historical_requires_evidence(self):
        h = [
            FinancialHistory(
                "BBRI",
                date(2025, 12, 31),
                date(2026, 3, 1),
                .60,
                "stockanalysis_web",
            )
        ]
        r = build_pit_state(
            "BBRI",
            date(2026, 8, 14),
            financial_history=h,
            require_financial_evidence=True,
        )
        self.assertIsNone(r.fundamental_score)
        self.assertIsNone(r.financial_period_end)

    def test_evidence_backed_fact_is_selected(self):
        h = [
            self.evidenced("2024-12-31", "2025-03-01", .55),
            self.evidenced("2025-12-31", "2026-03-01", .60),
        ]
        r = build_pit_state(
            "BBRI",
            date(2026, 8, 14),
            financial_history=h,
            require_financial_evidence=True,
        )
        self.assertEqual(r.financial_period_end, date(2025, 12, 31))
        self.assertEqual(r.fundamental_score, .60)
        self.assertEqual(r.financial_evidence_level, "listing_timestamp")

    def test_future_evidence_is_not_used(self):
        h = [
            FinancialHistory(
                "BBRI",
                date(2025, 12, 31),
                date(2026, 9, 1),
                .90,
                "stockanalysis_web",
                "listing_timestamp",
            )
        ]
        r = build_pit_state(
            "BBRI",
            date(2026, 8, 14),
            financial_history=h,
            require_financial_evidence=True,
        )
        self.assertIsNone(r.fundamental_score)

    def test_duplicate_snapshot_is_rejected(self):
        a = build_pit_state("BBRI", date(2026, 8, 14)).to_dict()
        b = build_pit_state("BBRI", date(2026, 8, 14)).to_dict()
        audit = audit_pit_states([a, b])
        self.assertEqual(audit["status"], "FAIL")
        self.assertEqual(audit["duplicate_observations"], 1)

    def test_fundamental_without_evidence_fails_audit(self):
        row = build_pit_state(
            "BBRI",
            date(2026, 8, 14),
            financial_history=[
                FinancialHistory(
                    "BBRI", date(2025, 12, 31), date(2026, 3, 1), .60, "sa"
                )
            ],
        ).to_dict()
        audit = audit_pit_states([row])
        self.assertEqual(audit["status"], "FAIL")
        self.assertEqual(audit["unevidenced_financial_observations"], 1)


if __name__ == "__main__":
    unittest.main()
