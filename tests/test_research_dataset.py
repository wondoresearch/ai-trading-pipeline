import json
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from app.research_dataset import (
    ResearchDatasetBuilder,
    ResearchDatasetStatus,
    ResearchRecordInput,
)


class TestResearchDataset(unittest.TestCase):
    tz = ZoneInfo("Asia/Jakarta")

    def event(self, sentiment_score=0.8, sentiment_label="positive"):
        return {
            "event_id": "evt-1",
            "ticker": "BBCA",
            "published_at_utc": datetime(2026, 8, 11, 2, 0, tzinfo=ZoneInfo("UTC")),
            "event_type": "earnings",
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
        }

    def resolution(self):
        return {
            "event_id": "evt-1",
            "ticker": "BBCA",
            "effective_time": datetime(2026, 8, 11, 9, 0, tzinfo=self.tz),
            "market_session": "regular",
            "resolution_rule": "regular_session",
            "is_trading_day": True,
            "is_tradeable_at_event": True,
            "is_same_session_effective": True,
        }

    def observation(self):
        return {
            "event_id": "evt-1",
            "ticker": "BBCA",
            "effective_time": datetime(2026, 8, 11, 9, 0, tzinfo=self.tz),
            "status": "observed",
            "price_source": "yahoo",
            "price_granularity": "daily",
        }

    def returns(self):
        return {
            "event_id": "evt-1",
            "ticker": "BBCA",
            "effective_time": datetime(2026, 8, 11, 9, 0, tzinfo=self.tz),
            "event_day_return": 0.01,
            "forward_returns": {"t1": 0.02, "t3": 0.03, "t5": 0.04, "t10": 0.05},
            "cumulative_returns": {"t1": 0.02, "t3": 0.03, "t5": 0.04, "t10": 0.05},
            "status": "observed",
        }

    def study(self):
        return {
            "event_id": "evt-1",
            "ticker": "BBCA",
            "effective_time": datetime(2026, 8, 11, 9, 0, tzinfo=self.tz),
            "event_date": "2026-08-11",
            "status": "observed",
            "windows": [
                {"name": "car_-1_1", "car": 0.01},
                {"name": "car_0_1", "car": 0.02},
                {"name": "car_0_3", "car": 0.03},
                {"name": "car_0_5", "car": 0.04},
                {"name": "car_0_10", "car": 0.05},
            ],
        }

    def impact(self):
        return {
            "event_id": "evt-1",
            "ticker": "BBCA",
            "effective_time": datetime(2026, 8, 11, 9, 0, tzinfo=self.tz),
            "abnormal_return": 0.012,
            "statistic": 2.1,
            "p_value": 0.03,
            "significant": True,
            "impact_direction": "POSITIVE",
            "impact_strength": "STRONG",
            "statistical_significance": "SIGNIFICANT",
            "sentiment_alignment": "ALIGNED",
            "impact_label": "POSITIVE_SIGNIFICANT",
            "status": "observed",
        }

    def input(self, **overrides):
        values = {
            "event": self.event(),
            "event_time": self.resolution(),
            "price_observation": self.observation(),
            "return_result": self.returns(),
            "event_study": self.study(),
            "event_impact": self.impact(),
        }
        values.update(overrides)
        return ResearchRecordInput(**values)

    def test_feature_target_separation(self):
        ds = ResearchDatasetBuilder().build([self.input()], generated_at=datetime(2026, 8, 12, 1, tzinfo=ZoneInfo("UTC")))
        record = ds.records[0]
        self.assertEqual(record.status, ResearchDatasetStatus.OBSERVED)
        self.assertIn("sentiment_score", record.features)
        self.assertNotIn("t1_return", record.features)
        self.assertNotIn("car", record.features)
        self.assertNotIn("p_value", record.features)
        self.assertNotIn("impact_label", record.features)
        self.assertEqual(record.targets["t1_return"], 0.02)
        self.assertEqual(record.targets["car"], 0.05)
        self.assertEqual(record.targets["impact_label"], "POSITIVE_SIGNIFICANT")

    def test_phase_outputs_are_preserved(self):
        ds = ResearchDatasetBuilder().build([self.input()])
        targets = ds.records[0].targets
        self.assertEqual(targets["event_day_return"], 0.01)
        self.assertEqual(targets["t3_return"], 0.03)
        self.assertEqual(targets["car_0_3"], 0.03)
        self.assertEqual(targets["p_value"], 0.03)
        self.assertTrue(targets["significant"])

    def test_missing_outcome_is_none_not_zero(self):
        returns = self.returns()
        returns["t10_return"] = None
        ds = ResearchDatasetBuilder().build([self.input(return_result=returns)])
        self.assertIsNone(ds.records[0].targets["t10_return"])
        self.assertEqual(ds.records[0].status, ResearchDatasetStatus.PARTIAL)

    def test_missing_sentiment_is_partial_and_not_target(self):
        event = self.event(sentiment_score=None, sentiment_label=None)
        ds = ResearchDatasetBuilder().build([self.input(event=event)])
        record = ds.records[0]
        self.assertEqual(record.status, ResearchDatasetStatus.PARTIAL)
        self.assertIsNone(record.features["sentiment_score"])
        self.assertIsNone(record.features["sentiment_label"])
        self.assertNotIn("sentiment_score", record.targets)

    def test_identity_mismatch_is_rejected(self):
        bad = self.study()
        bad["ticker"] = "BBRI"
        with self.assertRaises(ValueError):
            ResearchDatasetBuilder().build([self.input(event_study=bad)])

    def test_duplicate_is_explicit(self):
        ds = ResearchDatasetBuilder().build([self.input(), self.input()])
        self.assertEqual(ds.records[0].status, ResearchDatasetStatus.OBSERVED)
        self.assertEqual(ds.records[1].status, ResearchDatasetStatus.DUPLICATE)

    def test_deterministic_ordering(self):
        second = self.input()
        second_event = self.event()
        second_event["event_id"] = "evt-0"
        second_event["ticker"] = "TLKM"
        second_resolution = self.resolution()
        second_resolution["event_id"] = "evt-0"
        second_resolution["ticker"] = "TLKM"
        second_observation = self.observation()
        second_observation["event_id"] = "evt-0"
        second_observation["ticker"] = "TLKM"
        second_returns = self.returns()
        second_returns["event_id"] = "evt-0"
        second_returns["ticker"] = "TLKM"
        second_study = self.study()
        second_study["event_id"] = "evt-0"
        second_study["ticker"] = "TLKM"
        second_impact = self.impact()
        second_impact["event_id"] = "evt-0"
        second_impact["ticker"] = "TLKM"
        second = ResearchRecordInput(second_event, second_resolution, second_observation, second_returns, second_study, second_impact)
        ds = ResearchDatasetBuilder().build([self.input(), second], generated_at=datetime(2026, 8, 12, 1, tzinfo=ZoneInfo("UTC")))
        self.assertEqual([(r.event_id, r.ticker) for r in ds.records], [("evt-0", "TLKM"), ("evt-1", "BBCA")])

    def test_json_is_deterministic_and_json_safe(self):
        generated = datetime(2026, 8, 12, 1, tzinfo=ZoneInfo("UTC"))
        ds = ResearchDatasetBuilder().build([self.input()], generated_at=generated)
        first = ds.to_json()
        second = ds.to_json()
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(payload["records"][0]["targets"]["t10_return"], 0.05)

    def test_features_and_targets_are_disjoint(self):
        ds = ResearchDatasetBuilder().build([self.input()])
        feature_keys = set(ds.features()[0])
        target_keys = set(ds.targets()[0])
        self.assertTrue(feature_keys.isdisjoint(target_keys))
        self.assertTrue({"event_id", "ticker", "effective_time", "sentiment_score"} <= feature_keys)
        self.assertTrue({"event_day_return", "t1_return", "car", "impact_label"} <= target_keys)


if __name__ == "__main__":
    unittest.main()
