import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.research_dataset import ResearchDataset, ResearchRecord, ResearchDatasetStatus
from app.feature_engineering import FeatureEngineer, FeatureEngineeringStatus


class TestFeatureEngineer(unittest.TestCase):
    tz = ZoneInfo("Asia/Jakarta")

    def records(self, n=6):
        result = []
        labels = ["positive", "negative", "positive", "neutral", "positive", "negative"]
        types = ["earnings", "macro", "earnings", "policy", "earnings", "macro"]
        for i in range(n):
            t = datetime(2026, 1, 5, 9, 0, tzinfo=self.tz) + timedelta(days=i)
            result.append(
                ResearchRecord(
                    event_id=f"evt-{i}",
                    ticker="BBCA",
                    status=ResearchDatasetStatus.OBSERVED,
                    features={
                        "event_id": f"evt-{i}",
                        "ticker": "BBCA",
                        "effective_time": t.isoformat(),
                        "event_type": types[i % len(types)],
                        "sentiment_score": [-0.5, 0.2, 0.8, 0.0, 0.4, -0.2][i],
                        "sentiment_label": labels[i],
                    },
                    targets={
                        "t1_return": 0.01 * (i + 1),
                        "impact_label": "POSITIVE",
                    },
                    metadata={},
                )
            )
        return tuple(result)

    def dataset(self, records):
        return ResearchDataset(
            "phase8-v1", "research-event-v1",
            "2026-01-20T00:00:00+00:00", tuple(records)
        )

    def test_fit_transform_is_ready(self):
        ds = self.dataset(self.records())
        result = FeatureEngineer().fit_transform(ds)
        self.assertEqual(result.status, FeatureEngineeringStatus.READY)
        self.assertEqual(len(result.records), 6)
        self.assertEqual(len(result.feature_names), len(result.records[0].values))

    def test_targets_are_not_used_as_features(self):
        ds = self.dataset(self.records())
        result = FeatureEngineer().fit_transform(ds)
        self.assertNotIn("t1_return", result.feature_names)
        self.assertNotIn("impact_label", result.feature_names)

    def test_leakage_in_features_is_rejected(self):
        records = list(self.records())
        records[0].features["t1_return"] = 0.5
        with self.assertRaises(ValueError):
            FeatureEngineer().fit(self.dataset(records))

    def test_scaler_is_fitted_on_training_only(self):
        train = self.dataset(self.records(4))
        engineer = FeatureEngineer()
        engineer.fit(train)
        first = engineer.transform(train).records[0].values[0]
        # Same first observation must remain identical after transforming
        # a separate dataset containing an extreme validation observation.
        validation = self.dataset(self.records())
        again = engineer.transform(validation).records[0].values[0]
        self.assertEqual(first, again)

    def test_unseen_category_is_all_zero(self):
        engineer = FeatureEngineer().fit(self.dataset(self.records(4)))
        records = list(self.records(6))
        records[5].features["event_type"] = "unseen_future_type"
        result = engineer.transform(self.dataset([records[5]]))
        self.assertEqual(result.status, FeatureEngineeringStatus.READY)
        names = result.feature_names
        category_indexes = [
            i for i, name in enumerate(names) if name.startswith("event_type=")
        ]
        self.assertTrue(all(result.records[0].values[i] == 0.0 for i in category_indexes))

    def test_missing_numeric_is_explicitly_encoded_zero_after_scaling(self):
        records = list(self.records())
        records[0].features["sentiment_score"] = None
        engineer = FeatureEngineer().fit(self.dataset(records))
        result = engineer.transform(self.dataset([records[0]]))
        self.assertEqual(result.records[0].values[0], 0.0)

    def test_naive_time_is_rejected(self):
        records = list(self.records())
        records[0].features["effective_time"] = "2026-01-05T09:00:00"
        engineer = FeatureEngineer().fit(self.dataset(self.records()))
        result = engineer.transform(self.dataset([records[0]]))
        self.assertEqual(result.status, FeatureEngineeringStatus.INVALID_INPUT)

    def test_deterministic_feature_names_and_json(self):
        ds = self.dataset(self.records())
        engineer_a = FeatureEngineer().fit(ds)
        engineer_b = FeatureEngineer().fit(ds)
        a = engineer_a.transform(ds)
        b = engineer_b.transform(ds)
        self.assertEqual(a.feature_names, b.feature_names)
        self.assertEqual(a.to_json(), b.to_json())

    def test_transform_before_fit_is_explicit(self):
        result = FeatureEngineer().transform(self.dataset(self.records()))
        self.assertEqual(result.status, FeatureEngineeringStatus.NOT_FITTED)


if __name__ == "__main__":
    unittest.main()
