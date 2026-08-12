import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.research_dataset import ResearchDataset, ResearchRecord, ResearchDatasetStatus
from app.dataset_validator import DatasetValidator, ValidationStatus


class TestDatasetValidator(unittest.TestCase):
    tz = ZoneInfo("Asia/Jakarta")

    def records(self, n=10, target_values=None):
        target_values = target_values or ["POSITIVE"] * n
        result = []
        for i in range(n):
            t = datetime(2026, 1, 1, 9, 0, tzinfo=self.tz) + timedelta(days=i)
            result.append(
                ResearchRecord(
                    event_id=f"evt-{i}",
                    ticker="BBCA",
                    status=ResearchDatasetStatus.OBSERVED,
                    features={
                        "event_id": f"evt-{i}",
                        "ticker": "BBCA",
                        "effective_time": t.isoformat(),
                        "event_type": "news",
                        "sentiment_score": 0.5,
                    },
                    targets={
                        "event_day_return": 0.01,
                        "t1_return": 0.02,
                        "impact_label": target_values[i],
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

    def test_clean_dataset_passes(self):
        report = DatasetValidator().validate(self.dataset(self.records(10)))
        self.assertEqual(report.status, ValidationStatus.PASS)
        self.assertEqual(len(report.time_split.train), 7)
        self.assertEqual(len(report.time_split.validation), 1)
        self.assertEqual(len(report.time_split.test), 2)

    def test_future_outcome_in_feature_is_leakage(self):
        records = list(self.records(10))
        records[0].features["t1_return"] = 0.9
        report = DatasetValidator().validate(self.dataset(records))
        self.assertEqual(report.status, ValidationStatus.FAIL)
        issue = next(x for x in report.checks if x.check == "leakage_detection")
        self.assertEqual(issue.status, ValidationStatus.FAIL)

    def test_duplicate_key_fails(self):
        records = list(self.records(10))
        records[1] = records[0]
        report = DatasetValidator().validate(self.dataset(records))
        self.assertEqual(report.status, ValidationStatus.FAIL)

    def test_missing_feature_warns(self):
        records = list(self.records(10))
        records[0].features["sentiment_score"] = None
        records[1].features["sentiment_score"] = None
        records[2].features["sentiment_score"] = None
        report = DatasetValidator().validate(self.dataset(records))
        issue = next(x for x in report.checks if x.check == "missing_data_analysis")
        self.assertEqual(issue.status, ValidationStatus.WARN)

    def test_missing_target_fails_if_all_missing(self):
        records = list(self.records(10))
        for record in records:
            record.targets["impact_label"] = None
        report = DatasetValidator().validate(self.dataset(records))
        issue = next(x for x in report.checks if x.check == "target_validation")
        self.assertEqual(issue.status, ValidationStatus.FAIL)

    def test_rare_class_warns(self):
        labels = ["POSITIVE"] * 9 + ["NEGATIVE"]
        report = DatasetValidator().validate(self.dataset(self.records(10, labels)), min_class_count=2)
        issue = next(x for x in report.checks if x.check == "class_balance")
        self.assertEqual(issue.status, ValidationStatus.WARN)

    def test_naive_datetime_fails(self):
        records = list(self.records(10))
        records[0].features["effective_time"] = "2026-01-01T09:00:00"
        report = DatasetValidator().validate(self.dataset(records))
        issue = next(x for x in report.checks if x.check == "time_order_validation")
        self.assertEqual(issue.status, ValidationStatus.FAIL)

    def test_time_split_is_chronological(self):
        records = list(self.records(10))
        report = DatasetValidator().validate(self.dataset(records))
        split = report.time_split
        self.assertLess(max(split.train), min(split.validation))
        self.assertLess(max(split.validation), min(split.test))

    def test_deterministic_json(self):
        report = DatasetValidator().validate(self.dataset(self.records(10)))
        self.assertEqual(report.to_json(), report.to_json())


if __name__ == "__main__":
    unittest.main()
