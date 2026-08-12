import json
import unittest

from app.event_impact import (
    EventImpactEngine,
    ImpactDirection,
    ImpactLabel,
    ImpactStatus,
    ImpactStrength,
    ImpactThresholds,
    SentimentAlignment,
    StatisticalSignificance,
)


class O:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestEventImpactEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EventImpactEngine()
        self.event = O(event_id="e1", ticker="BBCA")
        self.sent_pos = O(label="POSITIVE")
        self.sent_neg = O(label="NEGATIVE")
        self.time = O(effective_time="2026-08-11T09:00:00+07:00")
        self.ret = O(event_day_return=0.03, ticker="BBCA")

    def study(self, car, significant=True, p_value=None):
        return O(
            windows=[O(window="car_0_1", car=car)],
            abnormal_return=0.02,
            statistic=2.1,
            p_value=p_value,
            significant=significant,
        )

    def test_positive_significant(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(.06, True))
        self.assertEqual(r.impact_label, ImpactLabel.POSITIVE_SIGNIFICANT)
        self.assertEqual(r.impact_strength, ImpactStrength.HIGH)

    def test_positive_insignificant(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(.03, False))
        self.assertEqual(r.impact_label, ImpactLabel.POSITIVE_INSIGNIFICANT)

    def test_negative_significant(self):
        r = self.engine.build(self.event, self.sent_neg, self.time, self.ret, self.study(-.06, True))
        self.assertEqual(r.impact_label, ImpactLabel.NEGATIVE_SIGNIFICANT)

    def test_negative_insignificant(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(-.03, False))
        self.assertEqual(r.impact_label, ImpactLabel.NEGATIVE_INSIGNIFICANT)
        self.assertEqual(r.sentiment_alignment, SentimentAlignment.CONTRADICTED)

    def test_neutral(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(0.0, False))
        self.assertEqual(r.impact_direction, ImpactDirection.NEUTRAL)
        self.assertEqual(r.impact_label, ImpactLabel.NEUTRAL)
        self.assertEqual(r.sentiment_alignment, SentimentAlignment.NEUTRAL)

    def test_unknown_without_significance(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(.03, None, None))
        self.assertEqual(r.statistical_significance, StatisticalSignificance.UNKNOWN)
        self.assertEqual(r.impact_label, ImpactLabel.UNKNOWN)

    def test_positive_sentiment_positive_reaction_aligned(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(.03))
        self.assertEqual(r.sentiment_alignment, SentimentAlignment.ALIGNED)

    def test_positive_sentiment_negative_reaction_contradicted(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(-.03))
        self.assertEqual(r.sentiment_alignment, SentimentAlignment.CONTRADICTED)

    def test_negative_sentiment_negative_reaction_aligned(self):
        r = self.engine.build(self.event, self.sent_neg, self.time, self.ret, self.study(-.03))
        self.assertEqual(r.sentiment_alignment, SentimentAlignment.ALIGNED)

    def test_missing_sentiment(self):
        r = self.engine.build(self.event, None, self.time, self.ret, self.study(.03))
        self.assertEqual(r.status, ImpactStatus.MISSING_SENTIMENT)
        self.assertIsNone(r.car)
        self.assertIsNone(r.abnormal_return)
        self.assertIsNone(r.statistic)
        self.assertIsNone(r.p_value)
        self.assertIsNone(r.significant)
        self.assertEqual(r.impact_direction, ImpactDirection.UNKNOWN)
        self.assertEqual(r.impact_label, ImpactLabel.UNKNOWN)

    def test_missing_return(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, None, self.study(.03))
        self.assertEqual(r.status, ImpactStatus.MISSING_RETURN)

    def test_missing_event_study(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, None)
        self.assertEqual(r.status, ImpactStatus.MISSING_EVENT_STUDY)

    def test_insufficient_statistics(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, O(windows=[]))
        self.assertEqual(r.status, ImpactStatus.INSUFFICIENT_STATISTICS)
        self.assertIsNone(r.car)
        self.assertEqual(r.impact_label, ImpactLabel.UNKNOWN)

    def test_invalid_input(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(.03, "false"))
        self.assertEqual(r.status, ImpactStatus.INVALID_INPUT)
        self.assertEqual(r.impact_label, ImpactLabel.UNKNOWN)

    def test_missing_event_time_is_invalid_input(self):
        r = self.engine.build(self.event, self.sent_pos, None, self.ret, self.study(.03))
        self.assertEqual(r.status, ImpactStatus.INVALID_INPUT)

    def test_multiticker_isolation(self):
        event = O(event_id="e2", ticker="BMRI")
        ret = O(event_day_return=.01, ticker="BMRI")
        r = self.engine.build(event, self.sent_pos, self.time, ret, self.study(.03))
        self.assertEqual((r.event_id, r.ticker), ("e2", "BMRI"))

    def test_none_is_not_zero(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, O(windows=[]))
        self.assertIsNone(r.car)
        self.assertIsNone(r.abnormal_return)
        self.assertNotEqual(r.car, 0)

    def test_phase6_window_is_explicit(self):
        study = O(windows=[
            O(window="car_0_5", car=.50),
            O(window="car_0_1", car=.03),
        ], significant=False)
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, study)
        self.assertEqual(r.car, .03)

    def test_p_value_can_determine_significance(self):
        study = O(windows=[O(window="car_0_1", car=.03)], significant=None, p_value=.01)
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, study)
        self.assertTrue(r.significant)
        self.assertEqual(r.statistical_significance, StatisticalSignificance.SIGNIFICANT)

    def test_deterministic_json(self):
        r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(.03))
        first = r.to_json()
        second = r.to_json()
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["impact_label"], "POSITIVE_SIGNIFICANT")

    def test_threshold_boundaries(self):
        for car, expected in (
            (.019999, ImpactStrength.LOW),
            (.02, ImpactStrength.MEDIUM),
            (.049999, ImpactStrength.MEDIUM),
            (.05, ImpactStrength.HIGH),
        ):
            r = self.engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(car, False))
            self.assertEqual(r.impact_strength, expected)

    def test_custom_thresholds(self):
        engine = EventImpactEngine(ImpactThresholds(.10, .20))
        r = engine.build(self.event, self.sent_pos, self.time, self.ret, self.study(.06, True))
        self.assertEqual(r.impact_strength, ImpactStrength.LOW)


if __name__ == "__main__":
    unittest.main()
