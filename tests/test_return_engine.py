import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.price_observation import (
    BaselinePrice,
    ObservationHorizon,
    PriceObservation,
    PriceObservationSet,
    PriceObservationStatus,
)
from app.return_engine import ReturnEngine, ReturnStatus


class TestReturnEngine(unittest.TestCase):
    def setUp(self):
        self.tz = ZoneInfo("Asia/Jakarta")
        self.engine = ReturnEngine()

    def observation(self, horizon, price, status=PriceObservationStatus.OBSERVED):
        return PriceObservation(
            horizon=horizon,
            observation_date=date(2026, 8, 11 + horizon.trading_day_offset),
            observation_time=datetime(
                2026, 8, 11 + horizon.trading_day_offset, 15, 50, tzinfo=self.tz
            ),
            open=price,
            high=price,
            low=price,
            close=price,
            adjusted_close=price,
            volume=1000.0,
            status=status,
        )

    def observation_set(self, baseline_price=100.0,
                        baseline_status=PriceObservationStatus.OBSERVED,
                        observations=None):
        return PriceObservationSet(
            event_id="evt-1",
            ticker="BBCA",
            effective_time=datetime(2026, 8, 11, 9, 0, tzinfo=self.tz),
            baseline=BaselinePrice(
                baseline_date=date(2026, 8, 10),
                baseline_time=datetime(2026, 8, 10, 15, 50, tzinfo=self.tz),
                raw_close=baseline_price,
                adjusted_close=baseline_price,
                status=baseline_status,
            ),
            observations=tuple(observations if observations is not None else [
                self.observation(ObservationHorizon.EVENT_DAY, 110.0),
                self.observation(ObservationHorizon.T1, 121.0),
                self.observation(ObservationHorizon.T3, 99.0),
                self.observation(ObservationHorizon.T5, 132.0),
                self.observation(ObservationHorizon.T10, 88.0),
            ]),
            status=PriceObservationStatus.OBSERVED,
            price_source="yahoo",
            price_granularity="daily",
        )

    @staticmethod
    def for_horizon(result, horizon):
        return next(item for item in result.forward_observations if item.horizon is horizon)

    def test_normal_event_day_return_uses_adjusted_close(self):
        result = self.engine.calculate(self.observation_set())
        self.assertEqual(result.event_day.adjusted_close, 110.0)
        self.assertAlmostEqual(result.event_day.event_day_return, 0.10)
        self.assertEqual(result.event_day.status, ReturnStatus.OBSERVED)

    def test_normal_t1_t3_t5_t10_forward_returns(self):
        result = self.engine.calculate(self.observation_set())
        expected = {
            ObservationHorizon.T1: 0.10,
            ObservationHorizon.T3: -0.10,
            ObservationHorizon.T5: 0.20,
            ObservationHorizon.T10: -0.20,
        }
        self.assertEqual([item.horizon for item in result.forward_observations], list(expected))
        for horizon, value in expected.items():
            self.assertAlmostEqual(self.for_horizon(result, horizon).forward_return, value)

    def test_cumulative_returns_use_phase_four_baseline(self):
        result = self.engine.calculate(self.observation_set())
        expected = {
            ObservationHorizon.T1: 0.21,
            ObservationHorizon.T3: -0.01,
            ObservationHorizon.T5: 0.32,
            ObservationHorizon.T10: -0.12,
        }
        for horizon, value in expected.items():
            self.assertAlmostEqual(self.for_horizon(result, horizon).cumulative_return, value)

    def test_event_day_forward_return_is_undefined(self):
        payload = self.engine.calculate(self.observation_set()).to_dict()["event_day"]
        self.assertNotIn("forward_return", payload)
        self.assertAlmostEqual(payload["event_day_return"], 0.10)

    def test_missing_baseline_preserves_valid_forward_returns(self):
        result = self.engine.calculate(self.observation_set(
            baseline_price=None,
            baseline_status=PriceObservationStatus.MISSING_BASELINE,
        ))
        t1 = self.for_horizon(result, ObservationHorizon.T1)
        self.assertEqual(result.status, ReturnStatus.MISSING_BASELINE)
        self.assertIsNone(result.event_day.event_day_return)
        self.assertEqual(result.event_day.status, ReturnStatus.MISSING_BASELINE)
        self.assertIsNone(t1.cumulative_return)
        self.assertEqual(t1.cumulative_status, ReturnStatus.MISSING_BASELINE)
        self.assertAlmostEqual(t1.forward_return, 0.10)
        self.assertEqual(t1.forward_status, ReturnStatus.OBSERVED)

    def test_missing_event_day_blocks_forward_but_not_cumulative_returns(self):
        observations = [
            self.observation(ObservationHorizon.EVENT_DAY, None,
                             PriceObservationStatus.MISSING_OBSERVATION),
            self.observation(ObservationHorizon.T1, 121.0),
            self.observation(ObservationHorizon.T3, 99.0),
            self.observation(ObservationHorizon.T5, 132.0),
            self.observation(ObservationHorizon.T10, 88.0),
        ]
        result = self.engine.calculate(self.observation_set(observations=observations))
        t1 = self.for_horizon(result, ObservationHorizon.T1)
        self.assertEqual(result.status, ReturnStatus.MISSING_EVENT_DAY)
        self.assertEqual(result.event_day.status, ReturnStatus.MISSING_EVENT_DAY)
        self.assertIsNone(t1.forward_return)
        self.assertEqual(t1.forward_status, ReturnStatus.MISSING_EVENT_DAY)
        self.assertAlmostEqual(t1.cumulative_return, 0.21)

    def test_missing_individual_forward_observation_is_explicit(self):
        observations = [
            self.observation(ObservationHorizon.EVENT_DAY, 110.0),
            self.observation(ObservationHorizon.T1, 121.0),
            self.observation(ObservationHorizon.T3, None,
                             PriceObservationStatus.MISSING_OBSERVATION),
            self.observation(ObservationHorizon.T5, 132.0),
            self.observation(ObservationHorizon.T10, 88.0),
        ]
        result = self.engine.calculate(self.observation_set(observations=observations))
        t3 = self.for_horizon(result, ObservationHorizon.T3)
        self.assertEqual(result.status, ReturnStatus.MISSING_FORWARD_OBSERVATION)
        self.assertIsNone(t3.cumulative_return)
        self.assertIsNone(t3.forward_return)
        self.assertEqual(t3.cumulative_status, ReturnStatus.MISSING_FORWARD_OBSERVATION)
        self.assertEqual(t3.forward_status, ReturnStatus.MISSING_FORWARD_OBSERVATION)

    def test_provider_error_is_preserved(self):
        result = self.engine.calculate(self.observation_set(
            baseline_price=None,
            baseline_status=PriceObservationStatus.PROVIDER_ERROR,
        ))
        self.assertEqual(result.status, ReturnStatus.PROVIDER_ERROR)
        self.assertEqual(result.event_day.status, ReturnStatus.PROVIDER_ERROR)
        self.assertTrue(all(
            item.cumulative_status is ReturnStatus.PROVIDER_ERROR
            for item in result.forward_observations
        ))

    def test_invalid_data_is_preserved(self):
        result = self.engine.calculate(self.observation_set(
            baseline_price=None,
            baseline_status=PriceObservationStatus.INVALID_DATA,
        ))
        self.assertEqual(result.status, ReturnStatus.INVALID_DATA)
        self.assertIsNone(result.event_day.event_day_return)
        self.assertEqual(result.event_day.status, ReturnStatus.INVALID_DATA)

    def test_zero_baseline_is_explicit_and_never_divides_by_zero(self):
        result = self.engine.calculate(self.observation_set(baseline_price=0.0))
        t1 = self.for_horizon(result, ObservationHorizon.T1)
        self.assertEqual(result.status, ReturnStatus.ZERO_BASELINE)
        self.assertIsNone(result.event_day.event_day_return)
        self.assertEqual(result.event_day.status, ReturnStatus.ZERO_BASELINE)
        self.assertIsNone(t1.cumulative_return)
        self.assertEqual(t1.cumulative_status, ReturnStatus.ZERO_BASELINE)
        self.assertAlmostEqual(t1.forward_return, 0.10)

    def test_zero_event_day_is_explicit_and_never_divides_by_zero(self):
        observations = [
            self.observation(ObservationHorizon.EVENT_DAY, 0.0),
            self.observation(ObservationHorizon.T1, 121.0),
            self.observation(ObservationHorizon.T3, 99.0),
            self.observation(ObservationHorizon.T5, 132.0),
            self.observation(ObservationHorizon.T10, 88.0),
        ]

        result = self.engine.calculate(self.observation_set(observations=observations))
        t1 = self.for_horizon(result, ObservationHorizon.T1)

        self.assertEqual(result.baseline_price, 100.0)
        self.assertEqual(result.event_day.adjusted_close, 0.0)
        self.assertEqual(result.status, ReturnStatus.ZERO_EVENT_DAY)
        self.assertIsNone(result.event_day.event_day_return)
        self.assertEqual(result.event_day.status, ReturnStatus.ZERO_EVENT_DAY)
        self.assertIsNone(t1.forward_return)
        self.assertEqual(t1.forward_status, ReturnStatus.ZERO_EVENT_DAY)
        self.assertAlmostEqual(t1.cumulative_return, 0.21)

    def test_serialization_is_deterministic_and_preserves_phase_four_semantics(self):
        observations = self.observation_set()
        first = self.engine.calculate(observations).to_dict()
        second = self.engine.calculate(observations).to_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["event_id"], "evt-1")
        self.assertEqual(first["ticker"], "BBCA")
        self.assertEqual(first["effective_time"], "2026-08-11T09:00:00+07:00")
        self.assertEqual(first["baseline_date"], "2026-08-10")
        self.assertEqual(first["baseline_price"], 100.0)
        self.assertEqual(first["price_source"], "yahoo")
        self.assertEqual(first["price_granularity"], "daily")


if __name__ == "__main__":
    unittest.main()
