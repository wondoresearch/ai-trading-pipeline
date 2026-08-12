import json
import math
import unittest
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from app.abnormal_return import AbnormalReturnEngine
from app.event_study import CrossSectionalAggregator, EventStudyEngine
from app.expected_return import MarketModelEstimator
from app.historical_return_data import (
    AlignedReturn, EventStudyStatus, HistoricalReturnData, HistoricalReturnDataProvider,
)
from app.price_observation import (
    BaselinePrice, ObservationHorizon, PriceObservation, PriceObservationSet,
    PriceObservationStatus,
)
from app.return_engine import ReturnEngine
from app.statistical_inference import InferenceEngine


class WeekdayCalendar:
    def previous_trading_day(self, value):
        value -= timedelta(days=1)
        while value.weekday() >= 5:
            value -= timedelta(days=1)
        return value

    def next_trading_day(self, value):
        value += timedelta(days=1)
        while value.weekday() >= 5:
            value += timedelta(days=1)
        return value


class FrameProvider:
    def __init__(self, frame, status="ok"):
        self.frame, self.status, self.calls = frame, status, []

    def get_history_with_status(self, *args):
        self.calls.append(args)
        return self.frame.copy(), self.status


class FixedDataProvider:
    def __init__(self, data):
        self.data = data

    def get_returns(self, ticker, event_date, start_offset, end_offset):
        return self.data


class TestPhase6EventStudy(unittest.TestCase):
    def setUp(self):
        self.event_date = date(2026, 8, 11)
        self.calendar = WeekdayCalendar()

    def records(self, start=-250, end=10):
        return tuple(AlignedReturn(offset, self.event_date + timedelta(days=offset),
                                   0.001 + 1.5 * (0.01 + offset / 100000),
                                   0.01 + offset / 100000)
                     for offset in range(start, end + 1))

    def price_set(self):
        tz = ZoneInfo("Asia/Jakarta")
        observations = tuple(PriceObservation(
            horizon, self.event_date + timedelta(days=horizon.trading_day_offset),
            datetime(2026, 8, 11, 15, 50, tzinfo=tz), 100, 100, 100, 100, 100, 1,
            PriceObservationStatus.OBSERVED)
            for horizon in ObservationHorizon)
        return PriceObservationSet("event-1", "BBCA", datetime(2026, 8, 11, 9, tzinfo=tz),
                                   BaselinePrice(self.event_date - timedelta(days=1), datetime(2026, 8, 10, 15, 50, tzinfo=tz), 100, 100, PriceObservationStatus.OBSERVED),
                                   observations, PriceObservationStatus.OBSERVED, "yahoo", "daily")

    def test_market_model_and_expected_return_formula(self):
        records = tuple(AlignedReturn(i, self.event_date, 0.01 + 2 * (i / 10000), i / 10000)
                        for i in range(120))
        model = MarketModelEstimator().fit(records)
        self.assertEqual(model.status, EventStudyStatus.OBSERVED)
        self.assertAlmostEqual(model.alpha, 0.01)
        self.assertAlmostEqual(model.beta, 2.0)
        self.assertAlmostEqual(MarketModelEstimator.expected(model, 0.03), 0.07)

    def test_minimum_120_estimation_observations(self):
        model = MarketModelEstimator().fit(self.records(-250, -132))
        self.assertEqual(model.observations, 119)
        self.assertEqual(model.status, EventStudyStatus.INSUFFICIENT_ESTIMATION_DATA)

    def test_abnormal_return_and_car_formula(self):
        model = MarketModelEstimator().fit(tuple(AlignedReturn(i, self.event_date, 0.01 + 2 * i / 10000, i / 10000) for i in range(120)))
        engine = AbnormalReturnEngine()
        one = engine.calculate(model, AlignedReturn(0, self.event_date, 0.08, 0.03))
        two = engine.calculate(model, AlignedReturn(1, self.event_date, 0.09, 0.03))
        self.assertAlmostEqual(one.abnormal_return, 0.01)
        self.assertAlmostEqual(engine.car([one, two]), 0.03)

    def test_data_provider_requires_price_before_minus_250_and_aligns_benchmark(self):
        dates = pd.date_range("2025-07-01", "2026-09-01", freq="D")
        stock = pd.DataFrame({"Date": dates, "Adj Close": range(100, 100 + len(dates))})
        market = pd.DataFrame({"Date": dates, "Close": range(1000, 1000 + len(dates))})
        stock_provider, market_provider = FrameProvider(stock), FrameProvider(market)
        data = HistoricalReturnDataProvider(stock_provider, market_provider, self.calendar).get_returns("BBCA", self.event_date)
        self.assertIn(-250, data.by_offset())
        self.assertIsInstance(data.by_offset()[-250].trading_date, date)
        self.assertLessEqual(pd.Timestamp(stock_provider.calls[0][1]), pd.Timestamp(self.event_date))
        self.assertEqual(data.status, EventStudyStatus.OBSERVED)

    def test_missing_benchmark_stock_provider_error_and_invalid_data_are_explicit(self):
        dates = pd.date_range("2025-07-01", "2026-09-01", freq="D")
        stock = pd.DataFrame({"Date": dates, "Adj Close": 100.0})
        market = pd.DataFrame({"Date": dates, "Close": 1000.0})
        missing_market = HistoricalReturnDataProvider(FrameProvider(stock), FrameProvider(pd.DataFrame()), self.calendar).get_returns("BBCA", self.event_date)
        provider_error = HistoricalReturnDataProvider(FrameProvider(stock, "provider_error"), FrameProvider(market), self.calendar).get_returns("BBCA", self.event_date)
        invalid = HistoricalReturnDataProvider(FrameProvider(stock.assign(**{"Adj Close": 0.0})), FrameProvider(market), self.calendar).get_returns("BBCA", self.event_date)
        self.assertEqual(missing_market.status, EventStudyStatus.MISSING_BENCHMARK_DATA)
        self.assertEqual(provider_error.status, EventStudyStatus.PROVIDER_ERROR)
        self.assertEqual(invalid.status, EventStudyStatus.INVALID_RETURN_DATA)

    def test_regression_failure_for_constant_market_return(self):
        constant = tuple(AlignedReturn(i, self.event_date, 0.02, 0.01) for i in range(120))
        model = MarketModelEstimator().fit(constant)
        self.assertEqual(model.status, EventStudyStatus.MODEL_FAILURE)

    def test_event_windows_and_phase_five_linkage_validation(self):
        data = HistoricalReturnData("BBCA", self.event_date, self.records(), (), (), EventStudyStatus.OBSERVED)
        observations = self.price_set()
        return_result = ReturnEngine().calculate(observations)
        result = EventStudyEngine(FixedDataProvider(data)).analyze(observations, return_result)
        self.assertEqual([item.name for item in result.windows], ["car_-1_1", "car_0_1", "car_0_3", "car_0_5", "car_0_10"])
        self.assertEqual(result.status, EventStudyStatus.OBSERVED)
        bad = return_result.__class__("other", return_result.ticker, return_result.effective_time, return_result.baseline_date, return_result.baseline_price, return_result.baseline_status, return_result.event_day, return_result.forward_observations, return_result.status, return_result.price_source, return_result.price_granularity)
        with self.assertRaises(ValueError):
            EventStudyEngine(FixedDataProvider(data)).analyze(observations, bad)

    def test_insufficient_event_data_is_explicit(self):
        records = self.records(-250, 0)
        data = HistoricalReturnData("BBCA", self.event_date, records, (), (), EventStudyStatus.OBSERVED)
        result = EventStudyEngine(FixedDataProvider(data)).analyze(self.price_set())
        self.assertEqual(result.status, EventStudyStatus.INSUFFICIENT_EVENT_DATA)
        self.assertTrue(any(item.car is None for item in result.windows))

    def test_bmp_kolari_pynnonen_overlap_adjustment_p_value_and_rejection(self):
        values = [1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 1.15, 0.85, 1.25]
        result = InferenceEngine().bmp_kolari_pynnonen(values, [0.2] * 45, [0.5] * 45)
        self.assertEqual(result.status, EventStudyStatus.OBSERVED)
        self.assertEqual(result.sample_size, 10)
        self.assertEqual(result.average_residual_correlation, 0.2)
        self.assertEqual(result.average_overlap, 0.5)
        self.assertLess(result.adjustment_factor, 1)
        self.assertGreaterEqual(result.p_value, 0)
        self.assertLessEqual(result.p_value, 1)
        self.assertEqual(result.rejection_at_0_05, result.p_value < 0.05)

    def test_kolari_pynnonen_does_not_scale_rho_by_overlap(self):
        values = [1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 1.15, 0.85, 1.25]
        result = InferenceEngine().bmp_kolari_pynnonen(values, [0.2] * 45, [0.5] * 45)
        expected_factor = math.sqrt((1 - 0.2) / (1 + 9 * 0.2))
        self.assertAlmostEqual(result.average_residual_correlation, 0.2)
        self.assertAlmostEqual(result.average_overlap, 0.5)
        self.assertAlmostEqual(result.adjustment_factor, expected_factor)

    def test_kolari_pynnonen_without_overlap_metadata_uses_rho_directly(self):
        values = [1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 1.15, 0.85, 1.25]
        result = InferenceEngine().bmp_kolari_pynnonen(values, [0.2] * 45, [])
        expected_factor = math.sqrt((1 - 0.2) / (1 + 9 * 0.2))
        self.assertIsNone(result.average_overlap)
        self.assertAlmostEqual(result.adjustment_factor, expected_factor)

    def test_aar_caar_and_minimum_cross_section(self):
        data = HistoricalReturnData("BBCA", self.event_date, self.records(), (), (), EventStudyStatus.OBSERVED)
        result = EventStudyEngine(FixedDataProvider(data)).analyze(self.price_set())
        small = CrossSectionalAggregator().aggregate([result] * 9, "car_0_1")
        full = CrossSectionalAggregator().aggregate([result] * 10, "car_0_1")
        self.assertEqual(small.status, EventStudyStatus.INSUFFICIENT_CROSS_SECTION)
        self.assertEqual(full.sample_size, 10)
        self.assertEqual(len(full.aar), 2)
        self.assertIsNotNone(full.caar)

    def test_serialization_is_deterministic_and_json_safe(self):
        data = HistoricalReturnData("BBCA", self.event_date, self.records(), (), (), EventStudyStatus.OBSERVED)
        result = EventStudyEngine(FixedDataProvider(data)).analyze(self.price_set()).to_dict()
        self.assertEqual(result, EventStudyEngine(FixedDataProvider(data)).analyze(self.price_set()).to_dict())
        self.assertIsInstance(json.dumps(result, allow_nan=False), str)


if __name__ == "__main__":
    unittest.main()
