from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .abnormal_return import AbnormalReturn, AbnormalReturnEngine
from .expected_return import MarketModel, MarketModelEstimator
from .historical_return_data import EventStudyStatus, HistoricalReturnDataProvider
from .price_observation import ObservationHorizon, PriceObservationSet
from .return_engine import ReturnResult
from .statistical_inference import InferenceEngine, InferenceResult


@dataclass(frozen=True)
class CARWindow:
    name: str
    start_offset: int
    end_offset: int
    car: Optional[float]
    standardized_car: Optional[float]
    status: EventStudyStatus

    def to_dict(self):
        return {"name": self.name, "start_offset": self.start_offset,
                "end_offset": self.end_offset, "car": self.car,
                "standardized_car": self.standardized_car, "status": self.status.value}


@dataclass(frozen=True)
class EventStudyResult:
    event_id: str
    ticker: str
    effective_time: object
    event_date: date
    status: EventStudyStatus
    model: MarketModel
    abnormal_returns: Tuple[AbnormalReturn, ...]
    windows: Tuple[CARWindow, ...]
    estimation_residuals: Tuple[Tuple[date, float], ...]

    def to_dict(self):
        return {"event_id": self.event_id, "ticker": self.ticker,
                "effective_time": self.effective_time.isoformat(), "event_date": self.event_date.isoformat(),
                "status": self.status.value,
                "model": {"alpha": self.model.alpha, "beta": self.model.beta,
                          "residual_variance": self.model.residual_variance,
                          "observations": self.model.observations, "status": self.model.status.value},
                "abnormal_returns": [{"offset": item.offset, "trading_date": item.trading_date.isoformat(),
                                        "actual_return": item.actual_return, "market_return": item.market_return,
                                        "expected_return": item.expected_return, "abnormal_return": item.abnormal_return}
                                      for item in self.abnormal_returns],
                "windows": [item.to_dict() for item in self.windows]}


class EventStudyEngine:
    WINDOWS = (("car_-1_1", -1, 1), ("car_0_1", 0, 1), ("car_0_3", 0, 3),
               ("car_0_5", 0, 5), ("car_0_10", 0, 10))

    def __init__(self, data_provider: Optional[HistoricalReturnDataProvider] = None,
                 estimator: Optional[MarketModelEstimator] = None,
                 abnormal_engine: Optional[AbnormalReturnEngine] = None) -> None:
        self.data_provider = data_provider or HistoricalReturnDataProvider()
        self.estimator = estimator or MarketModelEstimator()
        self.abnormal_engine = abnormal_engine or AbnormalReturnEngine()

    def analyze(self, observations: PriceObservationSet,
                return_result: Optional[ReturnResult] = None) -> EventStudyResult:
        self._validate_linkage(observations, return_result)
        event_observation = next((item for item in observations.observations
                                  if item.horizon is ObservationHorizon.EVENT_DAY), None)
        event_date = event_observation.observation_date if event_observation else observations.effective_time.date()
        data = self.data_provider.get_returns(observations.ticker, event_date, -250, 10)
        estimation = [item for item in data.returns if -250 <= item.offset <= -30]
        model = self.estimator.fit(estimation)
        if data.status is EventStudyStatus.PROVIDER_ERROR:
            return self._empty(observations, event_date, model, EventStudyStatus.PROVIDER_ERROR)
        if data.status is EventStudyStatus.INVALID_RETURN_DATA:
            return self._empty(observations, event_date, model, EventStudyStatus.INVALID_RETURN_DATA)
        if model.status is not EventStudyStatus.OBSERVED:
            return self._empty(observations, event_date, model, model.status)
        records = data.by_offset()
        daily = tuple(self.abnormal_engine.calculate(model, records[offset]) for offset in range(-1, 11) if offset in records)
        daily = tuple(item for item in daily if item is not None)
        windows = tuple(self._window(name, start, end, daily) for name, start, end in self.WINDOWS)
        status = EventStudyStatus.OBSERVED if all(item.status is EventStudyStatus.OBSERVED for item in windows) else EventStudyStatus.INSUFFICIENT_EVENT_DATA
        residuals = tuple((item.trading_date, item.stock_return - (model.alpha + model.beta * item.market_return)) for item in estimation)
        return EventStudyResult(observations.event_id, observations.ticker, observations.effective_time,
                                event_date, status, model, daily, windows, residuals)

    def _window(self, name, start, end, daily):
        selected = [item for item in daily if start <= item.offset <= end]
        if len(selected) != end - start + 1:
            return CARWindow(name, start, end, None, None, EventStudyStatus.INSUFFICIENT_EVENT_DATA)
        car = self.abnormal_engine.car(selected)
        variance = sum(item.prediction_variance for item in selected)
        standardized = car / sqrt(variance) if variance > 0 else None
        return CARWindow(name, start, end, car, standardized,
                         EventStudyStatus.OBSERVED if standardized is not None else EventStudyStatus.MODEL_FAILURE)

    def _empty(self, observations, event_date, model, status):
        windows = tuple(CARWindow(name, start, end, None, None, status) for name, start, end in self.WINDOWS)
        return EventStudyResult(observations.event_id, observations.ticker, observations.effective_time,
                                event_date, status, model, (), windows, ())

    @staticmethod
    def _validate_linkage(observations, result):
        if result is None:
            return
        if (result.event_id != observations.event_id or result.ticker != observations.ticker
                or result.effective_time != observations.effective_time):
            raise ValueError("ReturnResult does not match PriceObservationSet")


@dataclass(frozen=True)
class CrossSectionalResult:
    window: str
    aar: Tuple[Tuple[int, Optional[float]], ...]
    caar: Optional[float]
    sample_size: int
    inference: InferenceResult
    status: EventStudyStatus

    def to_dict(self):
        return {"window": self.window, "aar": [{"offset": offset, "value": value} for offset, value in self.aar],
                "caar": self.caar, "sample_size": self.sample_size,
                "inference": self.inference.to_dict(), "status": self.status.value}


class CrossSectionalAggregator:
    """Produces AAR/CAAR and dependence-robust inference for one CAR window."""

    def __init__(self, inference: Optional[InferenceEngine] = None) -> None:
        self.inference = inference or InferenceEngine()

    def aggregate(self, results: Sequence[EventStudyResult], window: str) -> CrossSectionalResult:
        eligible = [item for item in results if item.status is EventStudyStatus.OBSERVED
                    and next((item2 for item2 in item.windows if item2.name == window and item2.status is EventStudyStatus.OBSERVED), None)]
        chosen = [next(item2 for item2 in item.windows if item2.name == window) for item in eligible]
        offsets = range(chosen[0].start_offset, chosen[0].end_offset + 1) if chosen else ()
        aar = tuple((offset, float(np.mean([next(a.abnormal_return for a in item.abnormal_returns if a.offset == offset) for item in eligible]))) for offset in offsets)
        values = [item.standardized_car for item in chosen]
        correlations = self._correlations(eligible)
        overlaps = self._overlaps(eligible, chosen[0].start_offset, chosen[0].end_offset) if chosen else []
        inference = self.inference.bmp_kolari_pynnonen(values, correlations, overlaps)
        return CrossSectionalResult(window, aar, float(np.mean([item.car for item in chosen])) if chosen else None,
                                    len(chosen), inference, inference.status)

    @staticmethod
    def _correlations(results):
        values = []
        for left_index, left in enumerate(results):
            left_map = dict(left.estimation_residuals)
            for right in results[left_index + 1:]:
                right_map = dict(right.estimation_residuals)
                shared = sorted(set(left_map) & set(right_map))
                if len(shared) > 1:
                    values.append(float(np.corrcoef([left_map[d] for d in shared], [right_map[d] for d in shared])[0, 1]))
        return [item for item in values if np.isfinite(item)]

    @staticmethod
    def _overlaps(results, start, end):
        dates = [{a.trading_date for a in item.abnormal_returns if start <= a.offset <= end} for item in results]
        return [len(dates[i] & dates[j]) / (end - start + 1) for i in range(len(dates)) for j in range(i + 1, len(dates))]
