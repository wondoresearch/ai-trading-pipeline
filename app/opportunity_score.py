"""Risk-adjusted opportunity scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpportunityScore:
    expected_return: float
    confidence: float
    risk_penalty: float
    score: float


class OpportunityScorer:
    """Combine expected return, confidence and risk.

    The score is intentionally transparent rather than a black-box model.
    All inputs are point-in-time values supplied by the caller.
    """

    def score(
        self,
        expected_return: float,
        confidence: float,
        volatility: float,
        downside_deviation: float,
        max_drawdown: float,
        beta: float | None = None,
    ) -> OpportunityScore:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if volatility < 0 or downside_deviation < 0 or max_drawdown < 0:
            raise ValueError("risk values must be non-negative")

        beta_penalty = 0.0
        if beta is not None:
            beta_penalty = max(abs(beta) - 1.0, 0.0) * 0.02

        risk_penalty = (
            0.40 * volatility
            + 0.35 * downside_deviation
            + 0.20 * max_drawdown
            + beta_penalty
        )

        score = (
            expected_return * (0.60 + 0.40 * confidence)
            - risk_penalty
        )

        return OpportunityScore(
            expected_return=float(expected_return),
            confidence=float(confidence),
            risk_penalty=float(risk_penalty),
            score=float(score),
        )
