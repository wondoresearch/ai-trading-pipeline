
"""Configuration for the research-only opportunity runner."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpportunityRunnerConfig:
    output_path: Path = Path("output/opportunity_ranking.json")
    minimum_history: int = 60
    risk_minimum_history: int = 20
    top_n: int = 10

    def __post_init__(self) -> None:
        if self.minimum_history <= 0:
            raise ValueError("minimum_history must be positive")
        if self.risk_minimum_history <= 0:
            raise ValueError("risk_minimum_history must be positive")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
