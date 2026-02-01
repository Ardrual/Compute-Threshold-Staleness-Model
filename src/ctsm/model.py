from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


def efficiency_multiplier(t: float, *, tau: float) -> float:
    if t < 0:
        raise ValueError("t must be non-negative")
    if tau <= 0:
        raise ValueError("tau must be positive")
    return 2 ** (t / tau)


def effective_compute(c: float, a: float) -> float:
    return a * c


@dataclass(frozen=True)
class DriftModel:
    erisk: float
    tau: float
    update_interval: Optional[float] = None

    def __post_init__(self) -> None:
        if self.erisk <= 0:
            raise ValueError("erisk must be positive")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.update_interval is not None and self.update_interval <= 0:
            raise ValueError("update_interval must be positive")

    def a(self, t: float) -> float:
        return efficiency_multiplier(t, tau=self.tau)

    def effective_compute(self, c: float, t: float) -> float:
        return self.a(t) * c

    def ideal_threshold(self, t: float) -> float:
        return self.erisk / self.a(t)

    def update_time(self, t: float) -> float:
        if t < 0:
            raise ValueError("t must be non-negative")
        if self.update_interval is None:
            return t
        return math.floor(t / self.update_interval) * self.update_interval

    def threshold(self, t: float) -> float:
        if self.update_interval is None:
            return self.ideal_threshold(t)
        tn = self.update_time(t)
        return self.ideal_threshold(tn)

    def staleness(self, t: float) -> float:
        return self.threshold(t) / self.ideal_threshold(t)
