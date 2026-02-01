from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import numpy as np

_SCENARIO_ALIASES = {
    "baseline": "baseline",
    "default": "baseline",
    "threshold-sensitive": "threshold-sensitive",
    "threshold_sensitive": "threshold-sensitive",
    "strategic": "threshold-sensitive",
    "gaming": "threshold-sensitive",
}


def efficiency_multiplier(t: float, *, tau: float) -> float:
    if t < 0:
        raise ValueError("t must be non-negative")
    if tau <= 0:
        raise ValueError("tau must be positive")
    return 2 ** (t / tau)


def effective_compute(c: float, a: float) -> float:
    return a * c


def metrics_from_arrays(
    c: np.ndarray, a: float, erisk: float, threshold: float
) -> Dict[str, float]:
    c = np.asarray(c, dtype=float)
    risk = (a * c) >= erisk
    trigger = c >= threshold
    risky = int(risk.sum())
    fn = int((risk & ~trigger).sum())
    fnr = fn / risky if risky > 0 else float("nan")
    coverage = 1.0 - fnr if not math.isnan(fnr) else float("nan")
    return {"fnr": float(fnr), "coverage": float(coverage)}


@dataclass(frozen=True)
class DriftModel:
    erisk: float
    tau: float
    update_interval: Optional[float] = None
    mu0: float = 0.0
    r: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.erisk <= 0:
            raise ValueError("erisk must be positive")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.update_interval is not None and self.update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

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

    def mu_c(self, t: float) -> float:
        return self.mu0 + self.r * t

    def _normalize_scenario(self, scenario: str) -> str:
        if not isinstance(scenario, str):
            raise TypeError("scenario must be a string")
        key = scenario.strip().lower()
        if key not in _SCENARIO_ALIASES:
            known = ", ".join(sorted({v for v in _SCENARIO_ALIASES.values()}))
            raise ValueError(f"Unknown scenario '{scenario}'. Expected one of: {known}")
        return _SCENARIO_ALIASES[key]

    def _sample_baseline(self, t: float, n: int, rng: np.random.Generator) -> np.ndarray:
        mu = self.mu_c(t)
        return rng.lognormal(mean=mu, sigma=self.sigma, size=n)

    def _sample_threshold_sensitive(
        self,
        t: float,
        n: int,
        rng: np.random.Generator,
        *,
        beta: float,
        sigma_s: float,
    ) -> np.ndarray:
        threshold = self.threshold(t)
        if threshold <= 0:
            raise ValueError("threshold must be positive for threshold-sensitive sampling")
        if not (0.0 < beta < 1.0):
            raise ValueError("strategic_beta must be in (0, 1)")
        if sigma_s < 0:
            raise ValueError("strategic_sigma must be non-negative")
        mu = math.log(beta * threshold)
        samples = np.empty(n, dtype=float)
        filled = 0
        while filled < n:
            needed = n - filled
            draw = rng.lognormal(mean=mu, sigma=sigma_s, size=max(needed * 2, 10))
            draw = draw[draw < threshold]
            if draw.size == 0:
                continue
            take = min(draw.size, needed)
            samples[filled : filled + take] = draw[:take]
            filled += take
        return samples

    def sample_compute(
        self,
        t: float,
        n: int,
        rng: np.random.Generator,
        *,
        scenario: str = "baseline",
        strategic_share: float = 0.0,
        strategic_beta: float = 0.9,
        strategic_sigma: float = 1.0,
    ) -> np.ndarray:
        scenario = self._normalize_scenario(scenario)
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return np.empty(0, dtype=float)
        if scenario == "baseline":
            return self._sample_baseline(t, n, rng)
        if not (0.0 <= strategic_share <= 1.0):
            raise ValueError("strategic_share must be in [0, 1]")
        n_strat = int(rng.binomial(n, strategic_share))
        n_base = n - n_strat
        samples = []
        if n_base > 0:
            samples.append(self._sample_baseline(t, n_base, rng))
        if n_strat > 0:
            samples.append(
                self._sample_threshold_sensitive(
                    t, n_strat, rng, beta=strategic_beta, sigma_s=strategic_sigma
                )
            )
        combined = np.concatenate(samples) if samples else np.empty(0, dtype=float)
        rng.shuffle(combined)
        return combined

    def simulate_metrics(
        self,
        t: float,
        n: int,
        rng: Optional[np.random.Generator] = None,
        *,
        scenario: str = "baseline",
        strategic_share: float = 0.0,
        strategic_beta: float = 0.9,
        strategic_sigma: float = 1.0,
    ) -> Dict[str, float]:
        rng = rng or np.random.default_rng()
        c = self.sample_compute(
            t,
            n,
            rng,
            scenario=scenario,
            strategic_share=strategic_share,
            strategic_beta=strategic_beta,
            strategic_sigma=strategic_sigma,
        )
        return metrics_from_arrays(c, self.a(t), self.erisk, self.threshold(t))
