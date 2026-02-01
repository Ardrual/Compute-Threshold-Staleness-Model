"""Backtesting analysis for the CTSM efficiency model.

Compares the model's predictions about algorithmic efficiency gains
against historical data from real AI model releases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math

from ctsm.historical_data import (
    CapabilityMatch,
    HistoricalModel,
    BASELINE_DATE,
    get_all_matches,
    get_all_models,
    months_from_baseline,
    find_best_tau,
)


@dataclass
class MatchAnalysis:
    """Analysis of a single capability-matching pair."""
    
    match: CapabilityMatch
    tau: float
    actual_gap_months: float
    predicted_gap_months: float
    error_months: float
    
    @property
    def error_percent(self) -> float:
        """Relative error as percentage of actual gap."""
        if self.actual_gap_months == 0:
            return float('nan')
        return 100 * self.error_months / self.actual_gap_months


@dataclass
class BacktestResult:
    """Results from backtesting the efficiency model."""
    
    tau: float
    analyses: List[MatchAnalysis]
    mean_error: float
    rmse: float
    best_fit_tau: float
    
    @property
    def mean_abs_error(self) -> float:
        """Mean absolute error in months."""
        if not self.analyses:
            return float('nan')
        return sum(abs(a.error_months) for a in self.analyses) / len(self.analyses)


def analyze_match(match: CapabilityMatch, tau: float) -> MatchAnalysis:
    """Analyze a single capability-matching pair."""
    actual = match.time_gap_months
    predicted = match.predicted_time_gap(tau)
    error = actual - predicted  # Positive = model under-predicts time needed
    
    return MatchAnalysis(
        match=match,
        tau=tau,
        actual_gap_months=actual,
        predicted_gap_months=predicted,
        error_months=error,
    )


def run_backtest(
    tau: float = 8.0,
    matches: Optional[List[CapabilityMatch]] = None,
) -> BacktestResult:
    """
    Run backtesting analysis for a given tau value.
    
    Args:
        tau: Efficiency doubling time in months
        matches: List of capability matches to analyze (default: all)
    
    Returns:
        BacktestResult with detailed analysis
    """
    if matches is None:
        matches = get_all_matches()
    
    analyses = [analyze_match(m, tau) for m in matches]
    
    if not analyses:
        return BacktestResult(
            tau=tau,
            analyses=[],
            mean_error=float('nan'),
            rmse=float('nan'),
            best_fit_tau=tau,
        )
    
    errors = [a.error_months for a in analyses]
    mean_error = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
    
    best_tau, _ = find_best_tau(matches)
    
    return BacktestResult(
        tau=tau,
        analyses=analyses,
        mean_error=mean_error,
        rmse=rmse,
        best_fit_tau=best_tau,
    )


def generate_report(result: BacktestResult) -> str:
    """Generate a human-readable backtest report."""
    lines = [
        "=" * 60,
        "CTSM Efficiency Model Backtest Report",
        "=" * 60,
        "",
        f"Testing tau = {result.tau:.1f} months",
        f"Best-fit tau = {result.best_fit_tau:.1f} months",
        "",
        f"Mean error: {result.mean_error:+.1f} months",
        f"Mean absolute error: {result.mean_abs_error:.1f} months", 
        f"RMSE: {result.rmse:.1f} months",
        "",
        "-" * 60,
        "Individual Match Analysis:",
        "-" * 60,
    ]
    
    for a in result.analyses:
        lines.extend([
            "",
            f"{a.match.reference.name} → {a.match.matching.name}",
            f"  Benchmark: {a.match.benchmark}",
            f"  Compute ratio: {a.match.compute_ratio:.3f}x",
            f"  Actual gap: {a.actual_gap_months:.1f} months",
            f"  Predicted gap: {a.predicted_gap_months:.1f} months",
            f"  Error: {a.error_months:+.1f} months ({a.error_percent:+.0f}%)",
        ])
    
    lines.extend([
        "",
        "=" * 60,
        "Interpretation:",
        "  Positive error = model under-predicts time (efficiency gains slower than tau)",
        "  Negative error = model over-predicts time (efficiency gains faster than tau)",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def tau_sensitivity_analysis(
    tau_range: tuple = (4.0, 24.0),
    step: float = 1.0,
    matches: Optional[List[CapabilityMatch]] = None,
) -> List[tuple]:
    """
    Analyze how RMSE varies with different tau values.
    
    Returns list of (tau, rmse) tuples.
    """
    if matches is None:
        matches = get_all_matches()
    
    results = []
    tau = tau_range[0]
    while tau <= tau_range[1]:
        result = run_backtest(tau, matches)
        results.append((tau, result.rmse))
        tau += step
    
    return results
