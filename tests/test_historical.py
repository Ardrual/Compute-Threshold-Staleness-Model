"""Tests for historical data and backtesting modules."""

import math
from datetime import date

import pytest

from ctsm.historical_data import (
    HistoricalModel,
    CapabilityMatch,
    MODELS,
    get_all_models,
    get_all_matches,
    months_from_baseline,
    find_best_tau,
    BASELINE_DATE,
)
from ctsm.backtest import (
    analyze_match,
    run_backtest,
    tau_sensitivity_analysis,
)


class TestHistoricalModel:
    def test_log_flops(self):
        model = HistoricalModel(
            name="Test",
            release_date=date(2020, 1, 1),
            training_flops=1e24,
        )
        assert math.isclose(model.log_flops, math.log(1e24), rel_tol=1e-10)


class TestCapabilityMatch:
    def test_time_gap_months(self):
        ref = HistoricalModel("A", date(2020, 1, 1), 1e24)
        mat = HistoricalModel("B", date(2021, 1, 1), 1e23)
        match = CapabilityMatch(ref, mat, "test")
        # Approximately 12 months
        assert 11.9 < match.time_gap_months < 12.1

    def test_compute_ratio(self):
        ref = HistoricalModel("A", date(2020, 1, 1), 1e24)
        mat = HistoricalModel("B", date(2021, 1, 1), 1e23)
        match = CapabilityMatch(ref, mat, "test")
        assert math.isclose(match.compute_ratio, 0.1, rel_tol=1e-10)

    def test_predicted_time_gap(self):
        # If C_match = 0.5 * C_ref and tau = 8, then:
        # Δt = -τ * log₂(0.5) = -8 * (-1) = 8 months
        ref = HistoricalModel("A", date(2020, 1, 1), 1e24)
        mat = HistoricalModel("B", date(2020, 9, 1), 5e23)  # 0.5x compute
        match = CapabilityMatch(ref, mat, "test")
        
        predicted = match.predicted_time_gap(tau=8.0)
        assert math.isclose(predicted, 8.0, rel_tol=1e-10)

    def test_prediction_error(self):
        ref = HistoricalModel("A", date(2020, 1, 1), 1e24)
        # If actual gap is 10 months but predicted is 8, error = 2
        mat = HistoricalModel("B", date(2020, 11, 1), 5e23)
        match = CapabilityMatch(ref, mat, "test")
        
        error = match.prediction_error(tau=8.0)
        # Actual gap ≈ 10 months, predicted = 8, error ≈ 2
        assert 1.5 < error < 2.5


class TestDataIntegrity:
    def test_models_have_required_fields(self):
        for key, model in MODELS.items():
            assert model.name, f"{key} missing name"
            assert model.release_date, f"{key} missing release_date"
            assert model.training_flops > 0, f"{key} has invalid FLOPs"

    def test_get_all_models_sorted(self):
        models = get_all_models()
        dates = [m.release_date for m in models]
        assert dates == sorted(dates), "Models not sorted by release date"

    def test_capability_matches_valid(self):
        matches = get_all_matches()
        for match in matches:
            assert match.reference.release_date < match.matching.release_date, \
                f"Reference should be before matching: {match.reference.name} -> {match.matching.name}"


class TestBacktest:
    def test_analyze_match(self):
        ref = HistoricalModel("A", date(2020, 1, 1), 1e24)
        mat = HistoricalModel("B", date(2020, 9, 1), 5e23)
        match = CapabilityMatch(ref, mat, "test")
        
        analysis = analyze_match(match, tau=8.0)
        assert analysis.tau == 8.0
        assert analysis.match is match
        assert math.isclose(analysis.predicted_gap_months, 8.0, rel_tol=1e-10)

    def test_run_backtest_returns_result(self):
        result = run_backtest(tau=8.0)
        assert result.tau == 8.0
        assert len(result.analyses) > 0
        assert not math.isnan(result.rmse)
        assert result.best_fit_tau > 0

    def test_tau_sensitivity_analysis(self):
        results = tau_sensitivity_analysis(tau_range=(6.0, 12.0), step=2.0)
        assert len(results) == 4  # 6, 8, 10, 12
        taus = [r[0] for r in results]
        assert taus == [6.0, 8.0, 10.0, 12.0]


class TestMonthsFromBaseline:
    def test_baseline_is_zero(self):
        assert months_from_baseline(BASELINE_DATE) == 0.0

    def test_one_year_later(self):
        one_year = date(2020, 2, 1)
        months = months_from_baseline(one_year)
        assert 11.9 < months < 12.1
