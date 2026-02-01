import math

import numpy as np
import pytest

from ctsm import DriftModel, efficiency_multiplier, effective_compute, metrics_from_arrays


def test_efficiency_multiplier_tau():
    t = 3.0
    tau = 6.0
    expected = 2 ** (t / tau)
    assert math.isclose(efficiency_multiplier(t, tau=tau), expected, rel_tol=1e-12)


def test_effective_compute():
    assert effective_compute(10.0, 2.5) == 25.0


def test_ideal_threshold():
    model = DriftModel(erisk=100.0, tau=10.0)
    t = 10.0
    # A(t) = 2**(t/tau) = 2
    assert math.isclose(model.ideal_threshold(t), 50.0, rel_tol=1e-12)


def test_step_hold_threshold():
    model = DriftModel(erisk=100.0, tau=10.0, update_interval=6.0)
    t = 7.0
    tn = 6.0
    expected = 100.0 / (2 ** (tn / 10.0))
    assert math.isclose(model.threshold(t), expected, rel_tol=1e-12)


def test_staleness_step_hold():
    tau = 10.0
    model = DriftModel(erisk=100.0, tau=tau, update_interval=6.0)
    t = 8.0
    tn = 6.0
    expected = 2 ** ((t - tn) / tau)
    assert math.isclose(model.staleness(t), expected, rel_tol=1e-12)


def test_mu_c():
    model = DriftModel(erisk=1.0, tau=10.0, mu0=2.0, r=0.5, sigma=0.7)
    assert math.isclose(model.mu_c(4.0), 4.0, rel_tol=1e-12)


def test_metrics_from_arrays():
    c = np.array([1.0, 2.0, 3.0, 4.0])
    a = 1.0
    erisk = 3.0
    threshold = 2.5
    metrics = metrics_from_arrays(c, a, erisk, threshold)
    assert metrics["fnr"] == 0.0
    assert metrics["coverage"] == 1.0


def test_threshold_sensitive_samples_below_threshold():
    model = DriftModel(erisk=1e6, tau=12.0, update_interval=6.0, mu0=0.0, r=0.0, sigma=1.0)
    t = 5.0
    threshold = model.threshold(t)
    rng = np.random.default_rng(123)
    samples = model.sample_compute(
        t,
        2000,
        rng,
        scenario="threshold-sensitive",
        strategic_share=1.0,
        strategic_beta=0.9,
        strategic_sigma=0.5,
    )
    assert samples.size == 2000
    assert np.all(samples < threshold)


def test_unknown_scenario_raises():
    model = DriftModel(erisk=1.0, tau=10.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        model.sample_compute(0.0, 10, rng, scenario="not-a-scenario")
