import math

import pytest

from ctsm import DriftModel, efficiency_multiplier, effective_compute


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
