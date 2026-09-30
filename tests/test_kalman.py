"""
Correctness tests for the state-space engine.

The important ones are the *oracle* tests: the filter's log-likelihood and
innovations are checked against ``statsmodels``' independently written
implementation of the same state space, and the scalar fast path is checked
against the general matrix path.  Together they pin down the half of the model
that carries no novel mathematics, so any remaining error must be in the
network alternation.

Run with::

    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tvwnn.kalman import (
    forecast_state,
    kalman_filter,
    kalman_smoother,
)


def _dgp(n=200, d=2, seed=3, sigma2=0.3):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    Omega = np.eye(d) * 0.02 + 0.005
    b = rng.normal(size=d)
    y = np.empty(n)
    for t in range(n):
        b = b + rng.multivariate_normal(np.zeros(d), Omega)
        y[t] = Z[t] @ b + rng.normal(scale=np.sqrt(sigma2))
    return y, Z, sigma2, Omega, np.zeros(d), 1e6 * np.eye(d)


def test_filter_matches_statsmodels():
    """Oracle #1: our log-likelihood equals statsmodels' to ~1e-8."""
    sm = pytest.importorskip("statsmodels.api")
    y, Z, sigma2, Omega, b0, V0 = _dgp()
    n, d = Z.shape
    ours = kalman_filter(y, Z, sigma2, Omega, b0, V0)

    class _LL(sm.tsa.statespace.MLEModel):
        def __init__(self):
            super().__init__(
                endog=y,
                k_states=d,
                initialization="known",
                initial_state=b0,
                initial_state_cov=V0 + Omega,
            )
            self["design"] = Z.T.reshape(1, d, n)
            self["transition"] = np.eye(d)
            self["selection"] = np.eye(d)
            self["obs_cov"] = np.array([[sigma2]])
            self["state_cov"] = Omega

    res = _LL().filter([])
    assert abs(ours.loglik - res.llf) < 1e-7
    np.testing.assert_allclose(ours.v, res.forecasts_error.ravel(), atol=1e-8)
    np.testing.assert_allclose(ours.F, res.forecasts_error_cov.ravel(), atol=1e-7)
    np.testing.assert_allclose(ours.a_filt, res.filtered_state.T, atol=1e-8)


def test_scalar_fast_path_matches_general_path():
    """Oracle #2: the d=1 fast path reproduces the matrix code exactly.

    The general path is forced by padding the state with a near-degenerate
    second dimension that carries no signal.
    """
    y, Z, sigma2, _, _, _ = _dgp(d=1, seed=11)
    Omega, b0, V0 = np.array([[0.02]]), np.zeros(1), 1e6 * np.eye(1)

    f1 = kalman_filter(y, Z, sigma2, Omega, b0, V0)
    s1 = kalman_smoother(y, Z, sigma2, Omega, b0, V0, filt=f1)

    n = len(y)
    Z2 = np.c_[Z, np.zeros(n)]
    Om2 = np.diag([0.02, 1e-12])
    V02 = np.diag([1e6, 1e-12])
    f2 = kalman_filter(y, Z2, sigma2, Om2, np.zeros(2), V02)
    s2 = kalman_smoother(y, Z2, sigma2, Om2, np.zeros(2), V02, filt=f2)

    assert abs(f1.loglik - f2.loglik) < 1e-9
    np.testing.assert_allclose(s1.a_smooth[:, 0], s2.a_smooth[:, 0], atol=1e-10)
    np.testing.assert_allclose(s1.xi_var[:, 0, 0], s2.xi_var[:, 0, 0], atol=1e-10)
    np.testing.assert_allclose(s1.P_lag[:, 0, 0], s2.P_lag[:, 0, 0], atol=1e-10)


def test_smoother_is_at_least_as_precise_as_filter():
    """Smoothing uses more information, so it cannot be less certain."""
    y, Z, sigma2, Omega, b0, V0 = _dgp()
    f = kalman_filter(y, Z, sigma2, Omega, b0, V0)
    s = kalman_smoother(y, Z, sigma2, Omega, b0, V0, filt=f)
    filt_var = np.einsum("tdd->td", f.P_filt)
    smooth_var = np.einsum("tdd->td", s.P_smooth)
    assert np.all(smooth_var <= filt_var + 1e-8)


def test_lag_one_covariance_is_not_negligible():
    """Guard against silently dropping P_{t,t-1|n}.

    ``Var[xi_t] = P_t + P_{t-1} - 2 P_{t,t-1}``.  Omitting the cross term
    inflates Omega_hat and raises no error, so assert it is materially
    non-zero here.
    """
    y, Z, sigma2, Omega, b0, V0 = _dgp()
    s = kalman_smoother(y, Z, sigma2, Omega, b0, V0)
    naive = np.einsum("tdd->td", s.P_smooth[1:]) + np.einsum(
        "tdd->td", s.P_smooth[:-1]
    )
    correct = np.einsum("tdd->td", s.xi_var[1:])
    assert np.mean(correct) < 0.5 * np.mean(naive)


def test_forecast_state_inflates_variance_linearly():
    """A random-walk state gains exactly k * Omega of variance over k steps."""
    y, Z, sigma2, Omega, b0, V0 = _dgp()
    f = kalman_filter(y, Z, sigma2, Omega, b0, V0)
    m0, c0 = forecast_state(f, Omega, steps=0)
    m5, c5 = forecast_state(f, Omega, steps=5)
    np.testing.assert_allclose(m0, m5)  # random walk: the mean does not move
    np.testing.assert_allclose(c5 - c0, 5 * Omega, atol=1e-10)


def test_filter_rejects_bad_input():
    y, Z, sigma2, Omega, b0, V0 = _dgp()
    with pytest.raises(ValueError):
        kalman_filter(y[:-1], Z, sigma2, Omega, b0, V0)
    with pytest.raises(ValueError):
        kalman_filter(y, Z, -1.0, Omega, b0, V0)
