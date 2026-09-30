"""
Tests for the EM algorithm over the variance components.

These are the known-truth tests: with a *fixed* design matrix (no network in
the loop) the M step is an exact maximiser, so ``sigma2`` and ``Omega`` must
converge on the values that generated the data, and the marginal
log-likelihood must increase monotonically -- the defining property of EM.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvwnn.em import StateParams, expected_loglik, m_step, run_em
from tvwnn.kalman import kalman_filter, kalman_smoother


def _local_level(n=800, d=2, sigma2=0.25, seed=7):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    Omega = np.diag([0.010, 0.004][:d])
    beta = rng.normal(size=d)
    y = np.empty(n)
    for t in range(n):
        beta = beta + rng.multivariate_normal(np.zeros(d), Omega)
        y[t] = Z[t] @ beta + rng.normal(scale=np.sqrt(sigma2))
    return y, Z, sigma2, Omega


def test_em_recovers_the_truth():
    """The headline known-truth check."""
    y, Z, sigma2, Omega = _local_level()
    d = Z.shape[1]
    start = StateParams(
        sigma2=1.0, Omega=np.eye(d) * 0.1, b0=np.zeros(d), V0=1e6 * np.eye(d)
    )
    theta, info = run_em(y, Z, start, max_iter=500, tol=1e-11)

    assert abs(theta.sigma2 - sigma2) < 0.05 * sigma2 + 0.02
    np.testing.assert_allclose(
        np.diag(theta.Omega), np.diag(Omega), atol=0.006
    )
    assert info["n_iter"] >= 1


def test_loglikelihood_increases_monotonically():
    """EM's defining guarantee -- and a sharp test of the lag-one covariance.

    Dropping ``P_{t,t-1|n}`` from ``Var[xi_t]`` breaks monotonicity here.
    """
    y, Z, _, _ = _local_level(n=400)
    d = Z.shape[1]
    start = StateParams(
        sigma2=1.0, Omega=np.eye(d) * 0.05, b0=np.zeros(d), V0=1e6 * np.eye(d)
    )
    _, info = run_em(y, Z, start, max_iter=120, tol=1e-12, track=True)
    ll = np.asarray(info["loglik_path"])
    assert np.all(np.diff(ll) > -1e-7), "EM decreased the log-likelihood"


def test_m_step_increases_q():
    """One M step cannot lower Q at the point it was expanded around."""
    y, Z, _, _ = _local_level(n=300)
    d = Z.shape[1]
    theta = StateParams(
        sigma2=0.8, Omega=np.eye(d) * 0.05, b0=np.zeros(d), V0=1e6 * np.eye(d)
    )
    sm = kalman_smoother(y, Z, theta.sigma2, theta.Omega, theta.b0, theta.V0)
    q_before = expected_loglik(y, Z, sm, theta)
    q_after = expected_loglik(y, Z, sm, m_step(y, Z, sm, theta))
    assert q_after >= q_before - 1e-8


def test_diagonal_structure_is_respected():
    y, Z, _, _ = _local_level(n=300)
    d = Z.shape[1]
    start = StateParams(
        sigma2=1.0, Omega=np.eye(d) * 0.05, b0=np.zeros(d), V0=1e6 * np.eye(d)
    )
    theta, _ = run_em(y, Z, start, max_iter=30, omega_structure="diagonal")
    off = theta.Omega - np.diag(np.diag(theta.Omega))
    assert np.allclose(off, 0.0)


def test_variance_floor_holds():
    """A degenerate series must not drive sigma2 to zero and break the filter."""
    n, d = 120, 1
    Z = np.ones((n, d))
    y = np.zeros(n)  # perfectly explained by a constant state
    start = StateParams(
        sigma2=1.0, Omega=np.eye(d) * 0.1, b0=np.zeros(d), V0=np.eye(d)
    )
    theta, _ = run_em(y, Z, start, max_iter=50)
    assert theta.sigma2 > 0
    assert np.all(np.diag(theta.Omega) > 0)
    filt = kalman_filter(y, Z, theta.sigma2, theta.Omega, theta.b0, theta.V0)
    assert np.isfinite(filt.loglik)


def test_initialise_follows_appendix_a3():
    """Lognormal(-1, 1) starting values, as Appendix A.3 specifies."""
    rng = np.random.default_rng(0)
    th = StateParams.initialise(2, rng, v0_scale=1e6)
    assert th.sigma2 > 0
    assert th.Omega.shape == (2, 2)
    assert np.allclose(th.V0, 1e6 * np.eye(2))
    assert np.allclose(th.b0, 0.0)


def test_signal_to_noise_cap_is_enforced():
    """The cap bounds tr(Omega) / (d * sigma2) as advertised.

    It is off in `run_em` (which keeps EM's textbook guarantees) and on by
    default in `TVNN`, where it stops the state absorbing the serial
    correlation induced by the paper's overlapping h-step target.
    """
    y, Z, _, _ = _local_level(n=400)
    d = Z.shape[1]
    start = StateParams(
        sigma2=1.0, Omega=np.eye(d) * 0.05, b0=np.zeros(d), V0=1e6 * np.eye(d)
    )
    free, _ = run_em(y, Z, start, max_iter=80, sn_cap=None)
    capped, _ = run_em(y, Z, start, max_iter=80, sn_cap=1e-3)

    ratio = lambda th: np.trace(th.Omega) / (d * th.sigma2)
    assert ratio(capped) <= 1e-3 * (1 + 1e-8)
    assert ratio(free) > ratio(capped)


def test_tvnn_enables_the_cap_by_default():
    from tvwnn import TVNN
    from tvwnn.data import simulate_tvnn

    ds, _ = simulate_tvnn(n=150, seed=3)
    m = TVNN(d=1, n_outer=4, grad_steps=4, random_state=0).fit(ds.U, ds.y)
    assert m.summary()["sn_cap"] == 0.01
    ratio = np.trace(m.theta_.Omega) / (m.d * m.theta_.sigma2)
    assert ratio <= 0.01 * (1 + 1e-8)
