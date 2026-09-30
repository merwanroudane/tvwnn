"""
Kalman filter, RTS smoother and lag-one smoothed covariance for the TVNN
state-space block.

Implements the state space of Rudd, Bondell & Silver (2026), eq. (4):

    y_{t+h} = z_t' beta_t + eps_t,      eps_t ~ N(0, sigma2)
    beta_t  = beta_{t-1} + xi_t,        xi_t  ~ N(0, Omega)
    beta_0  = b0                        (Var[beta_0] = V0)

i.e. a local-level (random-walk coefficient) model with a scalar observation
and a d-dimensional state.  Everything here is pure NumPy: the network weights
are frozen while these recursions run (Algorithm 1 of the paper), so no
automatic differentiation is required.

Recursions follow Durbin & Koopman (2012) ch. 4.3 for the filter/smoother and
Shumway & Stoffer Property 6.3 / Holmes (2013) sec. 4 for the lag-one smoothed
covariance -- the latter is needed because

    Var[xi_t | y] = P_{t|n} + P_{t-1|n} - P_{t,t-1|n} - P_{t,t-1|n}'

and a plain RTS smoother does not return P_{t,t-1|n}.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FilterResult",
    "SmootherResult",
    "kalman_filter",
    "kalman_smoother",
    "forecast_state",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


def _sym(a: np.ndarray) -> np.ndarray:
    """Force exact symmetry; guards against drift over long recursions."""
    return 0.5 * (a + a.T)


@dataclass
class FilterResult:
    """Output of :func:`kalman_filter`.

    Attributes
    ----------
    a_pred, P_pred
        One-step-ahead *predicted* state moments, ``beta_{t|t-1}`` and
        ``P_{t|t-1}``.  These are the moments that enter the prediction error
        decomposition (eq. 8): they have **not** seen observation ``t``.
    a_filt, P_filt
        *Updated* (posterior) filtered moments, ``beta_{t|t}`` and ``P_{t|t}``.
    v, F
        Innovations ``v_t = y_t - z_t' beta_{t|t-1}`` and their variances
        ``F_t = z_t' P_{t|t-1} z_t + sigma2``.
    K
        Kalman gains, ``K_t = P_{t|t-1} z_t / F_t``.
    loglik
        Exact Gaussian log-likelihood
        ``-0.5 * sum_t [log(2*pi) + log F_t + v_t^2 / F_t]``.
    """

    a_pred: np.ndarray  # (n, d)
    P_pred: np.ndarray  # (n, d, d)
    a_filt: np.ndarray  # (n, d)
    P_filt: np.ndarray  # (n, d, d)
    v: np.ndarray  # (n,)
    F: np.ndarray  # (n,)
    K: np.ndarray  # (n, d)
    loglik: float


@dataclass
class SmootherResult:
    """Output of :func:`kalman_smoother`.

    ``a_smooth[t]``/``P_smooth[t]`` are the paper's ``beta_hat_t`` and
    ``Sigma_hat_{beta_t}`` in eq. (7); ``xi_mean``/``xi_var`` are
    ``xi_hat_t`` and ``Sigma_hat_{xi_t}``.

    ``a0_smooth``/``P0_smooth`` are the smoothed moments of the *initial*
    state ``beta_0``, which the M step uses to update ``b0``.
    """

    a_smooth: np.ndarray  # (n, d)
    P_smooth: np.ndarray  # (n, d, d)
    P_lag: np.ndarray  # (n, d, d)  Cov(beta_t, beta_{t-1} | y)
    xi_mean: np.ndarray  # (n, d)
    xi_var: np.ndarray  # (n, d, d)
    a0_smooth: np.ndarray  # (d,)
    P0_smooth: np.ndarray  # (d, d)
    loglik: float


def kalman_filter(
    y: np.ndarray,
    Z: np.ndarray,
    sigma2: float,
    Omega: np.ndarray,
    b0: np.ndarray,
    V0: np.ndarray,
) -> FilterResult:
    """Run the Kalman filter for the random-walk coefficient model.

    Parameters
    ----------
    y : (n,) ndarray
        Observations.  In the paper's notation ``y[t]`` is ``y_{t+h}``: the
        caller is responsible for the ``h``-step alignment, see
        :func:`tvwnn.data.build_supervised`.
    Z : (n, d) ndarray
        Regressors, ``Z[t] = z_t`` -- the learned basis from the network.
    sigma2 : float
        Observation variance.
    Omega : (d, d) ndarray
        State innovation covariance.
    b0 : (d,) ndarray
        Mean of the initial state ``beta_0``.
    V0 : (d, d) ndarray
        Covariance of the initial state.  ``V0 = 0`` reproduces the paper's
        literal ``beta_0 = b0``; a large multiple of the identity gives the
        usual diffuse-ish prior.

    Returns
    -------
    FilterResult
    """
    y = np.asarray(y, dtype=float).ravel()
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n, d = Z.shape
    if y.shape[0] != n:
        raise ValueError(f"y has length {y.shape[0]} but Z has {n} rows.")
    Omega = _sym(np.atleast_2d(np.asarray(Omega, dtype=float)))
    V0 = _sym(np.atleast_2d(np.asarray(V0, dtype=float)))
    b0 = np.asarray(b0, dtype=float).ravel()
    sigma2 = float(sigma2)
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be strictly positive.")

    if d == 1:
        return _filter_scalar(y, Z, sigma2, Omega, b0, V0)

    a_pred = np.empty((n, d))
    P_pred = np.empty((n, d, d))
    a_filt = np.empty((n, d))
    P_filt = np.empty((n, d, d))
    v = np.empty(n)
    F = np.empty(n)
    K = np.empty((n, d))

    # beta_0 ~ N(b0, V0) and beta_1 = beta_0 + xi_1, so the prediction for the
    # first observation is N(b0, V0 + Omega).
    a, P = b0.copy(), _sym(V0 + Omega)

    loglik = 0.0
    for t in range(n):
        a_pred[t] = a
        P_pred[t] = P

        z = Z[t]
        Pz = P @ z
        f = float(z @ Pz) + sigma2  # > 0 because sigma2 > 0
        k = Pz / f
        resid = y[t] - float(z @ a)

        v[t] = resid
        F[t] = f
        K[t] = k
        loglik += -0.5 * (_LOG_2PI + np.log(f) + resid * resid / f)

        a = a + k * resid
        P = _sym(P - np.outer(k, Pz))
        a_filt[t] = a
        P_filt[t] = P

        # Time update for the next iteration.
        P = _sym(P + Omega)

    return FilterResult(
        a_pred=a_pred,
        P_pred=P_pred,
        a_filt=a_filt,
        P_filt=P_filt,
        v=v,
        F=F,
        K=K,
        loglik=float(loglik),
    )


def _filter_scalar(
    y: np.ndarray,
    Z: np.ndarray,
    sigma2: float,
    Omega: np.ndarray,
    b0: np.ndarray,
    V0: np.ndarray,
) -> FilterResult:
    """Scalar fast path for ``d == 1``.

    Algebraically identical to the general branch, but every quantity is a
    Python float, which removes the NumPy per-call overhead that dominates a
    sequential recursion over tiny matrices.  Appendix A.3 of the paper selects
    ``d in {1, 2}``, so this path covers its most common setting; it is roughly
    an order of magnitude faster.
    """
    n = Z.shape[0]
    z = Z[:, 0]
    om = float(Omega[0, 0])
    a = float(b0[0])
    P = float(V0[0, 0]) + om

    a_pred = np.empty(n)
    P_pred = np.empty(n)
    a_filt = np.empty(n)
    P_filt = np.empty(n)
    v = np.empty(n)
    F = np.empty(n)
    K = np.empty(n)
    loglik = 0.0

    for t in range(n):
        a_pred[t] = a
        P_pred[t] = P
        zt = z[t]
        f = zt * zt * P + sigma2
        k = P * zt / f
        r = y[t] - zt * a

        v[t] = r
        F[t] = f
        K[t] = k
        loglik += -0.5 * (_LOG_2PI + np.log(f) + r * r / f)

        a = a + k * r
        P = P - k * zt * P
        a_filt[t] = a
        P_filt[t] = P
        P = P + om

    return FilterResult(
        a_pred=a_pred.reshape(n, 1),
        P_pred=P_pred.reshape(n, 1, 1),
        a_filt=a_filt.reshape(n, 1),
        P_filt=P_filt.reshape(n, 1, 1),
        v=v,
        F=F,
        K=K.reshape(n, 1),
        loglik=float(loglik),
    )


def kalman_smoother(
    y: np.ndarray,
    Z: np.ndarray,
    sigma2: float,
    Omega: np.ndarray,
    b0: np.ndarray,
    V0: np.ndarray,
    filt: FilterResult | None = None,
) -> SmootherResult:
    """Run the RTS smoother and the lag-one covariance recursion.

    Returns every quantity required by the E step, eq. (7) of the paper.

    Notes
    -----
    The lag-one smoothed covariance uses Shumway & Stoffer Property 6.3 with
    transition matrix ``Phi = I``:

        P_{n,n-1|n} = (I - K_n z_n') P_{n-1|n-1}
        P_{t,t-1|n} = P_{t|t} J_{t-1}' + J_t (P_{t+1,t|n} - P_{t|t}) J_{t-1}'

    Omitting this and using ``Var[xi_t] = P_{t|n} + P_{t-1|n}`` biases
    ``Omega_hat`` upward without raising any error -- the single easiest
    silent mistake to make in this model.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n, d = Z.shape
    Omega = _sym(np.atleast_2d(np.asarray(Omega, dtype=float)))
    V0 = _sym(np.atleast_2d(np.asarray(V0, dtype=float)))
    b0 = np.asarray(b0, dtype=float).ravel()

    if filt is None:
        filt = kalman_filter(y, Z, sigma2, Omega, b0, V0)

    if d == 1:
        return _smoother_scalar(filt, Z, Omega, b0, V0)

    a_pred, P_pred = filt.a_pred, filt.P_pred
    a_filt, P_filt = filt.a_filt, filt.P_filt

    # --- smoother gains J_t, t = -1 .. n-2, where index -1 refers to beta_0.
    # J[t] = P_{t|t} inv(P_{t+1|t}); J_init = V0 inv(P_{0|-1}).
    J = np.empty((n, d, d))  # J[t] used to smooth from t+1 back to t
    for t in range(n - 1):
        J[t] = np.linalg.solve(P_pred[t + 1].T, P_filt[t].T).T
    J_init = np.linalg.solve(P_pred[0].T, V0.T).T

    a_smooth = np.empty((n, d))
    P_smooth = np.empty((n, d, d))
    a_smooth[n - 1] = a_filt[n - 1]
    P_smooth[n - 1] = P_filt[n - 1]
    for t in range(n - 2, -1, -1):
        a_smooth[t] = a_filt[t] + J[t] @ (a_smooth[t + 1] - a_pred[t + 1])
        P_smooth[t] = _sym(
            P_filt[t] + J[t] @ (P_smooth[t + 1] - P_pred[t + 1]) @ J[t].T
        )

    # Smoothed moments of the initial state beta_0 (used by the M step for b0).
    a0_smooth = b0 + J_init @ (a_smooth[0] - a_pred[0])
    P0_smooth = _sym(V0 + J_init @ (P_smooth[0] - P_pred[0]) @ J_init.T)

    # --- lag-one smoothed covariance, Cov(beta_t, beta_{t-1} | y).
    eye = np.eye(d)
    P_lag = np.empty((n, d, d))
    if n == 1:
        P_lag[0] = (eye - np.outer(filt.K[0], Z[0])) @ V0
    else:
        P_lag[n - 1] = (eye - np.outer(filt.K[n - 1], Z[n - 1])) @ P_filt[n - 2]
        for t in range(n - 2, -1, -1):
            J_prev = J_init if t == 0 else J[t - 1]
            P_lag[t] = P_filt[t] @ J_prev.T + J[t] @ (P_lag[t + 1] - P_filt[t]) @ J_prev.T

    # --- innovations xi_t = beta_t - beta_{t-1}.
    a_prev = np.vstack([a0_smooth[None, :], a_smooth[:-1]])
    P_prev = np.concatenate([P0_smooth[None, :, :], P_smooth[:-1]], axis=0)
    xi_mean = a_smooth - a_prev
    xi_var = np.empty((n, d, d))
    for t in range(n):
        xi_var[t] = _sym(P_smooth[t] + P_prev[t] - P_lag[t] - P_lag[t].T)

    return SmootherResult(
        a_smooth=a_smooth,
        P_smooth=P_smooth,
        P_lag=P_lag,
        xi_mean=xi_mean,
        xi_var=xi_var,
        a0_smooth=a0_smooth,
        P0_smooth=P0_smooth,
        loglik=filt.loglik,
    )


def _smoother_scalar(
    filt: FilterResult,
    Z: np.ndarray,
    Omega: np.ndarray,
    b0: np.ndarray,
    V0: np.ndarray,
) -> SmootherResult:
    """Scalar fast path for the RTS smoother and lag-one covariance, ``d == 1``."""
    n = Z.shape[0]
    z = Z[:, 0]
    a_pred = filt.a_pred[:, 0]
    P_pred = filt.P_pred[:, 0, 0]
    a_filt = filt.a_filt[:, 0]
    P_filt = filt.P_filt[:, 0, 0]
    K = filt.K[:, 0]
    v0 = float(V0[0, 0])

    J = np.empty(n)
    J[: n - 1] = P_filt[: n - 1] / P_pred[1:]
    J[n - 1] = 0.0
    J_init = v0 / P_pred[0]

    a_s = np.empty(n)
    P_s = np.empty(n)
    a_s[n - 1] = a_filt[n - 1]
    P_s[n - 1] = P_filt[n - 1]
    for t in range(n - 2, -1, -1):
        a_s[t] = a_filt[t] + J[t] * (a_s[t + 1] - a_pred[t + 1])
        P_s[t] = P_filt[t] + J[t] * J[t] * (P_s[t + 1] - P_pred[t + 1])

    a0 = float(b0[0]) + J_init * (a_s[0] - a_pred[0])
    P0 = v0 + J_init * J_init * (P_s[0] - P_pred[0])

    P_lag = np.empty(n)
    if n == 1:
        P_lag[0] = (1.0 - K[0] * z[0]) * v0
    else:
        P_lag[n - 1] = (1.0 - K[n - 1] * z[n - 1]) * P_filt[n - 2]
        for t in range(n - 2, -1, -1):
            j_prev = J_init if t == 0 else J[t - 1]
            P_lag[t] = P_filt[t] * j_prev + J[t] * (P_lag[t + 1] - P_filt[t]) * j_prev

    a_prev = np.r_[a0, a_s[:-1]]
    P_prev = np.r_[P0, P_s[:-1]]
    xi_mean = a_s - a_prev
    xi_var = P_s + P_prev - 2.0 * P_lag

    return SmootherResult(
        a_smooth=a_s.reshape(n, 1),
        P_smooth=P_s.reshape(n, 1, 1),
        P_lag=P_lag.reshape(n, 1, 1),
        xi_mean=xi_mean.reshape(n, 1),
        xi_var=xi_var.reshape(n, 1, 1),
        a0_smooth=np.array([a0]),
        P0_smooth=np.array([[P0]]),
        loglik=filt.loglik,
    )


def forecast_state(
    filt: FilterResult,
    Omega: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the terminal filtered state forward ``steps`` periods.

    For a random walk the ``k``-step-ahead state forecast is the last filtered
    mean, with variance inflated by ``k * Omega``:

        E[beta_{T+k} | data] = beta_{T|T}
        Var[beta_{T+k} | data] = P_{T|T} + k * Omega

    This is what makes the out-of-sample forecast honest.  The measurement
    equation attaches ``y_{t+h}`` to state ``beta_t``, so at forecast origin
    ``o`` the last usable observation pair is ``(z_{o-h}, y_o)``; the state at
    index ``o`` must then be propagated ``h`` periods.  Skipping this step is
    the look-ahead trap discussed in the README.

    Parameters
    ----------
    filt : FilterResult
        Filter output over the training pairs.
    Omega : (d, d) ndarray
    steps : int
        Number of periods to propagate.  ``0`` returns the terminal filtered
        moments unchanged.

    Returns
    -------
    (mean, cov) : ((d,) ndarray, (d, d) ndarray)
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    Omega = _sym(np.atleast_2d(np.asarray(Omega, dtype=float)))
    mean = filt.a_filt[-1].copy()
    cov = _sym(filt.P_filt[-1] + steps * Omega)
    return mean, cov
