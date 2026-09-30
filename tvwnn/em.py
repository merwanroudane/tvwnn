"""
EM algorithm for the variance components ``theta = {sigma2, Omega, b0}``.

This is the inner ``for k = 0 : K`` loop of Algorithm 1 in Rudd, Bondell &
Silver (2026): with the network weights ``w`` -- and therefore the learned
basis ``z_t`` -- held fixed, the E step is one Kalman filter/smoother pass and
the M step has a closed form.

The M step maximises ``Q(theta | theta^(k), w)``, eq. (6) of the paper /
eq. (9) of the Supporting Information:

    Q = -1/(2 sigma2) sum_t [ (y_{t+h} - z_t' beta_hat_t)^2
                              + z_t' Sigma_hat_{beta_t} z_t ]
        - 1/2 sum_t tr( Omega^{-1} ( Sigma_hat_{xi_t}
                                     + xi_hat_t xi_hat_t' ) )
        - T/2 ( log|Omega| + log sigma2 )

Differentiating and setting to zero gives the updates implemented below.
These are the unconstrained MARSS updates of Holmes (2013, sec. 4), which the
paper delegates to by citation without printing them.

NOTE ON THE PRINTED EQUATION.  The main text prints the last term of eq. (3)
as ``-1/2 (log|Omega| + log sigma2)`` without the ``T`` multiplier.  Appendix
A.1 of the Supporting Information repeats that slip in its first display and
then derives ``-T/2`` correctly in eq. (9).  We use ``-T/2``.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .kalman import SmootherResult, kalman_filter, kalman_smoother

__all__ = ["StateParams", "m_step", "expected_loglik", "run_em"]


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


@dataclass
class StateParams:
    """The variance components ``theta = {sigma2, Omega, b0}`` plus ``V0``.

    ``V0`` is *not* estimated.  Appendix A.3 states only that ``b0`` is "fixed
    but unknown, estimated alongside sigma2 and Omega as part of the EM
    algorithm"; the initial state covariance is never mentioned.  We therefore
    treat it as a fixed hyperparameter -- see ``v0_scale`` in
    :class:`tvwnn.model.TVNN`.
    """

    sigma2: float
    Omega: np.ndarray
    b0: np.ndarray
    V0: np.ndarray

    @classmethod
    def initialise(
        cls,
        d: int,
        rng: np.random.Generator,
        v0_scale: float = 1e6,
        lognormal: tuple[float, float] = (-1.0, 1.0),
    ) -> "StateParams":
        """Draw starting values the way Appendix A.3 does.

        "Initial values for the covariance components were drawn from a
        Lognormal(-1, 1) distribution."  We read that as one draw for
        ``sigma2`` and ``d`` independent draws for the diagonal of ``Omega``.
        """
        mu, sd = lognormal
        sigma2 = float(rng.lognormal(mu, sd))
        omega_diag = rng.lognormal(mu, sd, size=d)
        return cls(
            sigma2=sigma2,
            Omega=np.diag(omega_diag),
            b0=np.zeros(d),
            V0=v0_scale * np.eye(d),
        )

    def copy(self) -> "StateParams":
        return StateParams(
            sigma2=float(self.sigma2),
            Omega=self.Omega.copy(),
            b0=self.b0.copy(),
            V0=self.V0.copy(),
        )


def m_step(
    y: np.ndarray,
    Z: np.ndarray,
    sm: SmootherResult,
    theta: StateParams,
    omega_structure: str = "full",
    min_variance: float = 1e-10,
    sn_cap: float | None = None,
) -> StateParams:
    """Closed-form M step.

    Implements

        sigma2_hat = (1/T) sum_t [ (y_t - z_t' beta_hat_t)^2
                                   + z_t' Sigma_hat_{beta_t} z_t ]
        Omega_hat  = (1/T) sum_t [ Sigma_hat_{xi_t} + xi_hat_t xi_hat_t' ]
        b0_hat     = beta_hat_0                       (smoothed initial state)

    Parameters
    ----------
    omega_structure : {"full", "diagonal"}
        ``"full"`` is the unconstrained maximiser of eq. (6) and is the
        default.  At the paper's own ``d in {1, 2}`` the difference is at most
        one off-diagonal parameter.  ``"diagonal"`` keeps only the diagonal,
        which is the constrained MARSS update of Holmes (2013).
    min_variance : float
        Floor applied to ``sigma2`` and to the diagonal of ``Omega`` so the
        filter cannot divide by zero if the EM drives a component to the
        boundary.
    sn_cap : float or None, default None
        Upper bound on the signal-to-noise ratio ``tr(Omega) / (d * sigma2)``.
        ``None`` -- the default here -- removes the bound, which is both the
        paper's literal model and the textbook EM.  :class:`tvwnn.model.TVNN`
        switches it on.

        **Why this exists.**  Unconstrained maximum likelihood for a
        random-walk-coefficient model is free to attribute *all* persistence
        in the data to coefficient drift rather than to noise, and on real
        macro data it does.  The paper's own design makes this worse: at
        horizon ``h`` the target ``y_{t+h} - y_t`` is an overlapping
        difference, so the errors are MA(h-1) by construction, and the state
        happily absorbs that serial correlation as drift.  On FRED-MD
        unemployment at ``h = 12`` the unconstrained EM returns
        ``tr(Omega)/sigma2 ~ 11``, meaning the coefficient fully
        re-randomises within a year; the Kalman gain approaches one, the state
        tracks ``y_t / z_t``, and propagating it ``h`` steps produced
        out-of-sample forecasts spanning ``[-18.4, 9.8]`` against a realised
        range of ``[-1.5, 4.0]``.

        Capping the ratio is the standard control, and is the frequentist
        analogue of the shrinkage prior on the signal-to-noise ratio in
        Goulet Coulombe (2020), "Time-Varying Parameters as Ridge
        Regressions".  With the default cap the same configuration gives
        ``[-2.4, 3.8]`` and roughly a five-fold lower mean squared error.

        The paper imposes no such bound.  Set ``sn_cap=None`` to reproduce it
        literally, and expect the instability above at long horizons.
    """
    y = np.asarray(y, dtype=float).ravel()
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n = Z.shape[0]

    # sigma2: residual plus the state-uncertainty correction z' Sigma z.
    fitted = np.einsum("td,td->t", Z, sm.a_smooth)
    resid2 = (y - fitted) ** 2
    curvature = np.einsum("td,tde,te->t", Z, sm.P_smooth, Z)
    sigma2 = float(np.mean(resid2 + curvature))

    # Omega: second moment of the smoothed innovations.
    outer = np.einsum("td,te->tde", sm.xi_mean, sm.xi_mean)
    Omega = _sym(np.mean(sm.xi_var + outer, axis=0))

    if omega_structure == "diagonal":
        Omega = np.diag(np.diag(Omega))
    elif omega_structure != "full":
        raise ValueError("omega_structure must be 'full' or 'diagonal'.")

    sigma2 = max(sigma2, min_variance)
    diag = np.clip(np.diag(Omega), min_variance, None)
    np.fill_diagonal(Omega, diag)

    if sn_cap is not None:
        d = Omega.shape[0]
        budget = float(sn_cap) * sigma2 * d
        trace = float(np.trace(Omega))
        if trace > budget > 0:
            Omega = Omega * (budget / trace)
            np.fill_diagonal(
                Omega, np.clip(np.diag(Omega), min_variance, None)
            )

    if n == 0:  # pragma: no cover - defensive
        raise ValueError("Cannot run the M step on an empty sample.")

    return replace(theta, sigma2=sigma2, Omega=Omega, b0=sm.a0_smooth.copy())


def expected_loglik(
    y: np.ndarray,
    Z: np.ndarray,
    sm: SmootherResult,
    theta: StateParams,
) -> float:
    """Evaluate ``Q(theta | theta^(k), w)``, eq. (6).

    Used only for monitoring: a correct EM makes this non-decreasing across
    iterations at fixed ``z_t``.
    """
    y = np.asarray(y, dtype=float).ravel()
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n = Z.shape[0]

    fitted = np.einsum("td,td->t", Z, sm.a_smooth)
    curvature = np.einsum("td,tde,te->t", Z, sm.P_smooth, Z)
    term1 = -np.sum((y - fitted) ** 2 + curvature) / (2.0 * theta.sigma2)

    Omega_inv = np.linalg.inv(theta.Omega)
    outer = np.einsum("td,te->tde", sm.xi_mean, sm.xi_mean)
    term2 = -0.5 * np.sum(np.einsum("de,ted->t", Omega_inv, sm.xi_var + outer))

    sign, logdet = np.linalg.slogdet(theta.Omega)
    if sign <= 0:  # pragma: no cover - defensive
        return -np.inf
    term3 = -0.5 * n * (logdet + np.log(theta.sigma2))

    return float(term1 + term2 + term3)


def run_em(
    y: np.ndarray,
    Z: np.ndarray,
    theta: StateParams,
    max_iter: int = 50,
    tol: float = 1e-6,
    omega_structure: str = "full",
    sn_cap: float | None = None,
    track: bool = False,
) -> tuple[StateParams, dict]:
    """Run EM for ``theta`` to convergence with the basis ``Z`` held fixed.

    This is the ``for k = 0 : K`` block of Algorithm 1.  Convergence is judged
    on the *marginal* log-likelihood returned by the filter, which is the
    quantity EM is guaranteed to increase.

    Parameters
    ----------
    max_iter : int
        Cap on EM iterations.  Appendix A.3 says EM was run "to convergence";
        ``max_iter`` is the safety net.
    tol : float
        Stop when the relative increase in the log-likelihood falls below this.
    sn_cap : float or None, default None
        Signal-to-noise bound; see :func:`m_step`.
        :class:`tvwnn.model.TVNN` turns it on (0.01) by default; the bare EM
        here leaves it off so the algorithm keeps its textbook guarantees.  Note that with a cap in
        force the M step is a *constrained* maximiser, so the log-likelihood
        is no longer guaranteed to increase monotonically.

    Returns
    -------
    (theta, info)
        ``info`` carries ``n_iter``, ``loglik``, and -- when ``track`` is set --
        the full ``loglik_path`` and ``q_path``.
    """
    theta = theta.copy()
    loglik_path: list[float] = []
    q_path: list[float] = []
    prev = -np.inf
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        filt = kalman_filter(y, Z, theta.sigma2, theta.Omega, theta.b0, theta.V0)
        sm = kalman_smoother(
            y, Z, theta.sigma2, theta.Omega, theta.b0, theta.V0, filt=filt
        )
        if track:
            loglik_path.append(filt.loglik)
            q_path.append(expected_loglik(y, Z, sm, theta))

        theta = m_step(
            y, Z, sm, theta, omega_structure=omega_structure, sn_cap=sn_cap
        )

        current = filt.loglik
        if np.isfinite(prev):
            denom = max(abs(prev), 1.0)
            if abs(current - prev) / denom < tol:
                prev = current
                break
        prev = current

    info = {"n_iter": n_iter, "loglik": float(prev)}
    if track:
        info["loglik_path"] = loglik_path
        info["q_path"] = q_path
    return theta, info
