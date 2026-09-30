"""
The time-varying neural network (TVNN) estimator.

Direct implementation of Algorithm 1 of Rudd, Bondell & Silver (2026):

    Initialise {w, theta}, Adam moments
    for i = 0 : N
        z_t = f_w(u_t)                                  # basis, network frozen
        for k = 0 : K                                   # EM for theta
            Kalman filter -> smoother -> closed-form M step
        b_t, V_t  from the Kalman filter                # states frozen
        optional momentum on b
        for l = 0 : M                                   # Adam on the network
            grad of [ log p(y | theta, w) - alpha ||w||^2 ]
    return w_hat, theta_hat

The two loops alternate over *different* objectives, exactly as the paper
specifies: ``Q`` (smoothed states) for the variance components, the prediction
error decomposition (filtered states) for the network weights.  Section 3 of
the paper explains why -- only filtered states exist out of sample, so
training the basis against smoothed states generalises badly.

Two switches expose questions the paper does not settle; see the README
section "Two things the paper leaves open".

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .em import StateParams, run_em
from .kalman import forecast_state, kalman_filter, kalman_smoother
from .network import BasisNetwork

__all__ = ["TVNN", "TVNNEnsemble", "FitHistory"]

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class FitHistory:
    """Per-outer-iteration diagnostics, for the convergence plots."""

    loss: list[float] = field(default_factory=list)
    loglik: list[float] = field(default_factory=list)
    sigma2: list[float] = field(default_factory=list)
    omega_trace: list[float] = field(default_factory=list)
    em_iters: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "loss": self.loss,
            "loglik": self.loglik,
            "sigma2": self.sigma2,
            "omega_trace": self.omega_trace,
            "em_iters": self.em_iters,
        }


class TVNN:
    """Neural network with time-varying final-layer weights.

    Parameters
    ----------
    d : int, default 1
        Number of basis functions, i.e. the dimension of the time-varying
        state.  Appendix A.3 of the paper selects from ``{1, 2}``.
    hidden_sizes : sequence of int, default ``(32,)``
        ``(32,)`` or ``(32, 16)`` in the paper.
    activation : str, default ``"tanh"``
    dropout : float, default 0.0
    fixed_effect : bool, default False
        Include the ``z_tilde_t`` output node of eq. (4).
    n_outer : int, default 30
        ``N``, the number of outer iterations.
    grad_steps : int, default 10
        ``M``, Adam steps per outer iteration.  The paper runs "EM to
        convergence for every 10 epochs of gradient updates" under a fixed
        budget of 300 epochs, so the defaults ``30 x 10`` reproduce exactly
        that budget.
    em_iter : int, default 50
        Cap ``K`` on the inner EM iterations ("run to convergence").
    em_tol : float, default 1e-6
    learning_rate : float, default 0.01
        A.3 draws it from ``0.01 * 2**s``, ``s in {0, 1, 2}``.
    l2 : float, default 0.32
        The ``alpha`` of Algorithm 1.  A.3 draws it from ``0.01 * 2**s``,
        ``s in {1, ..., 6}``; the TVNN most commonly selected ``s = 5``, i.e.
        ``0.32``, for INDPRO.
    momentum : float, default 0.0
        The ``gamma`` of the optional state-momentum line in Algorithm 1,
        ``b <- gamma * b + (1 - gamma) * b_prev``.  ``0.0`` disables it.
    loss : {"weighted_sse", "exact"}, default "weighted_sse"
        Which form of eq. (8) to optimise -- see the README.  ``"weighted_sse"``
        is the expression as printed in the paper; ``"exact"`` adds back the
        ``log F_t`` term that makes it the true Gaussian marginal likelihood.
    state : {"predicted", "filtered"}, default "predicted"
        Which Kalman moments play the role of ``b_t``, ``V_t`` in eq. (8).
        ``"predicted"`` uses ``beta_{t|t-1}``, which is what makes eq. (8) an
        actual *prediction error* decomposition (Durbin & Koopman ch. 7.2, the
        reference Appendix A.2 follows).  ``"filtered"`` is the literal reading
        of Algorithm 1's ``E[beta_t | y_t]`` and lets observation ``t`` inform
        its own prediction; it is provided for comparison only.
    omega_structure : {"full", "diagonal"}, default "full"
    sn_cap : float or None, default 0.01
        Upper bound on ``tr(Omega) / (d * sigma2)``.  Without it the EM
        attributes the serial correlation induced by the paper's overlapping
        ``h``-step target to coefficient drift, and long-horizon forecasts
        become unstable.  See :func:`tvwnn.em.m_step` for the full argument
        and the numbers.  ``None`` reproduces the paper literally.
    v0_scale : float, default 1e6
        Diagonal scale of ``V0``.  ``0.0`` gives the paper's literal
        ``beta_0 = b0``.
    standardize : bool, default True
        Standardise inputs and target internally.  Not mentioned in the paper,
        but a ``tanh`` network on raw macro data will not train; the scaling is
        inverted on output so it is invisible to the caller.
    normalize_basis : bool, default True
        Rescale each column of ``z_t`` to unit standard deviation over the
        training sample before it reaches the state-space layer.

        This is a *reparameterisation, not a restriction*: the model is
        invariant to ``z -> A'z``, ``beta -> A^{-1}beta`` (see "Identification"
        in the README), and this simply picks a well-conditioned point on that
        orbit.  It matters a great deal in practice.  The basis layer is linear
        and therefore unbounded, so on real macro data ``|z|`` drifts into the
        tens while ``beta`` shrinks to compensate; the Kalman gain
        ``K_t = P z_t / (z_t' P z_t + sigma2)`` then becomes enormous, the EM
        drives ``sigma2`` toward zero, and the filter starts chasing every
        surprise.  On FRED-MD unemployment that produced forecasts up to
        ``14.3`` against a realised range of ``[-1.5, 4.0]``.  Normalising the
        basis removes the pathology without changing the model class.

        The paper does not mention any scaling of ``z_t``.  Set ``False`` for
        the literal reading.
    random_state : int or None
    device : str or None
    verbose : bool, default False

    Attributes
    ----------
    theta_ : StateParams
        Fitted variance components.
    history_ : FitHistory
    smoothed_states_ : ndarray of shape (n, d)
        ``beta_hat_t``, the smoothed time-varying weights on the training
        sample.  Feed to :func:`tvwnn.plots.plot_time_varying_weights`.
    """

    def __init__(
        self,
        d: int = 1,
        hidden_sizes: Sequence[int] = (32,),
        activation: str = "tanh",
        dropout: float = 0.0,
        fixed_effect: bool = False,
        n_outer: int = 30,
        grad_steps: int = 10,
        em_iter: int = 50,
        em_tol: float = 1e-6,
        learning_rate: float = 0.01,
        l2: float = 0.32,
        momentum: float = 0.0,
        loss: str = "weighted_sse",
        state: str = "predicted",
        omega_structure: str = "full",
        sn_cap: float | None = 0.01,
        v0_scale: float = 1e6,
        standardize: bool = True,
        normalize_basis: bool = True,
        random_state: int | None = None,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        if loss not in {"weighted_sse", "exact"}:
            raise ValueError("loss must be 'weighted_sse' or 'exact'.")
        if state not in {"predicted", "filtered"}:
            raise ValueError("state must be 'predicted' or 'filtered'.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1).")

        self.d = int(d)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation = activation
        self.dropout = float(dropout)
        self.fixed_effect = bool(fixed_effect)
        self.n_outer = int(n_outer)
        self.grad_steps = int(grad_steps)
        self.em_iter = int(em_iter)
        self.em_tol = float(em_tol)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.momentum = float(momentum)
        self.loss = loss
        self.state = state
        self.omega_structure = omega_structure
        self.sn_cap = None if sn_cap is None else float(sn_cap)
        self.v0_scale = float(v0_scale)
        self.standardize = bool(standardize)
        self.normalize_basis = bool(normalize_basis)
        self.random_state = random_state
        self.device = torch.device(device) if device else torch.device("cpu")
        self.verbose = bool(verbose)

        self.net_: BasisNetwork | None = None
        self.theta_: StateParams | None = None
        self.history_: FitHistory = FitHistory()
        self.smoothed_states_: np.ndarray | None = None
        self.smoothed_state_cov_: np.ndarray | None = None
        self._U_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._x_mean = self._x_std = self._y_mean = self._y_std = None
        self._z_scale: np.ndarray | None = None

    # ------------------------------------------------------------------ utils

    def _scale_fit(self, U: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.standardize:
            self._x_mean = np.zeros(U.shape[1])
            self._x_std = np.ones(U.shape[1])
            self._y_mean, self._y_std = 0.0, 1.0
            return U, y
        self._x_mean = U.mean(axis=0)
        std = U.std(axis=0)
        std[std < 1e-12] = 1.0
        self._x_std = std
        self._y_mean = float(y.mean())
        ystd = float(y.std())
        self._y_std = ystd if ystd > 1e-12 else 1.0
        return (U - self._x_mean) / self._x_std, (y - self._y_mean) / self._y_std

    def _scale_x(self, U: np.ndarray) -> np.ndarray:
        return (U - self._x_mean) / self._x_std

    def _unscale_y(self, y: np.ndarray | float) -> np.ndarray | float:
        return y * self._y_std + self._y_mean

    def _tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arr, dtype=np.float64), device=self.device)

    def _set_basis_scale(self, Z: np.ndarray) -> None:
        """Fix the basis scale from the training sample (see ``normalize_basis``)."""
        if not self.normalize_basis:
            self._z_scale = np.ones(Z.shape[1])
            return
        scale = Z.std(axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        self._z_scale = scale

    def _rescale(self, Z: np.ndarray) -> np.ndarray:
        return Z if self._z_scale is None else Z / self._z_scale

    # -------------------------------------------------------------------- fit

    def fit(self, U: np.ndarray, y: np.ndarray) -> "TVNN":
        """Fit the model.

        Parameters
        ----------
        U : (n, P) array
            Design matrix whose row ``t`` is ``u_t`` (eq. 1): the stacked lags
            of the predictors and of the target available at time ``t``.
        y : (n,) array
            Target aligned to the *forecast*, i.e. ``y[t]`` is ``y_{t+h}``.
            Use :func:`tvwnn.data.build_supervised` to build the pair.

        Returns
        -------
        self
        """
        U = np.atleast_2d(np.asarray(U, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if U.shape[0] != y.shape[0]:
            raise ValueError(
                f"U has {U.shape[0]} rows but y has {y.shape[0]} entries; "
                "they must be aligned by build_supervised()."
            )
        if U.shape[0] < 10:
            raise ValueError("Need at least 10 observations to fit a TVNN.")

        Us, ys = self._scale_fit(U, y)
        self._U_train, self._y_train = Us, ys

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        rng = np.random.default_rng(self.random_state)

        self.net_ = BasisNetwork(
            n_inputs=Us.shape[1],
            d=self.d,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            dropout=self.dropout,
            fixed_effect=self.fixed_effect,
        ).to(self.device).double()

        theta = StateParams.initialise(self.d, rng, v0_scale=self.v0_scale)
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)

        U_t = self._tensor(Us)
        y_t = self._tensor(ys)
        self.history_ = FitHistory()
        b_prev: np.ndarray | None = None

        for _ in range(self.n_outer):
            # --- z_t = f_w(u_t), network frozen from here to the Adam block.
            Z, z_tilde = self.net_.basis_numpy(U_t)
            self._set_basis_scale(Z)
            Z = self._rescale(Z)
            target = ys - z_tilde if z_tilde is not None else ys

            # --- inner EM for theta (E step = filter + smoother, M step closed form)
            theta, em_info = run_em(
                target,
                Z,
                theta,
                max_iter=self.em_iter,
                tol=self.em_tol,
                omega_structure=self.omega_structure,
                sn_cap=self.sn_cap,
            )

            # --- b_t, V_t from the filter; frozen for the whole Adam block.
            filt = kalman_filter(
                target, Z, theta.sigma2, theta.Omega, theta.b0, theta.V0
            )
            if self.state == "predicted":
                b, V = filt.a_pred, filt.P_pred
            else:
                b, V = filt.a_filt, filt.P_filt

            if self.momentum > 0.0 and b_prev is not None and b_prev.shape == b.shape:
                b = self.momentum * b + (1.0 - self.momentum) * b_prev
            b_prev = b.copy()

            b_t = self._tensor(b)
            V_t = self._tensor(V)
            z_scale_t = self._tensor(self._z_scale)
            sigma2_t = float(theta.sigma2)

            # --- M Adam steps on the network, states frozen.
            self.net_.train()
            last_loss = float("nan")
            for _ in range(self.grad_steps):
                optimiser.zero_grad(set_to_none=True)
                z, zt = self.net_(U_t)
                z = z / z_scale_t
                resid = y_t - torch.einsum("td,td->t", z, b_t)
                if zt is not None:
                    resid = resid - zt
                F = sigma2_t + torch.einsum("td,tde,te->t", z, V_t, z)
                F = torch.clamp(F, min=1e-12)

                # Averaged over t, not summed.  A positive rescaling of the
                # likelihood leaves its maximiser unchanged but fixes what the
                # penalty `alpha` means -- and only the averaged reading makes
                # Appendix A.3's grid (0.02 to 0.64) the "heavy regularization"
                # it describes.  The baselines use the same convention, so `l2`
                # is comparable across models.
                # A caveat worth knowing: because of the 1/F weighting this
                # term sits on a different scale from the baselines' plain mean
                # squared error, roughly by a factor 1/sigma2, so the same `l2`
                # does not bite equally hard across models.  Rescaling by
                # sigma2 to equalise them was tried and made forecasts worse
                # (INDPRO h=6: 1.59 -> 1.83 relative MSE), so the direct
                # averaged reading of eq. (8) is kept.  The paper does not
                # specify the convention; see "A fourth open question" in the
                # README.
                if self.loss == "weighted_sse":
                    objective = torch.mean(resid**2 / F)
                else:  # "exact": the full Gaussian prediction error decomposition
                    objective = 0.5 * torch.mean(
                        _LOG_2PI + torch.log(F) + resid**2 / F
                    )

                total = objective + self.l2 * self.net_.l2_penalty()
                total.backward()
                optimiser.step()
                last_loss = float(total.detach())
            self.net_.eval()

            self.history_.loss.append(last_loss)
            self.history_.loglik.append(float(filt.loglik))
            self.history_.sigma2.append(float(theta.sigma2))
            self.history_.omega_trace.append(float(np.trace(theta.Omega)))
            self.history_.em_iters.append(int(em_info["n_iter"]))

            if self.verbose:
                print(
                    f"outer {len(self.history_.loss):3d}  "
                    f"loss={last_loss:12.4f}  loglik={filt.loglik:12.4f}  "
                    f"sigma2={theta.sigma2:.5f}  tr(Omega)={np.trace(theta.Omega):.3e}"
                )

        self.theta_ = theta

        # Final smoothed pass, for the time-varying weight plots.
        Z, z_tilde = self.net_.basis_numpy(U_t)
        Z = self._rescale(Z)
        target = ys - z_tilde if z_tilde is not None else ys
        sm = kalman_smoother(
            target, Z, theta.sigma2, theta.Omega, theta.b0, theta.V0
        )
        self.smoothed_states_ = sm.a_smooth
        self.smoothed_state_cov_ = sm.P_smooth
        return self

    # --------------------------------------------------------------- predict

    def _check_fitted(self) -> None:
        if self.net_ is None or self.theta_ is None:
            raise RuntimeError("Call fit() before predict()/forecast().")

    def _basis(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        Z, zt = self.net_.basis_numpy(self._tensor(self._scale_x(U)))
        return self._rescale(Z), zt

    def predict(self, U: np.ndarray | None = None) -> np.ndarray:
        """In-sample one-step-ahead fitted values, on the original scale.

        Uses the *predicted* state ``beta_{t|t-1}``, so each fitted value is a
        genuine out-of-sample prediction given the data up to that point.
        """
        self._check_fitted()
        if U is None:
            Us = self._U_train
            ys = self._y_train
        else:
            Us = self._scale_x(np.atleast_2d(np.asarray(U, dtype=float)))
            ys = None
        Z, zt = self.net_.basis_numpy(self._tensor(Us))
        Z = self._rescale(Z)
        if ys is None:
            raise ValueError(
                "predict() reproduces in-sample fitted values; pass U=None. "
                "For genuine out-of-sample points use forecast()."
            )
        target = ys - zt if zt is not None else ys
        filt = kalman_filter(
            target, Z, self.theta_.sigma2, self.theta_.Omega,
            self.theta_.b0, self.theta_.V0,
        )
        fitted = np.einsum("td,td->t", Z, filt.a_pred)
        if zt is not None:
            fitted = fitted + zt
        return np.asarray(self._unscale_y(fitted))

    def forecast(
        self,
        u_new: np.ndarray,
        gap: int = 0,
        return_variance: bool = False,
        U_hist: np.ndarray | None = None,
        y_hist: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Forecast ``y_{o+h}`` for one or more forecast origins.

        Parameters
        ----------
        u_new : (P,) or (k, P) array
            The predictor vector(s) ``u_o`` at the forecast origin(s), on the
            original scale.
        gap : int, default 0
            How many periods to propagate the terminal training state forward
            before applying it.  **Set this to ``h``.**  The measurement
            equation attaches ``y_{t+h}`` to state ``beta_t``, so if the
            training sample ends ``h`` periods before the origin -- which is
            what an honest pseudo-out-of-sample design requires -- the state
            at the origin is the last filtered state plus ``h`` random-walk
            steps:  ``E[beta_o] = beta_{T|T}``, ``Var = P_{T|T} + h * Omega``.
            :func:`tvwnn.evaluate.poos_experiment` sets this for you.
        return_variance : bool, default False
            Also return ``F = sigma2 + z' V z``, the predictive variance
            *conditional on the fitted network weights*.  The paper does not
            report density forecasts; this is an extension of ours and the
            variance ignores uncertainty in ``w``.
        U_hist, y_hist : array or None
            Run the Kalman filter over *this* history instead of the training
            sample, with the network weights and ``theta`` left frozen.  This
            is how the pseudo-out-of-sample engine folds in observations that
            arrived since the last re-estimation: the paper re-estimates the
            network only every 30 months, but the filter absorbs new data at
            almost no cost, which is the point of the state-space layer.
            Pass them on the original (unstandardised) scale.

        Returns
        -------
        ndarray, or (ndarray, ndarray) when ``return_variance``
        """
        self._check_fitted()
        u_new = np.atleast_2d(np.asarray(u_new, dtype=float))
        Z, zt = self._basis(u_new)

        if U_hist is None or y_hist is None:
            Us_hist, ys_hist = self._U_train, self._y_train
        else:
            Us_hist = self._scale_x(np.atleast_2d(np.asarray(U_hist, dtype=float)))
            y_arr = np.asarray(y_hist, dtype=float).ravel()
            ys_hist = (y_arr - self._y_mean) / self._y_std
            if Us_hist.shape[0] != ys_hist.shape[0]:
                raise ValueError("U_hist and y_hist must have the same length.")

        Z_train, zt_train = self.net_.basis_numpy(self._tensor(Us_hist))
        Z_train = self._rescale(Z_train)
        target = ys_hist - zt_train if zt_train is not None else ys_hist
        filt = kalman_filter(
            target, Z_train, self.theta_.sigma2, self.theta_.Omega,
            self.theta_.b0, self.theta_.V0,
        )
        mean, cov = forecast_state(filt, self.theta_.Omega, steps=gap)

        point = Z @ mean
        if zt is not None:
            point = point + zt
        point = np.asarray(self._unscale_y(point))

        if not return_variance:
            return point
        F = self.theta_.sigma2 + np.einsum("td,de,te->t", Z, cov, Z)
        return point, F * (self._y_std**2)

    # ------------------------------------------------------------- reporting

    def summary(self) -> dict[str, Any]:
        """Fitted variance components and the settings that produced them."""
        self._check_fitted()
        return {
            "d": self.d,
            "hidden_sizes": self.hidden_sizes,
            "loss": self.loss,
            "state": self.state,
            "sn_cap": self.sn_cap,
            "sigma2": float(self.theta_.sigma2),
            "Omega": self.theta_.Omega.copy(),
            "omega_trace": float(np.trace(self.theta_.Omega)),
            "b0": self.theta_.b0.copy(),
            "final_loglik": self.history_.loglik[-1] if self.history_.loglik else np.nan,
            "n_outer": self.n_outer,
            "grad_steps": self.grad_steps,
            "total_epochs": self.n_outer * self.grad_steps,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
        }


class TVNNEnsemble:
    """Average of ``n_members`` independently initialised TVNNs.

    Appendix A.3: "To reduce the variance of predictions, ensembles of 10
    networks were used for each evaluation period."  Tables 1 and 2 of the
    paper are therefore ensemble forecasts -- a single network will not
    reproduce them.

    Parameters
    ----------
    n_members : int, default 10
    random_state : int or None
        Seeds members ``random_state, random_state + 1, ...``.
    **kwargs
        Passed straight to :class:`TVNN`.
    """

    def __init__(
        self,
        n_members: int = 10,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_members = int(n_members)
        self.random_state = random_state
        self.kwargs = kwargs
        self.members_: list[TVNN] = []

    def fit(self, U: np.ndarray, y: np.ndarray) -> "TVNNEnsemble":
        base = 0 if self.random_state is None else int(self.random_state)
        self.members_ = []
        for m in range(self.n_members):
            seed = None if self.random_state is None else base + m
            member = TVNN(random_state=seed, **self.kwargs)
            member.fit(U, y)
            self.members_.append(member)
        return self

    @property
    def smoothed_states_(self) -> np.ndarray:
        """Smoothed states of the first member.

        Present so the evaluation engine can detect a state-carrying model and
        re-filter it at every origin.  The members' bases are unrelated to one
        another (see "Identification" in the README), so averaging their states
        would be meaningless -- only their *forecasts* are averaged.
        """
        if not self.members_:
            raise RuntimeError("Call fit() before accessing smoothed_states_.")
        return self.members_[0].smoothed_states_

    def forecast(
        self,
        u_new: np.ndarray,
        gap: int = 0,
        U_hist: np.ndarray | None = None,
        y_hist: np.ndarray | None = None,
    ) -> np.ndarray:
        """Mean forecast across members."""
        if not self.members_:
            raise RuntimeError("Call fit() before forecast().")
        preds = np.stack(
            [m.forecast(u_new, gap=gap, U_hist=U_hist, y_hist=y_hist)
             for m in self.members_]
        )
        return preds.mean(axis=0)

    def forecast_spread(self, u_new: np.ndarray, gap: int = 0) -> np.ndarray:
        """Standard deviation across members -- a crude uncertainty proxy."""
        if not self.members_:
            raise RuntimeError("Call fit() before forecast_spread().")
        preds = np.stack([m.forecast(u_new, gap=gap) for m in self.members_])
        return preds.std(axis=0, ddof=1) if len(preds) > 1 else np.zeros(preds.shape[1])
