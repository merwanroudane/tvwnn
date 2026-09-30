"""
Competing models from Table 1 and Table 2 of the paper.

All five share the interface of :class:`tvwnn.model.TVNN` -- ``fit(U, y)`` and
``forecast(u_new, gap=...)`` -- so the evaluation engine can treat them
interchangeably.  The ``gap`` argument is accepted and ignored by the
fixed-parameter models: only the TVNN carries a state that has to be
propagated.

* :class:`ARBenchmark`          -- the AR(4) benchmark (``P = 4``)
* :class:`RidgeARX`             -- ARX regularised with ridge
* :class:`FeedForwardForecaster`-- FFNN, the "similarly tuned" fixed-weight net
* :class:`RecurrentForecaster`  -- Elman RNN (Elman 1990) and LSTM
                                   (Hochreiter & Schmidhuber 1997)

Network settings follow Appendix A.3: ``tanh``, hidden sizes ``(32,)`` or
``(32, 16)``, Glorot initialisation, Adam, a fixed 300-epoch budget with no
early stopping, and an L2 penalty on the weights.  The recurrent
architectures use one hidden layer and heavier regularisation there.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from .network import glorot_init

__all__ = [
    "ARBenchmark",
    "RidgeARX",
    "FeedForwardForecaster",
    "RecurrentForecaster",
    "sequence_view",
]


class _Scaler:
    """Standardise inputs and target; invert on output."""

    def fit(self, U: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.x_mean = U.mean(axis=0)
        sd = U.std(axis=0)
        sd[sd < 1e-12] = 1.0
        self.x_std = sd
        self.y_mean = float(y.mean())
        ys = float(y.std())
        self.y_std = ys if ys > 1e-12 else 1.0
        return (U - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std

    def x(self, U: np.ndarray) -> np.ndarray:
        return (U - self.x_mean) / self.x_std

    def inv_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


# ------------------------------------------------------------------- linear


class ARBenchmark:
    """Ordinary least squares with an intercept.

    Section 4: "we deployed an autoregressive (AR) model using only the
    differenced target series and its lags as predictors".  Build its design
    matrix with ``factors=None`` so ``U`` holds the target lags alone.
    Solved by least squares with a pseudo-inverse, so collinear lags degrade
    gracefully instead of raising.
    """

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, U: np.ndarray, y: np.ndarray) -> "ARBenchmark":
        U = np.atleast_2d(np.asarray(U, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        X = np.hstack([np.ones((U.shape[0], 1)), U])
        self.coef_ = np.linalg.pinv(X) @ y
        return self

    def forecast(self, u_new: np.ndarray, gap: int = 0) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() first.")
        u_new = np.atleast_2d(np.asarray(u_new, dtype=float))
        X = np.hstack([np.ones((u_new.shape[0], 1)), u_new])
        return X @ self.coef_


class RidgeARX:
    """ARX regularised with ridge, penalty chosen by generalised cross-validation.

    Section 4: "an ARX model regularized with ridge regression".  The paper
    does not say how the penalty was selected; we use ``RidgeCV`` over a log
    grid, which is the standard choice.
    """

    def __init__(self, alphas: Sequence[float] | None = None) -> None:
        self.alphas = np.logspace(-3, 4, 40) if alphas is None else np.asarray(alphas)
        self.model_ = None
        self.scaler_ = _Scaler()

    def fit(self, U: np.ndarray, y: np.ndarray) -> "RidgeARX":
        from sklearn.linear_model import RidgeCV

        U = np.atleast_2d(np.asarray(U, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        Us, ys = self.scaler_.fit(U, y)
        self.model_ = RidgeCV(alphas=self.alphas).fit(Us, ys)
        return self

    def forecast(self, u_new: np.ndarray, gap: int = 0) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        u_new = np.atleast_2d(np.asarray(u_new, dtype=float))
        return self.scaler_.inv_y(self.model_.predict(self.scaler_.x(u_new)))


# ------------------------------------------------------------------ networks


class _TorchForecaster:
    """Shared Adam training loop for the neural baselines."""

    def __init__(
        self,
        epochs: int = 300,
        learning_rate: float = 0.01,
        l2: float = 0.32,
        random_state: int | None = None,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.random_state = random_state
        self.device = torch.device(device) if device else torch.device("cpu")
        self.verbose = bool(verbose)
        self.net_: nn.Module | None = None
        self.scaler_ = _Scaler()
        self.loss_path_: list[float] = []

    def _build(self, n_inputs: int) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    def _prepare(self, U: np.ndarray) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def _l2(self) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64)
        for name, param in self.net_.named_parameters():
            if "weight" in name:
                total = total + torch.sum(param**2)
        return total

    def fit(self, U: np.ndarray, y: np.ndarray) -> "_TorchForecaster":
        U = np.atleast_2d(np.asarray(U, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        Us, ys = self.scaler_.fit(U, y)

        self.net_ = self._build(Us.shape[1]).to(self.device).double()
        x = self._prepare(Us)
        target = torch.as_tensor(ys, device=self.device, dtype=torch.float64)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)

        self.net_.train()
        self.loss_path_ = []
        for epoch in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            pred = self.net_(x).squeeze(-1)
            # Mean, not sum, and the same convention as the TVNN's objective --
            # see "the loss is a mean" in the README.  A.3 says the networks use
            # "heavy regularization"; only the averaged reading makes its penalty
            # grid (0.02 to 0.64) heavy rather than negligible.
            loss = torch.mean((target - pred) ** 2) + self.l2 * self._l2()
            loss.backward()
            opt.step()
            self.loss_path_.append(float(loss.detach()))
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  epoch {epoch + 1:4d}  loss={self.loss_path_[-1]:.6f}")
        self.net_.eval()
        return self

    @torch.no_grad()
    def forecast(self, u_new: np.ndarray, gap: int = 0) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("Call fit() first.")
        u_new = np.atleast_2d(np.asarray(u_new, dtype=float))
        x = self._prepare(self.scaler_.x(u_new))
        pred = self.net_(x).squeeze(-1).cpu().numpy()
        return self.scaler_.inv_y(np.atleast_1d(pred))


class FeedForwardForecaster(_TorchForecaster):
    """Fixed-weight feedforward network -- the FFNN row of Tables 1 and 2.

    This is the direct comparator for the TVNN: identical architecture, but the
    final layer's weights are constant rather than drifting.  Holding
    ``hidden_sizes``, ``learning_rate``, ``l2`` and ``epochs`` equal across the
    two is what makes the comparison "similarly tuned".
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (32,),
        activation: str = "tanh",
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation = activation
        self.dropout = float(dropout)

    def _build(self, n_inputs: int) -> nn.Module:
        acts = {"tanh": nn.Tanh, "relu": nn.ReLU}
        layers: list[nn.Module] = []
        prev = n_inputs
        for width in self.hidden_sizes:
            layers += [nn.Linear(prev, width), acts[self.activation]()]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        net = nn.Sequential(*layers)
        net.apply(glorot_init)
        return net

    def _prepare(self, U: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(U, device=self.device, dtype=torch.float64)


def sequence_view(
    U: np.ndarray,
    kx: int,
    ky: int,
    n_factors: int,
) -> np.ndarray:
    """Reassemble the flat design ``u_t`` into a ``(n, L, k+1)`` sequence.

    :func:`tvwnn.data.build_lag_matrix` lays ``u_t`` out lag-major -- all
    factors at lag 0, all factors at lag 1, ..., then the target lags.  The
    recurrent architectures need ``(timestep, feature)`` instead, ordered
    oldest-to-newest, which is what this returns.  Slots with no data (when
    ``kx != ky``) are zero-filled.
    """
    U = np.atleast_2d(np.asarray(U, dtype=float))
    n = U.shape[0]
    L = max(kx, ky)
    k = n_factors
    expected = kx * k + ky
    if U.shape[1] != expected:
        raise ValueError(
            f"U has {U.shape[1]} columns but kx*n_factors + ky = {expected}."
        )
    out = np.zeros((n, L, k + 1))
    for step in range(L):
        lag = L - 1 - step  # oldest first
        if lag < kx and k > 0:
            out[:, step, :k] = U[:, lag * k : (lag + 1) * k]
        if lag < ky:
            out[:, step, k] = U[:, kx * k + lag]
    return out


class RecurrentForecaster(_TorchForecaster):
    """Elman RNN or LSTM over the lag sequence.

    Parameters
    ----------
    cell : {"rnn", "lstm"}
        ``"rnn"`` is the Elman network of Elman (1990); ``"lstm"`` is
        Hochreiter & Schmidhuber (1997).  Both use one hidden layer, per A.3.
    kx, ky, n_factors : int
        Needed to rebuild the ``(timestep, feature)`` view; pass the same
        values used to construct the dataset.
    hidden_size : int, default 32
    l2 : float
        A.3 gives the recurrent architectures heavier direct regularisation,
        ``0.01 * 2**s`` with ``s in {4, 5, 6}``, i.e. 0.16 to 0.64.
    """

    def __init__(
        self,
        cell: str = "rnn",
        kx: int = 4,
        ky: int = 4,
        n_factors: int = 5,
        hidden_size: int = 32,
        l2: float = 0.64,
        **kwargs,
    ) -> None:
        kwargs.setdefault("l2", l2)
        super().__init__(**kwargs)
        if cell not in {"rnn", "lstm"}:
            raise ValueError("cell must be 'rnn' or 'lstm'.")
        self.cell = cell
        self.kx = int(kx)
        self.ky = int(ky)
        self.n_factors = int(n_factors)
        self.hidden_size = int(hidden_size)

    def _build(self, n_inputs: int) -> nn.Module:
        n_features = self.n_factors + 1
        hidden = self.hidden_size
        cell_cls = nn.RNN if self.cell == "rnn" else nn.LSTM
        rnn = cell_cls(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            nonlinearity="tanh",
        ) if self.cell == "rnn" else cell_cls(
            input_size=n_features, hidden_size=hidden, num_layers=1, batch_first=True
        )
        head = nn.Linear(hidden, 1)
        head.apply(glorot_init)

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = rnn
                self.head = head

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :])

        return _Net()

    def _prepare(self, U: np.ndarray) -> torch.Tensor:
        seq = sequence_view(U, self.kx, self.ky, self.n_factors)
        return torch.as_tensor(seq, device=self.device, dtype=torch.float64)
