"""
The feedforward network that learns the basis ``z_t = f_w(u_t)``.

Architecture follows Appendix A.3 of Rudd, Bondell & Silver (2026):

    u_t  ->  Dense(32, tanh)  ->  [Dense(16, tanh)]  ->  Dense(d)  ->  z_t

Two details from A.3 that are easy to misread from the main text alone:

1.  ``d in {1, 2}``.  The "final hidden layer" of the paper is therefore a
    *bottleneck output layer of width one or two*, not the 32-unit hidden
    layer.  The time-varying state is one- or two-dimensional.
2.  The activation is ``tanh`` throughout, "chosen to allow for direct
    comparisons between networks with one and two hidden layers, and between
    the feedforward and recurrent architectures".

Weights are initialised with Glorot/Xavier uniform, as A.3 specifies.

The optional fixed-effect head implements the variant of eq. (4):

    y_{t+h} = z_t' beta_t + z_tilde_t + eps_t

where ``z_tilde_t`` is "another output node of the neural network", i.e. an
unrestricted scalar output carrying the non-time-varying part of the signal.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

__all__ = ["BasisNetwork", "glorot_init"]


def glorot_init(module: nn.Module) -> None:
    """Glorot & Bengio uniform initialisation with zero biases (A.3)."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BasisNetwork(nn.Module):
    """Maps the predictor vector ``u_t`` to the learned basis ``z_t``.

    Parameters
    ----------
    n_inputs : int
        Dimension ``P`` of ``u_t``.
    d : int, default 1
        Number of basis functions -- the dimension of the time-varying state.
        Appendix A.3 selects from ``{1, 2}``.
    hidden_sizes : sequence of int, default ``(32,)``
        Hidden layer widths.  A.3 uses ``(32,)`` for one hidden layer and
        ``(32, 16)`` for two.
    activation : {"tanh", "relu"}, default "tanh"
        A.3 uses ``tanh``.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.  A.3 says the
        feedforward architectures "leverage dropout with default parameters"
        without naming a rate, so we default to *off* and let the caller set
        it.  Dropout is active only in training mode: the basis handed to the
        Kalman recursions is always computed with the network in eval mode, so
        ``z_t`` is deterministic there.
    fixed_effect : bool, default False
        Add the scalar ``z_tilde_t`` output node of eq. (4).
    """

    def __init__(
        self,
        n_inputs: int,
        d: int = 1,
        hidden_sizes: Sequence[int] = (32,),
        activation: str = "tanh",
        dropout: float = 0.0,
        fixed_effect: bool = False,
    ) -> None:
        super().__init__()
        if d < 1:
            raise ValueError("d must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        acts = {"tanh": nn.Tanh, "relu": nn.ReLU}
        if activation not in acts:
            raise ValueError(f"activation must be one of {sorted(acts)}.")

        self.n_inputs = int(n_inputs)
        self.d = int(d)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.has_fixed_effect = bool(fixed_effect)

        layers: list[nn.Module] = []
        prev = self.n_inputs
        for width in self.hidden_sizes:
            layers.append(nn.Linear(prev, width))
            layers.append(acts[activation]())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = width
        self.body = nn.Sequential(*layers)

        # The bottleneck that produces the basis.  Linear, no activation: the
        # TVP layer sits directly on top of it.
        self.basis = nn.Linear(prev, self.d)
        self.fixed_head = nn.Linear(prev, 1) if fixed_effect else None

        self.apply(glorot_init)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(z, z_tilde)``.

        ``z`` has shape ``(n, d)``; ``z_tilde`` has shape ``(n,)`` when the
        fixed-effect head is enabled and is ``None`` otherwise.
        """
        hidden = self.body(u)
        z = self.basis(hidden)
        z_tilde = self.fixed_head(hidden).squeeze(-1) if self.fixed_head else None
        return z, z_tilde

    @torch.no_grad()
    def basis_numpy(self, u: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None]:
        """Deterministic basis for the Kalman recursions.

        Puts the network in eval mode (so dropout is disabled), evaluates, and
        restores the previous mode.  This is the ``z_t = f_{w^(j)}(u_t)`` line
        at the top of Algorithm 1's outer loop.
        """
        was_training = self.training
        self.eval()
        try:
            z, z_tilde = self.forward(u)
            z_np = z.detach().cpu().numpy().astype(float)
            zt_np = (
                z_tilde.detach().cpu().numpy().astype(float)
                if z_tilde is not None
                else None
            )
        finally:
            if was_training:
                self.train()
        return z_np, zt_np

    def l2_penalty(self) -> torch.Tensor:
        """Sum of squared weights, the ``||w||^2`` of Algorithm 1.

        Biases are excluded, which is the usual convention and matches what a
        Keras/PyTorch ``kernel_regularizer`` penalises.
        """
        total = torch.zeros((), dtype=torch.get_default_dtype())
        for name, param in self.named_parameters():
            if name.endswith("weight"):
                total = total + torch.sum(param**2)
        return total

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"n_inputs={self.n_inputs}, d={self.d}, "
            f"hidden_sizes={self.hidden_sizes}, "
            f"fixed_effect={self.has_fixed_effect}"
        )
