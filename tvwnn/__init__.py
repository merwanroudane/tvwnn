"""
tvwnn -- Time-Varying Weight Neural Networks
============================================

A faithful Python implementation of the TVNN of

    Rudd, W., H. Bondell & J. Silver (2026),
    "Augmenting Neural Networks With Time-Varying Weights",
    *Journal of Forecasting* 45(1): 22-28.  doi:10.1002/for.70014

A feedforward neural network learns a low-dimensional basis ``z_t = f_w(u_t)``;
the weights on that basis follow a random walk and are estimated by a Kalman
filter/smoother inside an EM algorithm, alternating with Adam steps on the
network weights.

Quick start
-----------
>>> from tvwnn import TVNN, simulate_tvnn
>>> ds, truth = simulate_tvnn(n=300, seed=0)
>>> model = TVNN(d=1, n_outer=20, grad_steps=10, random_state=0).fit(ds.U, ds.y)
>>> model.summary()["sigma2"]        # doctest: +SKIP

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
Licence: MIT
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"
__url__ = "https://github.com/merwanroudane/tvwnn"

from .baselines import (
    ARBenchmark,
    FeedForwardForecaster,
    RecurrentForecaster,
    RidgeARX,
)
from .data import (
    Dataset,
    PCAProjector,
    build_fred_dataset,
    build_supervised,
    diffusion_indices,
    fetch_fred_md,
    fetch_fred_qd,
    make_target,
    simulate_tvnn,
    target_kind_from_tcode,
    transform_panel,
)
from .em import StateParams, run_em
from .evaluate import (
    A3_GRID,
    A3_GRID_RECURRENT,
    PoosResult,
    dm_test,
    model_confidence_set,
    mse_table,
    poos_experiment,
    sample_hyperparameters,
    select_hyperparameters,
)
from .kalman import kalman_filter, kalman_smoother
from .model import TVNN, TVNNEnsemble
from .network import BasisNetwork
from .plots import (
    plot_convergence,
    plot_cssed,
    plot_forecast_vs_actual,
    plot_mse_by_horizon,
    plot_scree,
    plot_time_varying_weights,
    save_figure,
    set_journal_style,
)
from .tables import (
    dm_matrix,
    forecast_accuracy_table,
    model_summary_table,
    render_console,
    render_latex,
)

__all__ = [
    "__version__",
    # model
    "TVNN",
    "TVNNEnsemble",
    "BasisNetwork",
    "StateParams",
    # state space
    "kalman_filter",
    "kalman_smoother",
    "run_em",
    # baselines
    "ARBenchmark",
    "RidgeARX",
    "FeedForwardForecaster",
    "RecurrentForecaster",
    # data
    "Dataset",
    "PCAProjector",
    "fetch_fred_md",
    "fetch_fred_qd",
    "transform_panel",
    "diffusion_indices",
    "make_target",
    "target_kind_from_tcode",
    "build_supervised",
    "build_fred_dataset",
    "simulate_tvnn",
    # evaluation
    "poos_experiment",
    "PoosResult",
    "dm_test",
    "mse_table",
    "model_confidence_set",
    "A3_GRID",
    "A3_GRID_RECURRENT",
    "sample_hyperparameters",
    "select_hyperparameters",
    # reporting
    "forecast_accuracy_table",
    "render_console",
    "render_latex",
    "model_summary_table",
    "dm_matrix",
    "set_journal_style",
    "plot_mse_by_horizon",
    "plot_time_varying_weights",
    "plot_forecast_vs_actual",
    "plot_cssed",
    "plot_convergence",
    "plot_scree",
    "save_figure",
]
