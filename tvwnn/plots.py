"""
Publication-quality figures.

Everything here renders on a **white background** with a colour-blind-safe
palette, thin spines, no top/right axis, and a serif font stack -- the look of
a journal figure rather than a notebook default.  There is no dark variant by
design: dark figures do not print.

Figures provided
----------------
``plot_mse_by_horizon``        Figures 1 and 2 of the paper
``plot_time_varying_weights``  the fitted ``beta_hat_t`` path with a band
``plot_forecast_vs_actual``    forecasts against realisations, with recessions
``plot_cssed``                 cumulative squared error difference vs a benchmark
``plot_convergence``           the EM/Adam alternation, for diagnostics
``plot_scree``                 variance explained by the diffusion indices

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "JOURNAL_PALETTE",
    "NBER_RECESSIONS",
    "set_journal_style",
    "plot_mse_by_horizon",
    "plot_time_varying_weights",
    "plot_forecast_vs_actual",
    "plot_cssed",
    "plot_convergence",
    "plot_scree",
    "save_figure",
]

#: Colour-blind-safe qualitative palette (Okabe-Ito), all legible on white.
JOURNAL_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black, for the benchmark
]

#: NBER recession months over the paper's evaluation window and a little beyond.
NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]


def set_journal_style(font_scale: float = 1.0) -> None:
    """Apply the light, print-ready style used by every figure here.

    Call once at the top of a script.  Individual plot functions call it
    themselves, so this is only needed if you are drawing your own axes and
    want them to match.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "figure.dpi": 120,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11 * font_scale,
            "axes.titleweight": "bold",
            "axes.labelsize": 10 * font_scale,
            "axes.prop_cycle": mpl.cycler(color=JOURNAL_PALETTE),
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.6,
            "xtick.labelsize": 9 * font_scale,
            "ytick.labelsize": 9 * font_scale,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "#CCCCCC",
            "legend.fontsize": 9 * font_scale,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia", "serif"],
            "mathtext.fontset": "dejavuserif",
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
        }
    )


def _shade_recessions(ax: plt.Axes, spans: Sequence[tuple[str, str]] = NBER_RECESSIONS) -> None:
    lo, hi = ax.get_xlim()
    for start, end in spans:
        s = mpl.dates.date2num(pd.Timestamp(start))
        e = mpl.dates.date2num(pd.Timestamp(end))
        if e < lo or s > hi:
            continue
        ax.axvspan(max(s, lo), min(e, hi), color="#000000", alpha=0.07, lw=0, zorder=0)


def plot_mse_by_horizon(
    mse_by_model: Mapping[str, Mapping[int, float]],
    title: str = "",
    ylabel: str = "MSE",
    highlight: str | None = "TVNN",
    ax: plt.Axes | None = None,
):
    """Reproduce Figures 1 and 2: MSE against forecast horizon, one line per model.

    Parameters
    ----------
    mse_by_model : mapping model -> (horizon -> MSE)
        Pass the *best* predictor set per model, as the paper's figures do.
    highlight : str or None
        Drawn thicker with filled markers so it reads first.

    Returns
    -------
    (fig, ax)
    """
    set_journal_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.2, 4.0))

    for i, (name, series) in enumerate(mse_by_model.items()):
        hs = sorted(series)
        vals = [series[h] for h in hs]
        is_hi = name == highlight
        ax.plot(
            hs,
            vals,
            marker="o",
            color=JOURNAL_PALETTE[i % len(JOURNAL_PALETTE)],
            lw=2.6 if is_hi else 1.4,
            ms=7 if is_hi else 5,
            zorder=5 if is_hi else 3,
            label=name,
            alpha=1.0 if is_hi else 0.85,
        )

    ax.set_xlabel("Forecast horizon $h$ (months)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    all_h = sorted({h for s in mse_by_model.values() for h in s})
    ax.set_xticks(all_h)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    return fig, ax


def plot_time_varying_weights(
    states: np.ndarray,
    dates: Sequence | None = None,
    state_cov: np.ndarray | None = None,
    title: str = "Smoothed time-varying weights",
    shade_recessions: bool = True,
    ax: plt.Axes | None = None,
):
    """Plot ``beta_hat_t`` with a pointwise band.

    Parameters
    ----------
    states : (n, d) array
        ``model.smoothed_states_``.
    dates : sequence or None
        ``dataset.dates``; falls back to an integer axis.
    state_cov : (n, d, d) array or None
        ``model.smoothed_state_cov_``; draws a +/- 2 s.e. band.
    shade_recessions : bool
        Shade NBER recessions, which is where drift should show up.

    Notes
    -----
    The level and sign of each ``beta_j`` are **not identified**: the model is
    invariant to ``z -> A'z``, ``beta -> A^{-1}beta``.  Read the *shape* of
    these paths -- when they move, and how much -- not their values, and do not
    compare them across runs or seeds.
    """
    set_journal_style()
    states = np.atleast_2d(np.asarray(states, dtype=float))
    n, d = states.shape
    x = pd.DatetimeIndex(dates) if dates is not None else np.arange(n)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.0, 3.8))

    for j in range(d):
        colour = JOURNAL_PALETTE[j % len(JOURNAL_PALETTE)]
        ax.plot(x, states[:, j], color=colour, label=rf"$\hat\beta_{{{j + 1},t}}$")
        if state_cov is not None:
            se = np.sqrt(np.maximum(np.asarray(state_cov)[:, j, j], 0.0))
            ax.fill_between(
                x,
                states[:, j] - 2 * se,
                states[:, j] + 2 * se,
                color=colour,
                alpha=0.15,
                lw=0,
            )

    ax.axhline(0.0, color="#888888", lw=0.8, ls=(0, (4, 3)), zorder=1)
    if shade_recessions and dates is not None:
        _shade_recessions(ax)
    ax.set_ylabel("Weight")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_forecast_vs_actual(
    forecasts: pd.DataFrame,
    actual_col: str = "actual",
    models: Sequence[str] | None = None,
    title: str = "",
    ylabel: str = "",
    shade_recessions: bool = True,
    ax: plt.Axes | None = None,
):
    """Overlay forecasts on the realised series.

    The realisation is drawn in black so the models read as departures from it.
    """
    set_journal_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.4, 4.0))

    cols = models or [c for c in forecasts.columns if c != actual_col]
    ax.plot(
        forecasts.index,
        forecasts[actual_col],
        color="#000000",
        lw=2.0,
        label="Realised",
        zorder=6,
    )
    for i, col in enumerate(cols):
        ax.plot(
            forecasts.index,
            forecasts[col],
            color=JOURNAL_PALETTE[i % len(JOURNAL_PALETTE)],
            lw=1.3,
            alpha=0.9,
            label=col,
        )

    if shade_recessions:
        _shade_recessions(ax)
    ax.axhline(0.0, color="#888888", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    return fig, ax


def plot_cssed(
    errors: pd.DataFrame,
    benchmark: str,
    models: Sequence[str] | None = None,
    title: str = "Cumulative squared error difference vs. benchmark",
    shade_recessions: bool = True,
    ax: plt.Axes | None = None,
):
    """Cumulative squared error difference (Welch & Goyal style).

    Plots ``sum_{s<=t} [ e_bench_s^2 - e_model_s^2 ]``.  A **rising** line means
    the model is beating the benchmark over that stretch; a flat line means it
    is neither gaining nor losing.  This is the figure that shows *when* the
    gain accrues -- the paper reports only period-average MSE, which hides
    whether an edge comes from a handful of turning points or from everywhere.
    """
    set_journal_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.4, 4.0))

    if benchmark not in errors.columns:
        raise KeyError(f"{benchmark!r} is not a column of `errors`.")
    base = errors[benchmark].to_numpy() ** 2
    cols = models or [c for c in errors.columns if c != benchmark]

    for i, col in enumerate(cols):
        diff = np.nancumsum(base - errors[col].to_numpy() ** 2)
        ax.plot(
            errors.index,
            diff,
            color=JOURNAL_PALETTE[i % len(JOURNAL_PALETTE)],
            label=col,
            lw=2.2 if col == "TVNN" else 1.4,
            zorder=5 if col == "TVNN" else 3,
        )

    ax.axhline(0.0, color="#000000", lw=1.0)
    if shade_recessions:
        _shade_recessions(ax)
    ax.set_ylabel(f"CSSED vs. {benchmark}")
    ax.set_title(title)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    return fig, ax


def plot_convergence(history: Mapping[str, Sequence[float]], title: str = "TVNN training"):
    """Three-panel diagnostic of the EM/Adam alternation.

    Panels: the network objective, the marginal log-likelihood from the filter,
    and the variance components.  The log-likelihood should trend upward; large
    sawtooth swings mean the two loops are fighting and ``grad_steps`` is too
    large relative to the learning rate.
    """
    set_journal_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    it = np.arange(1, len(history["loss"]) + 1)
    axes[0].plot(it, history["loss"], color=JOURNAL_PALETTE[0])
    axes[0].set_title("Network objective")
    axes[0].set_xlabel("Outer iteration")

    axes[1].plot(it, history["loglik"], color=JOURNAL_PALETTE[2])
    axes[1].set_title("Marginal log-likelihood")
    axes[1].set_xlabel("Outer iteration")

    axes[2].plot(it, history["sigma2"], color=JOURNAL_PALETTE[1], label=r"$\sigma^2$")
    axes[2].plot(
        it, history["omega_trace"], color=JOURNAL_PALETTE[3], label=r"tr$(\Omega)$"
    )
    axes[2].set_yscale("log")
    axes[2].set_title("Variance components")
    axes[2].set_xlabel("Outer iteration")
    axes[2].legend(loc="best")

    fig.suptitle(title, y=1.04, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_scree(
    explained_variance_ratio: Sequence[float],
    title: str = "Diffusion indices: variance explained",
    ax: plt.Axes | None = None,
):
    """Scree plot for the principal components used as predictors."""
    set_journal_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5.6, 3.6))
    evr = np.asarray(explained_variance_ratio, dtype=float)
    k = np.arange(1, evr.size + 1)

    ax.bar(k, 100 * evr, color=JOURNAL_PALETTE[0], alpha=0.85, width=0.65)
    ax2 = ax.twinx()
    ax2.plot(k, 100 * np.cumsum(evr), color=JOURNAL_PALETTE[1], marker="o")
    ax2.set_ylabel("Cumulative (%)", color=JOURNAL_PALETTE[1])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(JOURNAL_PALETTE[1])
    ax2.tick_params(axis="y", colors=JOURNAL_PALETTE[1])

    ax.set_xticks(k)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def save_figure(fig, path: str, formats: Sequence[str] = ("png", "pdf")) -> list[str]:
    """Save a figure in several formats at 300 dpi; returns the paths written."""
    from pathlib import Path

    out = []
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = base.with_suffix(f".{fmt}")
        fig.savefig(target, format=fmt, dpi=300, bbox_inches="tight", facecolor="white")
        out.append(str(target))
    return out
