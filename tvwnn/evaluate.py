"""
Pseudo-out-of-sample evaluation, the Diebold-Mariano test, and a Model
Confidence Set.

Reproduces the evaluation protocol of section 4 of Rudd, Bondell & Silver
(2026):

* expanding training window starting in 1960;
* models re-estimated every ``retrain_every`` months (30 in the paper);
* a pseudo-out-of-sample period (1990-01 to 2014-12 in the paper);
* mean squared error, with Diebold-Mariano tests against a benchmark.

Two things this module does that the paper does not:

* the principal components are re-estimated **inside each training window**,
  so the diffusion indices carry no information from the evaluation period;
* a Model Confidence Set is available, because the paper reports many
  pairwise Diebold-Mariano tests with no multiplicity control.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .data import (
    PCAProjector,
    build_supervised,
    make_target,
    target_kind_from_tcode,
    transform_panel,
)

__all__ = [
    "PoosResult",
    "dm_test",
    "mse_table",
    "model_confidence_set",
    "poos_experiment",
]


# ------------------------------------------------------------ Diebold-Mariano


@dataclass
class DMResult:
    """Outcome of :func:`dm_test`."""

    statistic: float
    p_value: float
    mean_loss_differential: float
    n: int
    h: int
    harvey_adjusted: bool

    def stars(self, levels: Sequence[float] = (0.10, 0.05, 0.01)) -> str:
        """Significance markers, in the paper's convention (0.1 / 0.05 / 0.01)."""
        out = ""
        for lvl in sorted(levels, reverse=True):
            if self.p_value < lvl:
                out += "*"
        return out


def dm_test(
    errors_model: np.ndarray,
    errors_benchmark: np.ndarray,
    h: int = 1,
    loss: str = "squared",
    harvey_adjustment: bool = True,
    alternative: str = "two-sided",
) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy.

    Tests ``H0: E[L(e_model) - L(e_benchmark)] = 0``.  A *negative* statistic
    favours the model over the benchmark.

    Parameters
    ----------
    errors_model, errors_benchmark : (n,) arrays
        Forecast errors ``actual - forecast``.
    h : int, default 1
        Forecast horizon.  The long-run variance uses a rectangular kernel
        truncated at ``h - 1`` lags, which is the Diebold-Mariano (1995)
        prescription for ``h``-step-ahead forecasts.  The paper reports DM
        tests at ``h = 3, 6, 9, 12`` without stating the truncation; this is
        the standard choice.
    loss : {"squared", "absolute"}
    harvey_adjustment : bool, default True
        Apply the Harvey, Leybourne & Newbold (1997) small-sample correction
        and refer the statistic to ``t(n - 1)`` rather than the normal.  The
        paper does not say whether it did; with 300 evaluation points the
        difference is small but not zero.
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    DMResult
    """
    e1 = np.asarray(errors_model, dtype=float).ravel()
    e2 = np.asarray(errors_benchmark, dtype=float).ravel()
    if e1.shape != e2.shape:
        raise ValueError("The two error series must have the same length.")
    ok = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[ok], e2[ok]
    n = e1.size
    if n < 3:
        raise ValueError("Need at least 3 paired forecast errors.")

    if loss == "squared":
        d = e1**2 - e2**2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'.")

    dbar = float(np.mean(d))
    dc = d - dbar
    gamma0 = float(np.dot(dc, dc) / n)
    var = gamma0
    for k in range(1, max(h, 1)):
        if k >= n:
            break
        gamma_k = float(np.dot(dc[k:], dc[:-k]) / n)
        var += 2.0 * gamma_k
    var = max(var, 1e-16)

    stat = dbar / np.sqrt(var / n)

    if harvey_adjustment:
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        stat *= np.sqrt(max(factor, 1e-12))
        dist = stats.t(df=n - 1)
    else:
        dist = stats.norm()

    if alternative == "two-sided":
        p = float(2.0 * dist.sf(abs(stat)))
    elif alternative == "less":
        p = float(dist.cdf(stat))
    elif alternative == "greater":
        p = float(dist.sf(stat))
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'.")

    return DMResult(
        statistic=float(stat),
        p_value=min(p, 1.0),
        mean_loss_differential=dbar,
        n=n,
        h=h,
        harvey_adjusted=bool(harvey_adjustment),
    )


# ------------------------------------------------------- Model Confidence Set


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    n_boot: int = 1000,
    block_size: int | None = None,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Hansen, Lunde & Nason Model Confidence Set, range statistic.

    Iteratively eliminates the worst model until the null of equal predictive
    ability can no longer be rejected.  The surviving set contains the best
    model with probability at least ``1 - alpha``.

    The paper reports pairwise Diebold-Mariano tests across two targets, four
    horizons, three predictor sets and six models with no multiplicity
    adjustment; this is the correction.

    Parameters
    ----------
    losses : DataFrame of shape (n, k)
        One column of per-period losses (e.g. squared errors) per model.
    alpha : float, default 0.10
    n_boot : int, default 1000
    block_size : int or None
        Circular block bootstrap block length; defaults to ``ceil(sqrt(n))``.
    seed : int or None

    Returns
    -------
    DataFrame indexed by model name with columns ``eliminated_at``,
    ``mcs_pvalue`` and ``in_mcs``.
    """
    losses = losses.dropna(axis=0, how="any")
    names = list(losses.columns)
    L = losses.to_numpy(dtype=float)
    n, k = L.shape
    if k < 2:
        raise ValueError("Need at least two models.")
    if block_size is None:
        block_size = int(np.ceil(np.sqrt(n)))
    rng = np.random.default_rng(seed)

    # Circular block bootstrap indices, shared across all elimination rounds.
    n_blocks = int(np.ceil(n / block_size))
    boot_idx = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate(
            [(s + np.arange(block_size)) % n for s in starts]
        )[:n]
        boot_idx[b] = idx

    alive = list(range(k))
    eliminated_at: dict[str, int] = {}
    pvalues: dict[str, float] = {}
    running_max = 0.0
    step = 0

    while len(alive) > 1:
        step += 1
        sub = L[:, alive]
        m = sub.shape[1]
        means = sub.mean(axis=0)

        # Pairwise loss differentials d_ij = mean(L_i) - mean(L_j).
        dij = means[:, None] - means[None, :]

        boot_means = np.stack([sub[boot_idx[b]].mean(axis=0) for b in range(n_boot)])
        boot_dij = boot_means[:, :, None] - boot_means[:, None, :]
        centred = boot_dij - dij[None, :, :]
        var = centred.var(axis=0, ddof=1)
        np.fill_diagonal(var, np.inf)
        se = np.sqrt(np.maximum(var, 1e-16))

        t_obs = np.abs(dij) / se
        np.fill_diagonal(t_obs, 0.0)
        T_R = float(np.max(t_obs))

        t_boot = np.abs(centred) / se[None, :, :]
        for b in range(n_boot):
            np.fill_diagonal(t_boot[b], 0.0)
        T_R_boot = t_boot.reshape(n_boot, -1).max(axis=1)

        p = float(np.mean(T_R_boot >= T_R))
        running_max = max(running_max, p)

        if p >= alpha:
            for j in alive:
                pvalues.setdefault(names[j], running_max)
            break

        # Eliminate the model with the largest standardised excess loss.
        excess = (means - means.mean()) / np.sqrt(
            np.maximum(centred.mean(axis=2).var(axis=0, ddof=1), 1e-16)
        )
        worst_local = int(np.argmax(excess))
        worst = alive[worst_local]
        eliminated_at[names[worst]] = step
        pvalues[names[worst]] = running_max
        alive.remove(worst)

    for j in alive:
        pvalues.setdefault(names[j], max(running_max, alpha))

    rows = []
    for j, name in enumerate(names):
        rows.append(
            {
                "model": name,
                "avg_loss": float(L[:, j].mean()),
                "eliminated_at": eliminated_at.get(name, np.nan),
                "mcs_pvalue": pvalues.get(name, np.nan),
                "in_mcs": name not in eliminated_at,
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("avg_loss")


# ------------------------------------------------------------------ POOS loop


@dataclass
class PoosResult:
    """Forecasts and errors from :func:`poos_experiment`."""

    forecasts: pd.DataFrame  # index = target date; columns = models, plus "actual"
    target: str
    h: int
    n_factors: int | None
    P: int

    @property
    def errors(self) -> pd.DataFrame:
        actual = self.forecasts["actual"]
        cols = [c for c in self.forecasts.columns if c != "actual"]
        return self.forecasts[cols].rsub(actual, axis=0)

    @property
    def squared_errors(self) -> pd.DataFrame:
        return self.errors**2

    def mse(self) -> pd.Series:
        return self.squared_errors.mean().sort_values()

    def relative_mse(self, benchmark: str) -> pd.Series:
        """MSE relative to ``benchmark`` -- the scale used in Tables 1 and 2.

        The paper's table entries are below 1 for models that beat a naive
        constant forecast; reporting relative to the AR benchmark is the more
        common convention and is what this returns.
        """
        m = self.mse()
        if benchmark not in m.index:
            raise KeyError(f"{benchmark!r} is not among the evaluated models.")
        return m / m[benchmark]

    def dm_against(self, benchmark: str, **kwargs) -> pd.DataFrame:
        """Diebold-Mariano test of every model against ``benchmark``."""
        err = self.errors
        if benchmark not in err.columns:
            raise KeyError(f"{benchmark!r} is not among the evaluated models.")
        rows = []
        for col in err.columns:
            if col == benchmark:
                continue
            res = dm_test(err[col].to_numpy(), err[benchmark].to_numpy(), h=self.h, **kwargs)
            rows.append(
                {
                    "model": col,
                    "mse": float(np.mean(err[col] ** 2)),
                    "dm_stat": res.statistic,
                    "p_value": res.p_value,
                    "stars": res.stars(),
                }
            )
        return pd.DataFrame(rows).set_index("model").sort_values("mse")


def poos_experiment(
    panel: pd.DataFrame,
    tcodes: pd.Series,
    model_factories: Mapping[str, Callable[[], object]],
    target: str = "INDPRO",
    h: int = 12,
    n_factors: int | None = 5,
    kx: int = 4,
    ky: int = 4,
    poos_start: str = "1990-01-01",
    poos_end: str = "2014-12-01",
    train_start: str = "1960-01-01",
    retrain_every: int = 30,
    refilter_each_origin: bool = True,
    target_kind: str | None = None,
    tune: Mapping[str, Mapping | bool] | None = None,
    tune_draws: int = 8,
    valid_months: int = 90,
    verbose: bool = True,
) -> PoosResult:
    """Run the paper's pseudo-out-of-sample forecasting exercise.

    Parameters
    ----------
    panel : DataFrame
        Raw (untransformed) FRED-MD, from :func:`tvwnn.data.fetch_fred_md`.
    tcodes : Series
        McCracken-Ng transformation codes.
    model_factories : mapping name -> callable
        Each callable returns a *fresh, unfitted* model exposing ``fit(U, y)``
        and ``forecast(u_new, gap=...)``.  A factory rather than an instance,
        because every re-estimation block needs a new model.
    target : str
    h : int
    n_factors : int or None
        ``5`` -> ``P = 24``; ``15`` -> ``P = 64``; ``None`` -> all series, no
        PCA (the paper's ``P = 424``).  Pass ``n_factors=0`` for the pure AR
        benchmark (target lags only, ``P = ky``).
    retrain_every : int, default 30
        Months between re-estimations.
    tune : mapping name -> (grid or True), or None
        Re-select hyperparameters by random search at every re-estimation
        date, on the previous ``valid_months`` months, as section 4 and
        Appendix A.3 specify.  ``True`` uses :data:`A3_GRID`.  A factory
        listed here must accept the sampled keyword arguments.

        Without this, a run uses one fixed configuration for the whole
        evaluation period, which is *not* the paper's protocol -- A.3
        re-draws at each of the ten periods.
    tune_draws : int, default 8
        Configurations tried per re-estimation date.
    valid_months : int, default 90
    refilter_each_origin : bool, default True
        Re-run the Kalman filter over all data available at each origin, with
        the network weights frozen between re-estimations.  Only affects the
        TVNN; fixed-parameter models have no state to update.  Set ``False``
        to hold the state fixed between re-estimations too.
    verbose : bool

    Returns
    -------
    PoosResult

    Notes
    -----
    Alignment.  At origin ``o`` we predict ``y_{o+h}``.  The training sample
    is every pair ``(u_t, y_{t+h})`` with ``t + h <= o`` -- i.e. ``t <= o - h``
    -- so no observation dated after ``o`` is ever touched.  The state then
    has to be carried forward ``h`` periods, which is what ``gap=h`` does.
    """
    panel = panel.loc[panel.index >= pd.Timestamp(train_start)]
    level = panel[target].copy()
    stationary = transform_panel(panel, tcodes, drop_incomplete=True)
    level = level.reindex(stationary.index)

    kind = target_kind or target_kind_from_tcode(int(tcodes[target]))
    y_future = make_target(level, h=h, kind=kind)

    if target in stationary.columns:
        target_lagged = stationary[target].to_numpy()
    else:
        target_lagged = level.diff().to_numpy()
    others = stationary.drop(columns=[target], errors="ignore")
    dates = stationary.index

    poos_start_ts = pd.Timestamp(poos_start)
    poos_end_ts = pd.Timestamp(poos_end)
    origins = dates[(dates >= poos_start_ts) & (dates <= poos_end_ts)]
    if len(origins) == 0:
        raise ValueError("The pseudo-out-of-sample window contains no dates.")

    records: list[dict] = []
    fitted: dict[str, object] = {}
    cached: dict[str, np.ndarray] = {}
    P_used = 0

    for i, origin in enumerate(origins):
        refit = (i % retrain_every == 0) or not fitted

        # ---- data available at this origin: pairs with t + h <= origin.
        last_train = origin - pd.DateOffset(months=h)
        train_mask = dates <= last_train
        if train_mask.sum() < 60:
            continue

        # ---- factors: PCA estimated on the training window only.
        if n_factors is None:
            F_all = others.to_numpy()
        elif n_factors == 0:
            F_all = None
        else:
            if refit or "proj" not in cached:
                proj = PCAProjector(n_factors=n_factors).fit(others.to_numpy()[train_mask])
                cached["proj"] = proj
            F_all = cached["proj"].transform(others.to_numpy())

        ds = build_supervised(
            factors=F_all,
            target_lagged=target_lagged,
            target_future=y_future.to_numpy(),
            dates=dates,
            h=h,
            kx=kx,
            ky=ky,
        )
        P_used = ds.U.shape[1]

        train_sel = ds.dates <= last_train
        U_tr, y_tr = ds.U[train_sel], ds.y[train_sel]
        if U_tr.shape[0] < 60:
            continue

        # ---- u_o at the origin itself (target unknown, so not in `ds`).
        origin_row = _design_row_at(
            F_all, target_lagged, dates, origin, kx=kx, ky=ky
        )
        if origin_row is None:
            continue

        actual = y_future.get(origin, np.nan)
        if not np.isfinite(actual):
            continue

        if refit:
            if verbose:
                print(
                    f"  re-estimating at {origin:%Y-%m}  "
                    f"(train n={U_tr.shape[0]}, P={P_used})"
                )
            tr_dates = ds.dates[train_sel]
            for name, factory in model_factories.items():
                params: dict = {}
                if tune and name in tune:
                    grid = tune[name]
                    grid = None if grid is True else grid
                    params = select_hyperparameters(
                        factory, U_tr, y_tr, tr_dates,
                        valid_months=valid_months, n_draws=tune_draws,
                        h=h, grid=grid, seed=hash((name, i)) % (2**32),
                    )
                    if verbose:
                        shown = {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in params.items()}
                        print(f"    {name}: {shown}")
                model = factory(**params)
                model.fit(U_tr, y_tr)
                fitted[name] = model

        row = {"origin": origin, "date": origin + pd.DateOffset(months=h), "actual": float(actual)}
        for name, model in fitted.items():
            try:
                if refilter_each_origin and hasattr(model, "smoothed_states_"):
                    pred = model.forecast(
                        origin_row, gap=h, U_hist=U_tr, y_hist=y_tr
                    )
                else:
                    pred = model.forecast(origin_row, gap=h)
                row[name] = float(np.ravel(pred)[0])
            except Exception as exc:  # keep one bad model from killing the run
                if verbose:
                    print(f"    {name} failed at {origin:%Y-%m}: {exc}")
                row[name] = np.nan
        records.append(row)

    if not records:
        raise RuntimeError(
            "No forecasts were produced; check the POOS window and sample length."
        )

    out = pd.DataFrame(records).set_index("date").drop(columns=["origin"])
    return PoosResult(
        forecasts=out, target=target, h=h, n_factors=n_factors, P=P_used
    )


def _design_row_at(
    factors: np.ndarray | None,
    target_lagged: np.ndarray,
    dates: pd.DatetimeIndex,
    origin: pd.Timestamp,
    kx: int,
    ky: int,
) -> np.ndarray | None:
    """Build the single design row ``u_o`` at a forecast origin.

    Mirrors :func:`tvwnn.data.build_lag_matrix`'s lag-major layout exactly:
    all factors at lag 0, all factors at lag 1, ..., then the target lags.
    Returns ``None`` if any required lag is missing.
    """
    pos = dates.get_indexer([origin])[0]
    if pos < 0:
        return None
    blocks: list[np.ndarray] = []
    if factors is not None:
        F = np.atleast_2d(np.asarray(factors, dtype=float))
        if pos - kx + 1 < 0:
            return None
        for lag in range(kx):
            blocks.append(F[pos - lag])
    yl = np.asarray(target_lagged, dtype=float).ravel()
    if pos - ky + 1 < 0:
        return None
    blocks.append(np.array([yl[pos - lag] for lag in range(ky)]))

    row = np.concatenate(blocks)
    return None if not np.all(np.isfinite(row)) else row.reshape(1, -1)


def mse_table(
    results: Sequence[PoosResult],
    benchmark: str = "AR",
) -> pd.DataFrame:
    """Assemble a Table 1 / Table 2 style frame from several POOS runs.

    Rows are ``(model, P)``, columns are horizons, entries are MSE relative to
    ``benchmark``, with Diebold-Mariano stars attached.
    """
    frames = []
    for res in results:
        m = res.mse()
        base = m.get(benchmark, np.nan)
        dm = res.dm_against(benchmark) if benchmark in res.forecasts.columns else None
        for model, value in m.items():
            stars = ""
            if dm is not None and model in dm.index:
                stars = dm.loc[model, "stars"]
            frames.append(
                {
                    "model": model,
                    "P": res.P,
                    "h": res.h,
                    "mse": float(value),
                    "relative_mse": float(value / base) if np.isfinite(base) else np.nan,
                    "stars": stars,
                }
            )
    return pd.DataFrame(frames)


# ------------------------------------------------- per-period hyperparameters

#: Appendix A.3's candidate grid, as Tables 3 and 4 print it.
#: Note that A.3's *prose* swaps the two scale sets relative to those tables;
#: the tables are authoritative, so "lr scale" is {0,1,2} and "L2 scale" {1..6}.
A3_GRID = {
    "d": [1, 2],
    "hidden_sizes": [(32,), (32, 16)],
    "l2_scale": [1, 2, 3, 4, 5, 6],       # l2 = 0.01 * 2**scale
    "lr_scale": [0, 1, 2],                # learning_rate = 0.01 * 2**scale
}
#: The recurrent architectures get heavier direct regularisation (A.3).
A3_GRID_RECURRENT = dict(A3_GRID, l2_scale=[4, 5, 6], hidden_sizes=[(32,)])


def sample_hyperparameters(
    rng: np.random.Generator,
    grid: Mapping[str, Sequence] = None,
) -> dict:
    """Draw one configuration from Appendix A.3's grid by random search."""
    grid = A3_GRID if grid is None else grid
    draw = {k: list(v)[int(rng.integers(len(v)))] for k, v in grid.items()}
    return {
        "d": draw["d"],
        "hidden_sizes": draw["hidden_sizes"],
        "l2": 0.01 * 2 ** draw["l2_scale"],
        "learning_rate": 0.01 * 2 ** draw["lr_scale"],
    }


def select_hyperparameters(
    make_model: Callable[..., object],
    U: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    valid_months: int = 90,
    n_draws: int = 8,
    h: int = 1,
    grid: Mapping[str, Sequence] = None,
    seed: int | None = 0,
) -> dict:
    """Random search on a held-out validation tail, as section 4 specifies.

    "Hyperparameters selected based on validation performance on the previous
    90 months of data", drawn by "random search over a grid of values", redone
    at each of the ten evaluation periods.

    Parameters
    ----------
    make_model : callable
        Takes the sampled keyword arguments and returns an unfitted model.
    U, y, dates
        The training block available at this re-estimation date.
    valid_months : int, default 90
    n_draws : int, default 8
        Configurations to try.  The paper does not say how many; 8 keeps a
        full study tractable, and more is strictly better.
    h : int
        Used only to leave an ``h``-month gap between the fitting sample and
        the validation block, so validation targets are not already known
        from the fitting data.

    Returns
    -------
    dict of the best-scoring keyword arguments.

    Notes
    -----
    This is the piece of the paper's protocol that a fixed configuration
    misses, and it is not cosmetic: A.3 chose heavy regularisation precisely
    because "a lack of similarity in distribution, at times, between the
    validation sets and POOS periods", which is a per-period judgement.
    """
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(dates)
    cutoff = dates[-1] - pd.DateOffset(months=valid_months)
    fit_mask = dates <= cutoff - pd.DateOffset(months=h)
    val_mask = dates > cutoff
    if fit_mask.sum() < 60 or val_mask.sum() < 12:
        return sample_hyperparameters(rng, grid)

    best, best_score = None, np.inf
    for _ in range(n_draws):
        params = sample_hyperparameters(rng, grid)
        try:
            model = make_model(**params)
            model.fit(U[fit_mask], y[fit_mask])
            pred = np.ravel(model.forecast(U[val_mask], gap=h))
            score = float(np.mean((y[val_mask] - pred) ** 2))
        except Exception:
            continue
        if np.isfinite(score) and score < best_score:
            best, best_score = params, score

    return best if best is not None else sample_hyperparameters(rng, grid)


__all__ += ["A3_GRID", "A3_GRID_RECURRENT", "sample_hyperparameters",
            "select_hyperparameters"]
