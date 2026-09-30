"""
Data plumbing: FRED-MD, the McCracken-Ng transformations, diffusion indices,
and the lag/horizon alignment of eq. (1).

The empirical study of Rudd, Bondell & Silver (2026) uses the FRED-MD monthly
database (McCracken & Ng 2016), stationarised with that paper's own transform
codes, compressed to principal components ("diffusion indices", Stock & Watson
2002), and lagged with ``kx = ky = 4``.

Alignment is the delicate part.  The measurement equation attaches ``y_{t+h}``
to state ``beta_t``, so a supervised pair is ``(u_t, y_{t+h})``.  At forecast
origin ``o`` only observations dated ``<= o`` exist, so the last *usable*
training pair is ``(u_{o-h}, y_o)``.  :func:`build_supervised` and
:func:`tvwnn.evaluate.poos_experiment` enforce this; getting it wrong is the
look-ahead trap described in the README.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "FRED_MD_CURRENT_URL",
    "FRED_QD_CURRENT_URL",
    "fetch_fred_qd",
    "target_kind_from_tcode",
    "TCODE_DESCRIPTIONS",
    "Dataset",
    "fetch_fred_md",
    "apply_transformation",
    "transform_panel",
    "diffusion_indices",
    "make_target",
    "build_lag_matrix",
    "build_supervised",
    "build_fred_dataset",
    "simulate_tvnn",
]

FRED_MD_CURRENT_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/monthly/current.csv"
)
FRED_MD_VINTAGE_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/monthly/{vintage}.csv"
)
FRED_QD_CURRENT_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/quarterly/current.csv"
)
FRED_QD_VINTAGE_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/quarterly/{vintage}.csv"
)

#: McCracken & Ng (2016) transformation codes.
TCODE_DESCRIPTIONS = {
    1: "level (no transformation)",
    2: "first difference",
    3: "second difference",
    4: "logarithm",
    5: "first difference of logarithm",
    6: "second difference of logarithm",
    7: "first difference of the growth rate",
}


@dataclass
class Dataset:
    """A ready-to-fit supervised dataset.

    Attributes
    ----------
    U : (n, P) ndarray
        Design matrix, row ``t`` is ``u_t`` of eq. (1).
    y : (n,) ndarray
        Target, ``y[t]`` is ``y_{t+h}``.
    dates : DatetimeIndex of length n
        The *forecast origin* of each row, i.e. the date of ``u_t``.
    target_dates : DatetimeIndex of length n
        The date of each ``y_{t+h}``.
    h : int
    feature_names : list of str
    """

    U: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex
    h: int
    feature_names: list[str]

    def __len__(self) -> int:
        return self.U.shape[0]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset(n={len(self)}, P={self.U.shape[1]}, h={self.h}, "
            f"{self.dates[0]:%Y-%m} to {self.dates[-1]:%Y-%m})"
        )


# --------------------------------------------------------------------- FRED-MD


def fetch_fred_md(
    vintage: str | None = None,
    cache_dir: str | os.PathLike[str] | None = "data",
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download (and cache) a FRED-MD vintage.

    Parameters
    ----------
    vintage : str or None
        ``"YYYY-MM"`` for a dated vintage, or ``None`` for ``current.csv``.

        The paper does **not** state which vintage it used, and FRED-MD is
        revised monthly, so its "105 series after removing variables with
        missing entries" cannot be reproduced exactly.  Pin a vintage here for
        your own reproducibility.
    cache_dir : path or None
        Where to keep the CSV.  ``None`` disables caching.
    force_download : bool

    Returns
    -------
    (data, tcodes)
        ``data`` is indexed by month with one column per series; ``tcodes`` is
        the transformation code per column.
    """
    url = FRED_MD_CURRENT_URL if vintage is None else FRED_MD_VINTAGE_URL.format(vintage=vintage)
    name = "current.csv" if vintage is None else f"{vintage}.csv"

    path: Path | None = None
    if cache_dir is not None:
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = path / name

    if path is not None and path.exists() and not force_download:
        raw = pd.read_csv(path)
    else:
        raw = pd.read_csv(url)
        if path is not None:
            raw.to_csv(path, index=False)

    date_col = raw.columns[0]
    tcode_row = raw[raw[date_col].astype(str).str.strip().str.lower().str.startswith("transform")]
    if tcode_row.empty:
        raise ValueError(
            "Could not find the 'Transform:' row in the FRED-MD file; "
            "the file layout may have changed."
        )
    tcodes = tcode_row.iloc[0].drop(labels=[date_col]).astype(float).astype(int)

    body = raw.drop(index=tcode_row.index).copy()
    body = body[body[date_col].notna()]
    body[date_col] = pd.to_datetime(body[date_col], format="mixed")
    body = body.set_index(date_col).sort_index()
    body.index.name = "date"
    body = body.apply(pd.to_numeric, errors="coerce")
    return body, tcodes


def fetch_fred_qd(
    vintage: str | None = None,
    cache_dir: str | os.PathLike[str] | None = "data",
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download (and cache) a FRED-QD vintage -- the quarterly companion database.

    FRED-QD (McCracken & Ng 2020) carries ~245 quarterly series, including the
    national-accounts aggregates that have no monthly counterpart (real GDP,
    consumption, investment, the GDP deflator).  The file layout differs from
    FRED-MD: it carries **two** metadata rows, ``factors`` and ``transform``,
    rather than one.

    Everything downstream is frequency-agnostic -- the same transformation
    codes, the same ``kx``/``ky`` lag construction, the same ``h``-step
    alignment -- so a quarterly study is the monthly one with ``h`` read as
    quarters.  Note that ``h = 4`` quarters is the same *economic* horizon as
    ``h = 12`` months but only a third of the overlap, which is exactly the
    contrast :mod:`examples.other_targets_study` is built to exploit.

    Returns
    -------
    (data, tcodes)
        Same shape as :func:`fetch_fred_md`.
    """
    url = FRED_QD_CURRENT_URL if vintage is None else FRED_QD_VINTAGE_URL.format(vintage=vintage)
    name = "qd_current.csv" if vintage is None else f"qd_{vintage}.csv"

    path: Path | None = None
    if cache_dir is not None:
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = path / name

    if path is not None and path.exists() and not force_download:
        raw = pd.read_csv(path)
    else:
        raw = pd.read_csv(url)
        if path is not None:
            raw.to_csv(path, index=False)

    date_col = raw.columns[0]
    tag = raw[date_col].astype(str).str.strip().str.lower()
    tcode_row = raw[tag.str.startswith("transform")]
    if tcode_row.empty:
        raise ValueError(
            "Could not find the 'transform' row in the FRED-QD file; "
            "the file layout may have changed."
        )
    tcodes = tcode_row.iloc[0].drop(labels=[date_col]).astype(float).astype(int)

    meta = raw[tag.str.startswith(("transform", "factors"))].index
    body = raw.drop(index=meta).copy()
    body = body[body[date_col].notna()]
    body[date_col] = pd.to_datetime(body[date_col], format="mixed")
    body = body.set_index(date_col).sort_index()
    body.index.name = "date"
    body = body.apply(pd.to_numeric, errors="coerce")
    return body, tcodes


def apply_transformation(series: pd.Series, tcode: int) -> pd.Series:
    """Apply one McCracken-Ng transformation code."""
    x = series.astype(float)
    if tcode == 1:
        return x
    if tcode == 2:
        return x.diff()
    if tcode == 3:
        return x.diff().diff()
    if tcode == 4:
        return np.log(x)
    if tcode == 5:
        return np.log(x).diff()
    if tcode == 6:
        return np.log(x).diff().diff()
    if tcode == 7:
        return (x / x.shift(1) - 1.0).diff()
    raise ValueError(f"Unknown transformation code {tcode!r}; expected 1-7.")


def transform_panel(
    data: pd.DataFrame,
    tcodes: pd.Series,
    drop_incomplete: bool = True,
    trim_ragged_edges: bool = True,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Stationarise every column, then drop columns with any missing entry.

    The paper keeps "105 series after removing variables with missing
    entries"; ``drop_incomplete`` reproduces that rule.

    Parameters
    ----------
    trim_ragged_edges : bool, default True
        Drop leading and trailing rows where fewer than ``min_coverage`` of
        the series are observed, *before* dropping columns.

        This matters more than it looks.  FRED-MD's most recent one or two
        months are ragged -- the latest vintage at the time of writing has
        only 47 of 126 series filled in for its final month -- so applying
        ``dropna(axis=1)`` to the raw file would throw away three quarters of
        the database, INDPRO and UNRATE included, on the strength of a few
        unreleased observations.  Trimming first recovers ~121 complete series
        over 1960-2014, against the paper's 105 on its own (unstated) vintage.
    min_coverage : float, default 0.95
        Fraction of series that must be present for a row to be kept.
    """
    out = {}
    for col in data.columns:
        if col not in tcodes.index:
            continue
        out[col] = apply_transformation(data[col], int(tcodes[col]))
    frame = pd.DataFrame(out, index=data.index)
    # The transforms burn up to two leading observations.
    frame = frame.iloc[2:]

    if trim_ragged_edges and frame.shape[1] > 0:
        coverage = frame.notna().mean(axis=1).to_numpy()
        good = np.flatnonzero(coverage >= min_coverage)
        if good.size == 0:
            raise ValueError(
                f"No row reaches {min_coverage:.0%} coverage; lower min_coverage."
            )
        frame = frame.iloc[good[0] : good[-1] + 1]

    if drop_incomplete:
        frame = frame.dropna(axis=1, how="any")
    return frame


def diffusion_indices(
    X: np.ndarray,
    n_factors: int,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Principal components of the stationarised panel (Stock & Watson 2002).

    Returns ``(factors, explained_variance_ratio)``.  Computed by SVD on the
    standardised panel, with signs fixed so the largest-magnitude loading of
    each component is positive -- otherwise the sign is arbitrary and plots
    flip between runs.

    Notes
    -----
    In a pseudo-out-of-sample exercise the PCA must be estimated on the
    training window only; :func:`tvwnn.evaluate.poos_experiment` does that.
    Fitting it once on the full sample leaks information.
    """
    X = np.asarray(X, dtype=float)
    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xs = (X - mu) / sd
    else:
        Xs = X - X.mean(axis=0)

    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    k = min(n_factors, Vt.shape[0])
    factors = U[:, :k] * S[:k]
    loadings = Vt[:k]
    for j in range(k):
        if loadings[j][np.argmax(np.abs(loadings[j]))] < 0:
            factors[:, j] *= -1.0
    total = float(np.sum(S**2))
    evr = (S[:k] ** 2) / total if total > 0 else np.zeros(k)
    return factors, evr


# ------------------------------------------------------------------- targets


def target_kind_from_tcode(tcode: int) -> str:
    """Pick the forecast-target transform implied by a McCracken-Ng code.

    Section 4 of the paper defines two targets: for UNRATE "the aggregate h
    month change" ``y_{t+h} - y_t``, and for INDPRO "the log hth difference"
    ``log y_{t+h} - log y_t``.  It gives no rule for anything else.

    The rule this implements is the obvious generalisation: series the
    McCracken-Ng codes treat in logs (4, 5, 6 -- levels of real activity,
    prices, money) get the log difference; series treated in levels or simple
    differences (1, 2, 3 -- rates and spreads, already in percentage points)
    get the plain difference.  That reproduces the paper's two choices exactly
    -- INDPRO is code 5, UNRATE is code 2 -- and extends them sensibly to any
    other target.

    Pass ``kind=`` explicitly to override.
    """
    if tcode in (4, 5, 6):
        return "logdiff"
    if tcode in (1, 2, 3, 7):
        return "diff"
    raise ValueError(f"Unknown transformation code {tcode!r}; expected 1-7.")


def make_target(level: pd.Series, h: int, kind: str) -> pd.Series:
    """Build the forecast target of section 4 of the paper.

    Parameters
    ----------
    level : Series
        The *untransformed* target series (e.g. raw INDPRO or UNRATE).
    h : int
        Forecast horizon in months.
    kind : {"diff", "logdiff"}
        ``"diff"``   -> ``y_{t+h} - y_t``      (UNRATE: "the aggregate h month change")
        ``"logdiff"``-> ``log y_{t+h} - log y_t`` (INDPRO: "the log hth difference")

    Returns
    -------
    Series indexed by the *forecast origin* ``t``, so ``out[t]`` is the value
    to be predicted from ``u_t``.  The last ``h`` entries are ``NaN``.
    """
    x = level.astype(float)
    if kind == "diff":
        fut = x.shift(-h) - x
    elif kind == "logdiff":
        fut = np.log(x).shift(-h) - np.log(x)
    else:
        raise ValueError("kind must be 'diff' or 'logdiff'.")
    fut.name = f"{level.name}_h{h}_{kind}"
    return fut


def build_lag_matrix(
    values: np.ndarray,
    n_lags: int,
    prefix: str = "x",
) -> tuple[np.ndarray, list[str]]:
    """Stack ``n_lags`` lags, most recent first.

    Row ``t`` of the result is ``[v_t, v_{t-1}, ..., v_{t-n_lags+1}]`` flattened
    across columns.  The first ``n_lags - 1`` rows are ``NaN``.

    This is the ``x'_{t-kx+1:t}`` block of eq. (1).  Note that the paper prints
    the range as ``t-kx:t``, which would be ``kx + 1`` lags and contradicts its
    own ``P = p*kx + ky``; the empirical ``P = 24 = 4 + 5*4`` confirms that
    exactly ``kx`` lags are intended.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape[0] == 1 and values.ndim == 2 and values.shape[1] > 1:
        pass
    n, k = values.shape
    out = np.full((n, k * n_lags), np.nan)
    names: list[str] = []
    for lag in range(n_lags):
        block = np.full((n, k), np.nan)
        if lag == 0:
            block = values
        else:
            block[lag:] = values[:-lag]
        out[:, lag * k : (lag + 1) * k] = block
        names.extend(f"{prefix}{j + 1}_L{lag}" for j in range(k))
    return out, names


def build_supervised(
    factors: np.ndarray | None,
    target_lagged: np.ndarray,
    target_future: np.ndarray,
    dates: pd.DatetimeIndex,
    h: int,
    kx: int = 4,
    ky: int = 4,
) -> Dataset:
    """Assemble ``(u_t, y_{t+h})`` pairs.

    Parameters
    ----------
    factors : (n, k) array or None
        Stationarised predictors -- principal components, or the full panel.
        ``None`` gives a pure autoregression (the paper's AR benchmark).
    target_lagged : (n,) array
        The *stationarised* target series, used for its own lags inside
        ``u_t``.
    target_future : (n,) array
        The forecast target from :func:`make_target`, aligned to the origin.
    dates : DatetimeIndex of length n
    h, kx, ky : int

    Returns
    -------
    Dataset
        Rows containing any ``NaN`` -- from the lag burn-in or from the final
        ``h`` observations where the target is unknown -- are dropped.
    """
    target_lagged = np.asarray(target_lagged, dtype=float).reshape(-1, 1)
    n = target_lagged.shape[0]

    blocks: list[np.ndarray] = []
    names: list[str] = []
    if factors is not None:
        F = np.atleast_2d(np.asarray(factors, dtype=float))
        if F.shape[0] != n:
            raise ValueError("factors and target_lagged must have the same length.")
        Fl, fnames = build_lag_matrix(F, kx, prefix="f")
        blocks.append(Fl)
        names.extend(fnames)
    Yl, ynames = build_lag_matrix(target_lagged, ky, prefix="y")
    blocks.append(Yl)
    names.extend(ynames)

    U = np.hstack(blocks)
    y = np.asarray(target_future, dtype=float).ravel()
    if y.shape[0] != n:
        raise ValueError("target_future must have the same length as target_lagged.")

    ok = np.isfinite(U).all(axis=1) & np.isfinite(y)
    idx = np.flatnonzero(ok)
    dates = pd.DatetimeIndex(dates)
    origin = dates[idx]
    tgt_dates = origin + pd.DateOffset(months=h)

    return Dataset(
        U=U[idx],
        y=y[idx],
        dates=origin,
        target_dates=pd.DatetimeIndex(tgt_dates),
        h=h,
        feature_names=names,
    )


def build_fred_dataset(
    target: str = "INDPRO",
    h: int = 12,
    n_factors: int | None = 5,
    kx: int = 4,
    ky: int = 4,
    start: str | None = "1960-01-01",
    end: str | None = None,
    vintage: str | None = None,
    cache_dir: str | os.PathLike[str] | None = "data",
    data: pd.DataFrame | None = None,
    tcodes: pd.Series | None = None,
    target_kind: str | None = None,
) -> Dataset:
    """One-call FRED-MD pipeline reproducing the paper's predictor sets.

    Parameters
    ----------
    target : str
        Any FRED-MD column.  The forecast transform defaults to
        :func:`target_kind_from_tcode`, which reproduces the paper's own two
        choices (INDPRO -> log ``h``-th difference, UNRATE -> aggregate
        ``h``-month change) and generalises them.  Override with
        ``target_kind``.
    target_kind : {"diff", "logdiff"} or None
    n_factors : int or None
        ``5`` reproduces ``P = 24`` (4 target lags + 4 lags x 5 PCs).
        ``15`` reproduces ``P = 64``.
        ``None`` uses **all** stationarised series without PCA, reproducing the
        paper's high-dimensional ``P = 424`` setting.
    data, tcodes
        Pre-loaded FRED-MD, to avoid re-downloading.

    Warnings
    --------
    The PCA here is estimated on the whole sample, which is fine for a
    demonstration but leaks information in a forecasting evaluation.  Use
    :func:`tvwnn.evaluate.poos_experiment`, which re-estimates it inside each
    training window.
    """
    if data is None or tcodes is None:
        data, tcodes = fetch_fred_md(vintage=vintage, cache_dir=cache_dir)
    if target not in data.columns:
        raise ValueError(f"{target!r} is not a column of this FRED-MD vintage.")

    if start is not None:
        data = data.loc[data.index >= pd.Timestamp(start)]
    if end is not None:
        data = data.loc[data.index <= pd.Timestamp(end)]

    level = data[target].copy()
    panel = transform_panel(data, tcodes, drop_incomplete=True)
    level = level.reindex(panel.index)

    kind = target_kind if target_kind else target_kind_from_tcode(int(tcodes[target]))
    y_future = make_target(level, h=h, kind=kind)

    target_lagged = panel[target].to_numpy() if target in panel.columns else level.diff().to_numpy()
    others = panel.drop(columns=[target], errors="ignore")

    if n_factors is None:
        factors = others.to_numpy()
    else:
        factors, _ = diffusion_indices(others.to_numpy(), n_factors=n_factors)

    return build_supervised(
        factors=factors,
        target_lagged=target_lagged,
        target_future=y_future.to_numpy(),
        dates=panel.index,
        h=h,
        kx=kx,
        ky=ky,
    )


# ------------------------------------------------------------------ simulation


def simulate_tvnn(
    n: int = 400,
    p: int = 3,
    h: int = 1,
    d_true: int = 1,
    sigma: float = 0.3,
    omega: float = 0.02,
    seed: int | None = 0,
    nonlinear: bool = True,
) -> tuple[Dataset, dict]:
    """Simulate from a known-truth TVNN-like DGP.

    A nonlinear basis with genuinely drifting coefficients:

        z_t = tanh(W u_t)          (fixed random W, d_true columns)
        beta_t = beta_{t-1} + xi_t,  xi_t ~ N(0, omega * I)
        y_{t+h} = z_t' beta_t + eps_t

    Used by the test suite and the quickstart.  What should be recovered is the
    *forecasting behaviour*: with genuinely drifting coefficients a TVNN should
    beat an otherwise identical fixed-weight network by a wide, statistically
    significant margin.

    What should **not** be compared to ``truth`` directly is ``sigma2`` or
    ``Omega``.  The model standardises ``y`` internally, so both are on a
    rescaled axis; and the network learns its own basis, which is related to
    the simulated ``z`` only up to an invertible linear map, so ``Omega_hat``
    lives in different units by construction.  See "Identification" in the
    README.  To check the variance components against truth, feed a *fixed*
    design straight to :func:`tvwnn.em.run_em` instead -- that is what
    ``tests/test_em.py`` does.

    Returns
    -------
    (dataset, truth)
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    for j in range(p):  # give the predictors some persistence
        for t in range(1, n):
            x[t, j] = 0.6 * x[t - 1, j] + 0.8 * x[t, j]

    W = rng.normal(scale=1.0, size=(p, d_true))
    z = np.tanh(x @ W) if nonlinear else x @ W

    beta = np.zeros((n, d_true))
    beta[0] = rng.normal(scale=1.0, size=d_true)
    for t in range(1, n):
        beta[t] = beta[t - 1] + rng.normal(scale=np.sqrt(omega), size=d_true)

    signal = np.einsum("td,td->t", z, beta)
    y_future = signal + rng.normal(scale=sigma, size=n)

    dates = pd.date_range("1970-01-01", periods=n, freq="MS")
    ds = build_supervised(
        factors=x,
        target_lagged=np.r_[0.0, np.diff(signal)],
        target_future=y_future,
        dates=dates,
        h=h,
        kx=1,
        ky=1,
    )
    truth = {
        "beta": beta,
        "z": z,
        "sigma2": sigma**2,
        "omega": omega,
        "W": W,
        "signal": signal,
    }
    return ds, truth


class PCAProjector:
    """Fit principal components on a training window and project later data.

    In a pseudo-out-of-sample exercise the diffusion indices must be estimated
    on the training window only and then *applied* to the evaluation period.
    Re-fitting the PCA on the full sample leaks information about the future
    into every forecast.

    Examples
    --------
    >>> proj = PCAProjector(n_factors=5).fit(X_train)
    >>> F_train = proj.transform(X_train)
    >>> F_test = proj.transform(X_test)
    """

    def __init__(self, n_factors: int) -> None:
        self.n_factors = int(n_factors)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.loadings_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAProjector":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-12] = 1.0
        self.std_ = sd
        Xs = (X - self.mean_) / self.std_
        _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        k = min(self.n_factors, Vt.shape[0])
        loadings = Vt[:k].copy()
        for j in range(k):
            if loadings[j][np.argmax(np.abs(loadings[j]))] < 0:
                loadings[j] *= -1.0
        self.loadings_ = loadings
        total = float(np.sum(S**2))
        self.explained_variance_ratio_ = (S[:k] ** 2) / total if total > 0 else np.zeros(k)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.loadings_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return ((X - self.mean_) / self.std_) @ self.loadings_.T


__all__.append("PCAProjector")
