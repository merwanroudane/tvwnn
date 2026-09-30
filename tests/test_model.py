"""
Tests for the TVNN estimator, the alignment rules, and the evaluation tools.

The alignment tests are the ones that matter most: they are what stops the
library from quietly giving itself a look-ahead advantage, which is the single
failure mode that would make every number it prints look good and be wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tvwnn import TVNN, TVNNEnsemble, dm_test, simulate_tvnn
from tvwnn.baselines import (
    ARBenchmark,
    FeedForwardForecaster,
    RecurrentForecaster,
    RidgeARX,
    sequence_view,
)
from tvwnn.data import build_lag_matrix, build_supervised, make_target
from tvwnn.evaluate import model_confidence_set


# ------------------------------------------------------------------ alignment


def test_make_target_is_forward_looking_and_leaves_no_leak():
    x = pd.Series(np.arange(10.0), index=pd.date_range("2000-01-01", periods=10, freq="MS"))
    out = make_target(x, h=3, kind="diff")
    assert np.isnan(out.iloc[-3:]).all()          # last h are unknowable
    assert out.iloc[0] == pytest.approx(3.0)      # x[3] - x[0]


def test_build_lag_matrix_orders_most_recent_first():
    v = np.arange(6.0).reshape(6, 1)
    L, names = build_lag_matrix(v, n_lags=3, prefix="y")
    assert np.isnan(L[:2]).any()
    np.testing.assert_allclose(L[3], [3.0, 2.0, 1.0])
    assert names == ["y1_L0", "y1_L1", "y1_L2"]


def test_supervised_pair_uses_only_past_predictors():
    """u_t must contain nothing dated after t, and y[t] must be y_{t+h}."""
    n, h = 40, 4
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    level = pd.Series(np.arange(n, dtype=float), index=dates, name="lvl")
    fut = make_target(level, h=h, kind="diff")
    factors = np.arange(n, dtype=float).reshape(n, 1)

    ds = build_supervised(factors, level.diff().to_numpy(), fut.to_numpy(),
                          dates, h=h, kx=2, ky=2)
    # column 0 is factor lag 0 == the origin's own value
    for i, origin in enumerate(ds.dates):
        assert ds.U[i, 0] == pytest.approx(float(dates.get_loc(origin)))
        assert ds.target_dates[i] == origin + pd.DateOffset(months=h)
        assert ds.y[i] == pytest.approx(h)  # a unit-slope series


def test_forecast_gap_reproduces_random_walk_inflation():
    """gap=h must widen the predictive variance by exactly h * Omega."""
    ds, _ = simulate_tvnn(n=150, seed=1)
    m = TVNN(d=1, n_outer=3, grad_steps=3, random_state=0).fit(ds.U, ds.y)
    _, f0 = m.forecast(ds.U[-1:], gap=0, return_variance=True)
    _, f6 = m.forecast(ds.U[-1:], gap=6, return_variance=True)
    z, _ = m._basis(ds.U[-1:])
    expected = 6.0 * float((z @ m.theta_.Omega @ z.T).item()) * (m._y_std**2)
    assert float(np.ravel(f6 - f0)[0]) == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------- model


def test_tvnn_beats_a_fixed_weight_network_on_drifting_data():
    """The paper's central claim, on a DGP where it must hold."""
    ds, _ = simulate_tvnn(n=420, seed=0)
    ntr = 350
    Utr, ytr, Ute, yte = ds.U[:ntr], ds.y[:ntr], ds.U[ntr:], ds.y[ntr:]

    tv = TVNN(d=1, n_outer=25, grad_steps=10, l2=0.02, random_state=0).fit(Utr, ytr)
    preds = np.array(
        [
            tv.forecast(
                Ute[i : i + 1], gap=1,
                U_hist=np.vstack([Utr, Ute[:i]]), y_hist=np.r_[ytr, yte[:i]],
            )[0]
            for i in range(len(yte))
        ]
    )
    ff = FeedForwardForecaster(
        hidden_sizes=(32,), epochs=250, l2=0.02, random_state=0
    ).fit(Utr, ytr)

    e_tv, e_ff = yte - preds, yte - ff.forecast(Ute)
    assert np.mean(e_tv**2) < np.mean(e_ff**2)
    assert dm_test(e_tv, e_ff, h=1).p_value < 0.05


def test_both_loss_variants_run_and_differ():
    """weighted_sse (as printed) and exact (with log F) are both available."""
    ds, _ = simulate_tvnn(n=200, seed=2)
    kw = dict(d=1, n_outer=5, grad_steps=5, random_state=0)
    a = TVNN(loss="weighted_sse", **kw).fit(ds.U, ds.y).summary()
    b = TVNN(loss="exact", **kw).fit(ds.U, ds.y).summary()
    assert np.isfinite(a["sigma2"]) and np.isfinite(b["sigma2"])
    assert a["sigma2"] != b["sigma2"]


def test_both_state_variants_run():
    ds, _ = simulate_tvnn(n=200, seed=2)
    kw = dict(d=1, n_outer=5, grad_steps=5, random_state=0)
    for state in ("predicted", "filtered"):
        m = TVNN(state=state, **kw).fit(ds.U, ds.y)
        assert np.isfinite(m.summary()["sigma2"])


def test_fixed_effect_head_changes_the_fit():
    ds, _ = simulate_tvnn(n=200, seed=4)
    kw = dict(d=1, n_outer=5, grad_steps=5, random_state=0)
    plain = TVNN(**kw).fit(ds.U, ds.y)
    fe = TVNN(fixed_effect=True, **kw).fit(ds.U, ds.y)
    assert fe.net_.has_fixed_effect and not plain.net_.has_fixed_effect
    assert np.isfinite(fe.summary()["sigma2"])


def test_ensemble_averages_members():
    ds, _ = simulate_tvnn(n=180, seed=5)
    ens = TVNNEnsemble(
        n_members=3, d=1, n_outer=4, grad_steps=4, random_state=0
    ).fit(ds.U, ds.y)
    pred = ens.forecast(ds.U[-1:], gap=1)
    members = np.array([m.forecast(ds.U[-1:], gap=1)[0] for m in ens.members_])
    assert pred[0] == pytest.approx(members.mean())
    assert ens.forecast_spread(ds.U[-1:], gap=1)[0] > 0


def test_reproducible_under_a_seed():
    ds, _ = simulate_tvnn(n=150, seed=6)
    kw = dict(d=1, n_outer=4, grad_steps=4, random_state=42)
    a = TVNN(**kw).fit(ds.U, ds.y).forecast(ds.U[-1:], gap=1)
    b = TVNN(**kw).fit(ds.U, ds.y).forecast(ds.U[-1:], gap=1)
    np.testing.assert_allclose(a, b)


def test_rejects_misaligned_input():
    ds, _ = simulate_tvnn(n=100, seed=7)
    with pytest.raises(ValueError):
        TVNN(n_outer=1, grad_steps=1).fit(ds.U, ds.y[:-1])


# ------------------------------------------------------------------ baselines


def test_baselines_all_fit_and_forecast():
    ds, _ = simulate_tvnn(n=200, p=3, seed=8)
    for model in (
        ARBenchmark(),
        RidgeARX(),
        FeedForwardForecaster(epochs=30, random_state=0),
        RecurrentForecaster(cell="rnn", kx=1, ky=1, n_factors=3, epochs=30, random_state=0),
        RecurrentForecaster(cell="lstm", kx=1, ky=1, n_factors=3, epochs=30, random_state=0),
    ):
        model.fit(ds.U, ds.y)
        out = np.ravel(model.forecast(ds.U[-1:]))
        assert out.shape == (1,) and np.isfinite(out[0])


def test_sequence_view_is_oldest_first():
    kx = ky = 2
    k = 2
    # u = [f1_L0, f2_L0, f1_L1, f2_L1, y_L0, y_L1]
    U = np.array([[10.0, 20.0, 11.0, 21.0, 30.0, 31.0]])
    seq = sequence_view(U, kx, ky, n_factors=k)
    assert seq.shape == (1, 2, 3)
    np.testing.assert_allclose(seq[0, 0], [11.0, 21.0, 31.0])  # lag 1, older
    np.testing.assert_allclose(seq[0, 1], [10.0, 20.0, 30.0])  # lag 0, newest


# ----------------------------------------------------------------- evaluation


def test_dm_sign_convention_and_stars():
    rng = np.random.default_rng(0)
    good = rng.normal(scale=0.5, size=250)
    bad = rng.normal(scale=2.0, size=250)
    res = dm_test(good, bad, h=1)
    assert res.statistic < 0          # negative favours the first argument
    assert res.p_value < 0.01
    assert res.stars() == "***"
    assert dm_test(good, good.copy(), h=1).p_value > 0.99


def test_dm_harvey_adjustment_is_conservative():
    rng = np.random.default_rng(1)
    e1 = rng.normal(scale=1.0, size=80)
    e2 = rng.normal(scale=1.2, size=80)
    plain = dm_test(e1, e2, h=6, harvey_adjustment=False)
    adj = dm_test(e1, e2, h=6, harvey_adjustment=True)
    assert abs(adj.statistic) <= abs(plain.statistic) + 1e-12
    assert adj.p_value >= plain.p_value


def test_model_confidence_set_keeps_the_best_model():
    rng = np.random.default_rng(2)
    n = 400
    losses = pd.DataFrame(
        {
            "best": rng.chisquare(1, n) * 0.5,
            "mid": rng.chisquare(1, n) * 1.0,
            "worst": rng.chisquare(1, n) * 4.0,
        }
    )
    out = model_confidence_set(losses, alpha=0.10, n_boot=200, seed=0)
    assert out.loc["best", "in_mcs"]
    assert not out.loc["worst", "in_mcs"]
