"""
Reproduce the empirical study of Rudd, Bondell & Silver (2026) on real data.

Forecasts US industrial production (INDPRO) and the unemployment rate
(UNRATE) from the FRED-MD database, over a pseudo-out-of-sample period, with
the paper's six competitors, and writes Table 1 / Table 2 and Figures 1 / 2.

Usage
-----
Quick pass -- one target, one horizon, small predictor set, ~2 minutes::

    python examples/run_empirical_study.py --quick

Default -- both targets, four horizons, P in {24, 64}, single networks::

    python examples/run_empirical_study.py

The paper's full configuration -- add P = 424 and 10-network ensembles.
Expect *hours*::

    python examples/run_empirical_study.py --predictor-sets 24 64 424 --ensemble 10

Outputs land in ``tables/`` and ``figures/``.

Note on reproducibility
-----------------------
The paper does not state which FRED-MD vintage it used, and the database is
revised monthly, so its "105 series after removing variables with missing
entries" is not reproducible.  Pass ``--vintage 2015-01`` to pin one.  With a
recent vintage you will get ~115 complete series over 1960-2014, so the
high-dimensional setting is ``P = 460`` rather than the paper's 424; the
arithmetic (``4 target lags + 4 lags x n series``) is identical.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tvwnn import (
    A3_GRID,
    A3_GRID_RECURRENT,
    ARBenchmark,
    FeedForwardForecaster,
    RecurrentForecaster,
    RidgeARX,
    TVNN,
    TVNNEnsemble,
    dm_matrix,
    fetch_fred_md,
    forecast_accuracy_table,
    model_confidence_set,
    mse_table,
    plot_cssed,
    plot_forecast_vs_actual,
    plot_mse_by_horizon,
    plot_time_varying_weights,
    poos_experiment,
    render_console,
    render_latex,
    save_figure,
)

TABLES = Path("tables")
FIGURES = Path("figures")

# Appendix A.3: the TVNN most commonly selected L2 scale 5 (0.32) for INDPRO
# and 1 (0.02) for UNRATE, learning-rate scale 0 (0.01) or 2 (0.04), d in {1,2},
# one hidden layer of 32 units.
DEFAULTS = {
    "INDPRO": dict(d=1, hidden_sizes=(32,), l2=0.32, learning_rate=0.01),
    "UNRATE": dict(d=2, hidden_sizes=(32,), l2=0.02, learning_rate=0.01),
}


def build_model_factories(
    target: str,
    n_factors: int | None,
    kx: int,
    ky: int,
    n_series: int,
    ensemble: int,
    seed: int,
) -> dict:
    """One factory per row of Tables 1 and 2.

    A *factory* rather than an instance, because the evaluation engine builds a
    fresh model at every re-estimation date.
    """
    cfg = DEFAULTS.get(target.upper(), DEFAULTS["INDPRO"])
    k = n_series if n_factors is None else n_factors

    # Factories accept **params so `poos_experiment(tune=...)` can inject a
    # configuration drawn by random search at each re-estimation date; with no
    # tuning they fall back to the A.3 modal values in DEFAULTS.
    def make_tvnn(**params):
        kwargs = {**cfg, **params, "n_outer": 30, "grad_steps": 10,
                  "random_state": seed}
        if ensemble > 1:
            return TVNNEnsemble(n_members=ensemble, **kwargs)
        return TVNN(**kwargs)

    def make_ffnn(**params):
        q = {**cfg, **params}
        return FeedForwardForecaster(
            hidden_sizes=q["hidden_sizes"], epochs=300,
            learning_rate=q["learning_rate"], l2=q["l2"], random_state=seed,
        )

    def make_rnn(cell, **params):
        q = {**cfg, "l2": 0.64, **params}
        return RecurrentForecaster(
            cell=cell, kx=kx, ky=ky, n_factors=k, epochs=300,
            learning_rate=q["learning_rate"], l2=q["l2"], random_state=seed,
        )

    return {
        "ARX": lambda **_: RidgeARX(),
        "FFNN": make_ffnn,
        "RNN": lambda **p: make_rnn("rnn", **p),
        "LSTM": lambda **p: make_rnn("lstm", **p),
        "TVNN": make_tvnn,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=["INDPRO", "UNRATE"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 9, 12])
    ap.add_argument(
        "--predictor-sets", nargs="+", type=int, default=[24, 64],
        help="P values. 24 -> 5 PCs, 64 -> 15 PCs, 424 -> all series (no PCA).",
    )
    ap.add_argument("--ensemble", type=int, default=1,
                    help="TVNN ensemble members; the paper uses 10.")
    ap.add_argument("--poos-start", default="1990-01-01")
    ap.add_argument("--poos-end", default="2014-12-01")
    ap.add_argument("--train-start", default="1960-01-01")
    ap.add_argument("--retrain-every", type=int, default=30)
    ap.add_argument("--kx", type=int, default=4)
    ap.add_argument("--ky", type=int, default=4)
    ap.add_argument("--vintage", default=None, help="FRED-MD vintage, e.g. 2015-01.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune", action="store_true",
                    help="Re-select hyperparameters by random search over the "
                         "Appendix A.3 grid at every re-estimation date, on the "
                         "previous 90 months. This is the paper's protocol; "
                         "without it one fixed configuration is used throughout.")
    ap.add_argument("--tune-draws", type=int, default=8)
    ap.add_argument("--quick", action="store_true",
                    help="One target, h=12, P=24, POOS 2005-2014.")
    args = ap.parse_args()

    if args.quick:
        args.targets = ["UNRATE"]
        args.horizons = [12]
        args.predictor_sets = [24]
        args.poos_start = "2005-01-01"
        args.retrain_every = 60

    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    print("Loading FRED-MD ...")
    panel, tcodes = fetch_fred_md(vintage=args.vintage, cache_dir="data")
    print(f"  {panel.shape[1]} series, {panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}")

    # How many series survive stationarisation, for the P = 424 setting.
    from tvwnn.data import transform_panel

    window = panel.loc[args.train_start : args.poos_end]
    n_series = transform_panel(window, tcodes).shape[1]
    print(f"  {n_series} complete series after McCracken-Ng transformation")

    p_to_factors = {24: 5, 64: 15}
    started = time.time()

    for target in args.targets:
        all_results = []
        per_model_best: dict[str, dict[int, float]] = {}
        keep_for_plots = None

        for P in args.predictor_sets:
            n_factors = p_to_factors.get(P)  # None -> all series, no PCA
            factories = build_model_factories(
                target, n_factors, args.kx, args.ky, n_series - 1,
                args.ensemble, args.seed,
            )
            factories = {"AR": lambda **_: ARBenchmark(), **factories}

            tune_spec = None
            if args.tune:
                tune_spec = {n: (A3_GRID_RECURRENT if n in ("RNN", "LSTM")
                                 else A3_GRID)
                             for n in factories if n != "AR"}

            for h in args.horizons:
                label = f"{target}  P={P}  h={h}"
                print(f"\n=== {label} " + "=" * max(0, 46 - len(label)))

                # The AR benchmark uses target lags only, so it gets its own run.
                ar_res = poos_experiment(
                    panel, tcodes, {"AR": lambda **_: ARBenchmark()},
                    target=target, h=h, n_factors=0, kx=args.kx, ky=args.ky,
                    poos_start=args.poos_start, poos_end=args.poos_end,
                    train_start=args.train_start, retrain_every=args.retrain_every,
                    verbose=False,
                )
                rest = {k: v for k, v in factories.items() if k != "AR"}
                res = poos_experiment(
                    panel, tcodes, rest,
                    target=target, h=h, n_factors=n_factors, kx=args.kx, ky=args.ky,
                    poos_start=args.poos_start, poos_end=args.poos_end,
                    train_start=args.train_start, retrain_every=args.retrain_every,
                    tune=tune_spec, tune_draws=args.tune_draws, verbose=True,
                )
                res.forecasts["AR"] = ar_res.forecasts["AR"]

                print(res.dm_against("AR").to_string(float_format=lambda v: f"{v:8.4f}"))
                all_results.append(res)

                for model, value in res.mse().items():
                    slot = per_model_best.setdefault(model, {})
                    slot[h] = min(slot.get(h, np.inf), float(value))

                if keep_for_plots is None or (P == args.predictor_sets[0]
                                              and h == max(args.horizons)):
                    keep_for_plots = res

        # ---------------------------------------------------------- tables
        long = mse_table(all_results, benchmark="AR")
        table = forecast_accuracy_table(long, value="relative_mse", benchmark="AR")
        title = (
            f"POOS MSE relative to the AR benchmark: {target}, "
            f"{args.poos_start[:7]} to {args.poos_end[:7]}"
        )
        console = render_console(table, title=title, bold_min_within="P")
        print("\n" + console)

        (TABLES / f"table_{target.lower()}.txt").write_text(console, encoding="utf-8")
        (TABLES / f"table_{target.lower()}.tex").write_text(
            render_latex(table, caption=title, label=f"tab:{target.lower()}"),
            encoding="utf-8",
        )
        long.to_csv(TABLES / f"results_{target.lower()}.csv", index=False)

        # The comparison the paper never reports: every pairwise DM test.
        if keep_for_plots is not None:
            pm = dm_matrix(keep_for_plots.errors, h=keep_for_plots.h)
            pm.to_csv(TABLES / f"dm_pairwise_{target.lower()}.csv")
            print(f"\nPairwise DM p-values (h={keep_for_plots.h}, P={keep_for_plots.P}):")
            print(pm.to_string(float_format=lambda v: f"{v:6.3f}"))

            mcs = model_confidence_set(
                keep_for_plots.squared_errors, alpha=0.10, n_boot=500, seed=args.seed
            )
            mcs.to_csv(TABLES / f"mcs_{target.lower()}.csv")
            print("\nModel Confidence Set (alpha = 0.10):")
            print(mcs.to_string(float_format=lambda v: f"{v:8.4f}"))

        # --------------------------------------------------------- figures
        fig, _ = plot_mse_by_horizon(
            per_model_best,
            title=f"Target {target}",
            ylabel="POOS MSE (best predictor set)",
            highlight="TVNN",
        )
        save_figure(fig, str(FIGURES / f"mse_by_horizon_{target.lower()}"))

        if keep_for_plots is not None:
            fig, _ = plot_cssed(
                keep_for_plots.errors, benchmark="AR",
                title=(f"{target}: cumulative squared error difference vs. AR "
                       f"(h={keep_for_plots.h}, P={keep_for_plots.P})"),
            )
            save_figure(fig, str(FIGURES / f"cssed_{target.lower()}"))

            fig, _ = plot_forecast_vs_actual(
                keep_for_plots.forecasts,
                title=f"{target}: realised vs. forecast (h={keep_for_plots.h})",
                ylabel="h-month change" if target.upper() == "UNRATE" else "log h-th difference",
            )
            save_figure(fig, str(FIGURES / f"forecasts_{target.lower()}"))

        # A final full-sample TVNN, purely to show the drifting weights.
        print(f"\nFitting a full-sample TVNN on {target} for the weight plot ...")
        from tvwnn.data import build_fred_dataset

        ds = build_fred_dataset(
            target=target, h=max(args.horizons), n_factors=5,
            kx=args.kx, ky=args.ky, start=args.train_start,
            end=args.poos_end, data=panel, tcodes=tcodes,
        )
        cfg = DEFAULTS.get(target.upper(), DEFAULTS["INDPRO"])
        m = TVNN(n_outer=30, grad_steps=10, random_state=args.seed, **cfg)
        m.fit(ds.U, ds.y)
        fig, _ = plot_time_varying_weights(
            m.smoothed_states_, dates=ds.dates, state_cov=m.smoothed_state_cov_,
            title=(f"{target}: smoothed time-varying weights "
                   f"(d={cfg['d']}, h={max(args.horizons)})"),
        )
        save_figure(fig, str(FIGURES / f"weights_{target.lower()}"))

    print(f"\nDone in {time.time() - started:.0f}s. "
          f"Tables in {TABLES}/, figures in {FIGURES}/.")


if __name__ == "__main__":
    main()
