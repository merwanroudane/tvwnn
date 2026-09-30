"""
Where does the TVNN actually help?  A sweep across targets and horizons.

The paper studies two series (INDPRO, UNRATE) at four horizons, all of them
``h > 1``.  At ``h > 1`` its forecast target ``y_{t+h} - y_t`` is an
*overlapping* difference, so the errors are MA(``h-1``) by construction -- and
a random-walk coefficient is an excellent way to absorb serial correlation
that the model has no other way to express.

That suggests a sharp test the paper never runs: **``h = 1``**, where the
target does not overlap and there is no induced serial correlation for the
state to soak up.  If the TVNN's difficulties on real data come from the
overlap, they should largely disappear at ``h = 1``.

This script runs AR / ARX / FFNN / TVNN over several FRED-MD targets at
``h in {1, 3, 12}`` and reports relative MSE with Diebold-Mariano tests
against both the AR benchmark and the fixed-weight FFNN -- the latter being
the comparison the paper's own claim rests on and never tests.

Usage::

    python examples/other_targets_study.py
    python examples/other_targets_study.py --targets CPIAUCSL PAYEMS --horizons 1

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
    ARBenchmark,
    FeedForwardForecaster,
    RidgeARX,
    TVNN,
    dm_test,
    fetch_fred_md,
    plot_mse_by_horizon,
    poos_experiment,
    render_console,
    render_latex,
    save_figure,
    target_kind_from_tcode,
)

TABLES = Path("tables")
FIGURES = Path("figures")

#: A spread of macro and financial targets, chosen to vary in persistence and
#: in how much nonlinearity the literature attributes to them.
DEFAULT_TARGETS = ["INDPRO", "UNRATE", "CPIAUCSL", "PAYEMS", "FEDFUNDS", "S&P 500"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 12])
    ap.add_argument("--n-factors", type=int, default=5)      # P = 24
    ap.add_argument("--poos-start", default="1990-01-01")
    ap.add_argument("--poos-end", default="2014-12-01")
    ap.add_argument("--train-start", default="1960-01-01")
    ap.add_argument("--retrain-every", type=int, default=60)
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--l2", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    print("Loading FRED-MD ...")
    panel, tcodes = fetch_fred_md(cache_dir="data")

    rows: list[dict] = []
    started = time.time()

    for target in args.targets:
        if target not in panel.columns:
            print(f"  skipping {target!r}: not in this vintage")
            continue
        kind = target_kind_from_tcode(int(tcodes[target]))
        print(f"\n### {target}  (tcode {int(tcodes[target])} -> {kind} target)")

        for h in args.horizons:
            ar = poos_experiment(
                panel, tcodes, {"AR": lambda **_: ARBenchmark()},
                target=target, h=h, n_factors=0,
                poos_start=args.poos_start, poos_end=args.poos_end,
                train_start=args.train_start, retrain_every=args.retrain_every,
                verbose=False,
            )
            res = poos_experiment(
                panel, tcodes,
                {
                    "ARX": lambda **_: RidgeARX(),
                    "FFNN": lambda **_: FeedForwardForecaster(
                        hidden_sizes=(32,), epochs=300, l2=args.l2,
                        random_state=args.seed,
                    ),
                    "TVNN": lambda **_: TVNN(
                        d=args.d, hidden_sizes=(32,), l2=args.l2,
                        n_outer=30, grad_steps=10, random_state=args.seed,
                    ),
                },
                target=target, h=h, n_factors=args.n_factors,
                poos_start=args.poos_start, poos_end=args.poos_end,
                train_start=args.train_start, retrain_every=args.retrain_every,
                verbose=False,
            )
            res.forecasts["AR"] = ar.forecasts["AR"]

            mse = res.mse()
            err = res.errors
            base = mse["AR"]
            for model in ("AR", "ARX", "FFNN", "TVNN"):
                if model not in mse:
                    continue
                stars = ""
                if model != "AR":
                    stars = dm_test(
                        err[model].to_numpy(), err["AR"].to_numpy(), h=h
                    ).stars()
                rows.append(
                    {
                        "target": target, "h": h, "model": model,
                        "mse": float(mse[model]),
                        "relative_mse": float(mse[model] / base),
                        "stars": stars,
                    }
                )

            vs_ff = dm_test(err["TVNN"].to_numpy(), err["FFNN"].to_numpy(), h=h)
            print(
                f"  h={h:<3d} AR 1.000 | ARX {mse['ARX'] / base:5.3f} | "
                f"FFNN {mse['FFNN'] / base:5.3f} | TVNN {mse['TVNN'] / base:5.3f}"
                f"   TVNN vs FFNN: p={vs_ff.p_value:.3f} {vs_ff.stars()}"
            )

    long = pd.DataFrame(rows)
    long.to_csv(TABLES / "other_targets.csv", index=False)

    # ---- Table: rows are (target, model), columns are horizons.
    wide_v = long.pivot_table(index=["target", "model"], columns="h", values="relative_mse")
    wide_s = long.pivot_table(index=["target", "model"], columns="h", values="stars", aggfunc="first")
    table = pd.DataFrame(index=wide_v.index, columns=wide_v.columns, dtype=object)
    for idx in wide_v.index:
        for col in wide_v.columns:
            v = wide_v.loc[idx, col]
            st = wide_s.loc[idx, col]
            table.loc[idx, col] = "" if not np.isfinite(v) else f"{v:.3f}{st if isinstance(st, str) else ''}"

    order = {m: i for i, m in enumerate(["AR", "ARX", "FFNN", "TVNN"])}
    table = table.reset_index()
    table["_k"] = table["model"].map(lambda m: order.get(m, 9))
    table = table.sort_values(["target", "_k"]).drop(columns="_k").set_index(["target", "model"])
    table.columns = [f"h = {c}" for c in table.columns]
    table.attrs["benchmark"] = "AR"

    title = (
        f"POOS MSE relative to the AR benchmark, P={4 * args.n_factors + 4}, "
        f"{args.poos_start[:7]} to {args.poos_end[:7]}"
    )
    console = render_console(table, title=title, bold_min_within="target")
    print("\n" + console)
    (TABLES / "other_targets.txt").write_text(console, encoding="utf-8")
    (TABLES / "other_targets.tex").write_text(
        render_latex(table, caption=title, label="tab:other-targets",
                     bold_min_within="target"),
        encoding="utf-8",
    )

    # ---- Figure: one panel per target.
    for target in long["target"].unique():
        sub = long[long["target"] == target]
        by_model = {
            m: dict(zip(g["h"], g["relative_mse"]))
            for m, g in sub.groupby("model")
        }
        fig, _ = plot_mse_by_horizon(
            by_model, title=f"Target {target}",
            ylabel="POOS MSE relative to AR", highlight="TVNN",
        )
        save_figure(fig, str(FIGURES / f"other_{target.replace(' ', '_').lower()}"))

    print(f"\nDone in {time.time() - started:.0f}s.")


if __name__ == "__main__":
    main()
