"""
tvwnn quickstart -- five minutes, no download required.

Fits a TVNN to a simulated series with genuinely drifting coefficients,
compares it with an identically tuned fixed-weight network and an AR
benchmark, and writes the three core figures.

Run::

    python examples/quickstart.py

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

import numpy as np

from tvwnn import (
    ARBenchmark,
    FeedForwardForecaster,
    TVNN,
    dm_test,
    model_summary_table,
    plot_convergence,
    plot_time_varying_weights,
    save_figure,
    simulate_tvnn,
)
from tvwnn.plots import plot_cssed
import pandas as pd

OUT = "figures"


def main() -> None:
    # ------------------------------------------------ 1. data
    # A nonlinear basis with random-walk coefficients -- the DGP the TVNN is
    # designed for.  `ds.y[t]` is already y_{t+h}; `ds.U[t]` is u_t.
    ds, truth = simulate_tvnn(n=420, p=3, h=1, sigma=0.3, omega=0.02, seed=0)
    print(ds)

    n_train = 350
    U_tr, y_tr = ds.U[:n_train], ds.y[:n_train]
    U_te, y_te = ds.U[n_train:], ds.y[n_train:]

    # ------------------------------------------------ 2. fit
    # Defaults reproduce Appendix A.3's budget: 30 outer x 10 Adam = 300 epochs.
    tvnn = TVNN(
        d=1,
        hidden_sizes=(32,),
        activation="tanh",
        n_outer=30,
        grad_steps=10,
        learning_rate=0.01,
        l2=0.02,
        random_state=0,
        verbose=False,
    ).fit(U_tr, y_tr)

    print()
    print(model_summary_table(tvnn.summary()))

    # ------------------------------------------------ 3. forecast
    # The filter absorbs each new observation as it arrives; the network stays
    # frozen.  gap=h carries the state forward to the forecast origin.
    preds = np.array(
        [
            tvnn.forecast(
                U_te[i : i + 1],
                gap=ds.h,
                U_hist=np.vstack([U_tr, U_te[:i]]),
                y_hist=np.r_[y_tr, y_te[:i]],
            )[0]
            for i in range(len(y_te))
        ]
    )

    ffnn = FeedForwardForecaster(
        hidden_sizes=(32,), epochs=300, learning_rate=0.01, l2=0.02, random_state=0
    ).fit(U_tr, y_tr)
    ar = ARBenchmark().fit(U_tr, y_tr)

    errors = pd.DataFrame(
        {
            "TVNN": y_te - preds,
            "FFNN": y_te - ffnn.forecast(U_te),
            "AR": y_te - ar.forecast(U_te),
        }
    )

    # ------------------------------------------------ 4. evaluate
    print()
    print("Out-of-sample accuracy")
    print("-" * 52)
    print(f"{'model':<10}{'MSE':>12}{'vs AR':>10}{'DM p':>10}{'':>6}")
    mse = (errors**2).mean()
    for name in ("AR", "FFNN", "TVNN"):
        if name == "AR":
            print(f"{name:<10}{mse[name]:>12.4f}{1.0:>10.3f}{'':>10}{'':>6}")
            continue
        r = dm_test(errors[name].to_numpy(), errors["AR"].to_numpy(), h=ds.h)
        print(
            f"{name:<10}{mse[name]:>12.4f}{mse[name] / mse['AR']:>10.3f}"
            f"{r.p_value:>10.4f}{r.stars():>6}"
        )
    print("-" * 52)

    # The comparison the paper never tests: TVNN against the fixed-weight net.
    r = dm_test(errors["TVNN"].to_numpy(), errors["FFNN"].to_numpy(), h=ds.h)
    print(f"TVNN vs FFNN: DM = {r.statistic:.3f}, p = {r.p_value:.4f} {r.stars()}")
    print("(negative statistic favours the TVNN)")

    # ------------------------------------------------ 5. figures
    fig, _ = plot_time_varying_weights(
        tvnn.smoothed_states_,
        dates=ds.dates[:n_train],
        state_cov=tvnn.smoothed_state_cov_,
        title="Smoothed time-varying weights (simulated data)",
        shade_recessions=False,
    )
    print("\nwrote", save_figure(fig, f"{OUT}/quickstart_weights"))

    fig, _ = plot_convergence(tvnn.history_.as_dict(), title="TVNN training (simulated)")
    print("wrote", save_figure(fig, f"{OUT}/quickstart_convergence"))

    errors.index = ds.dates[n_train:]
    fig, _ = plot_cssed(errors, benchmark="AR", shade_recessions=False)
    print("wrote", save_figure(fig, f"{OUT}/quickstart_cssed"))


if __name__ == "__main__":
    main()
