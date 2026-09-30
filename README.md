# tvwnn — Time-Varying Weight Neural Networks

[![PyPI](https://img.shields.io/pypi/v/tvwnn.svg)](https://pypi.org/project/tvwnn/)
[![Python](https://img.shields.io/pypi/pyversions/tvwnn.svg)](https://pypi.org/project/tvwnn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/merwanroudane/tvwnn/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-30%20passing-brightgreen.svg)](https://github.com/merwanroudane/tvwnn/tree/main/tests)

```bash
pip install tvwnn
```

A faithful, tested Python implementation of the **TVNN**:

> Rudd, W., H. Bondell & J. Silver (2026). "Augmenting Neural Networks With Time-Varying Weights." *Journal of Forecasting* **45**(1): 22–28. [doi:10.1002/for.70014](https://doi.org/10.1002/for.70014)

A feedforward network learns a low-dimensional basis `z_t = f_w(u_t)`. The weights sitting on that basis are **not constant** — they follow a random walk, estimated by a Kalman filter/smoother inside an EM algorithm that alternates with Adam steps on the network. The result combines a neural network's ability to find nonlinearity with a time-varying-parameter model's ability to track structural change.

**Author:** Dr Merwan Roudane · [merwanroudane920@gmail.com](mailto:merwanroudane920@gmail.com) · [github.com/merwanroudane](https://github.com/merwanroudane)
**PyPI:** <https://pypi.org/project/tvwnn/> · **Repository:** <https://github.com/merwanroudane/tvwnn> · **Licence:** MIT

> **Read [§8b](#8b-two-things-that-only-show-up-on-real-data) and [§8c](#8c-where-the-tvnn-actually-helps-other-targets-other-data) before using this on real data.** Every component is verified independently, but this implementation does *not* reproduce the paper's published forecast accuracy on its own two targets, and two conventions the paper leaves unstated materially affect the results. It does reproduce the paper's central claim — time-varying weights beating fixed ones — on the federal funds rate.

---

## Contents

1. [The model in one screen](#1-the-model-in-one-screen)
2. [Installation](#2-installation)
3. [Sixty-second quickstart](#3-sixty-second-quickstart)
4. [Real economic data in ten lines](#4-real-economic-data-in-ten-lines)
5. [How this code was written, step by step](#5-how-this-code-was-written-step-by-step) ← the build guide
6. [Syntax reference](#6-syntax-reference)
7. [Faithfulness: code step → paper equation](#7-faithfulness-code-step--paper-equation)
8. [Three things the paper leaves open](#8-three-things-the-paper-leaves-open)
8b. [Two things that only show up on real data](#8b-two-things-that-only-show-up-on-real-data)
8c. [Where the TVNN actually helps: other targets, other data](#8c-where-the-tvnn-actually-helps-other-targets-other-data)
9. [Reproducing the paper's tables and figures](#9-reproducing-the-papers-tables-and-figures)
10. [Output gallery](#10-output-gallery)
11. [Testing](#11-testing)
12. [Practical guidance and troubleshooting](#12-practical-guidance-and-troubleshooting)
13. [Citation](#13-citation)

---

## 1. The model in one screen

Start from a univariate target `y_t` and a predictor panel `x_t ∈ ℝᵖ`. Stack `kx` predictor lags and `ky` target lags into

```
u_t = ( x'_{t-kx+1:t} , y'_{t-ky+1:t} )'          dim(u_t) = P = p·kx + ky        (eq. 1)
```

A **linear** time-varying-parameter model would write `y_{t+h} = u_t'β_t + ε_t` with `β` a random walk. The TVNN inserts a network first:

```
z_t = f_w(u_t) ∈ ℝ^d                    the learned basis (a bottleneck layer)

y_{t+h} = z_t' β_t + ε_t                ε_t ~ N(0, σ²)                           (eq. 4)
β_t     = β_{t-1} + ξ_t                 ξ_t ~ N(0, Ω),   β_0 = b_0
```

Optionally a scalar output node `z̃_t` carries the non-drifting part: `y_{t+h} = z_t'β_t + z̃_t + ε_t`.

**Why this is cheap.** The state dimension is `d`, a design choice, *not* `P`. Appendix A.3 of the paper selects **`d ∈ {1, 2}`**, so the Kalman recursions are trivial even when `P = 424`. That is the whole reason the method scales where ordinary high-dimensional TVP models do not.

**How it is estimated.** Two loops over *different* objectives, and the split is deliberate:

| Loop | Optimises | Uses | Why |
|---|---|---|---|
| inner, EM | `θ = {σ², Ω, b_0}` | **smoothed** states | closed-form M step (Holmes 2013) |
| outer, Adam | network weights `w` | **filtered** states | only filtered states exist out of sample |

Training the basis against smoothed states would generalise badly, because at forecast time you have no future data to smooth with. Inside the Adam block, `b_t` and `V_t` are frozen constants — which is why a gradient step costs exactly what an ordinary network's does.

```
Algorithm 1
  initialise {w, θ}, Adam moments
  for i = 0 : N
      z_t = f_w(u_t)                            # basis, network frozen
      for k = 0 : K                             # EM to convergence
          Kalman filter → RTS smoother → closed-form M step for θ
      b_t, V_t ← Kalman filter                  # states frozen from here
      optional momentum:  b ← γ·b + (1−γ)·b_prev
      for ℓ = 0 : M                             # Adam on the network
          ∇_w [ log p(y | θ, w) − α‖w‖² ]
  return ŵ, θ̂
```

Defaults here are `N = 30`, `M = 10` → **300 epochs**, exactly the fixed budget Appendix A.3 specifies.

---

## 2. Installation

```bash
pip install tvwnn
```

Or from source, for development:

```bash
git clone https://github.com/merwanroudane/tvwnn.git
cd tvwnn
pip install -e .
```

Requires Python ≥ 3.10, `numpy`, `pandas`, `scipy`, `torch`, `scikit-learn`, `matplotlib`. CPU-only PyTorch is plenty — at `d ∈ {1,2}` there is nothing for a GPU to do.

For the test suite (which cross-checks against an independent Kalman implementation):

```bash
pip install -e ".[test]"
```

---

## 3. Sixty-second quickstart

```python
import numpy as np
from tvwnn import TVNN, simulate_tvnn, dm_test, FeedForwardForecaster

# A DGP with a nonlinear basis and genuinely drifting coefficients.
ds, truth = simulate_tvnn(n=420, p=3, h=1, sigma=0.3, omega=0.02, seed=0)

n_train = 350
U_tr, y_tr = ds.U[:n_train], ds.y[:n_train]
U_te, y_te = ds.U[n_train:], ds.y[n_train:]

model = TVNN(d=1, hidden_sizes=(32,), n_outer=30, grad_steps=10,
             l2=0.02, random_state=0).fit(U_tr, y_tr)

print(model.summary()["sigma2"], model.summary()["omega_trace"])
```

Forecast, expanding the filter one observation at a time while the network stays frozen:

```python
preds = np.array([
    model.forecast(U_te[i:i+1], gap=ds.h,
                   U_hist=np.vstack([U_tr, U_te[:i]]),
                   y_hist=np.r_[y_tr, y_te[:i]])[0]
    for i in range(len(y_te))
])

ffnn = FeedForwardForecaster(hidden_sizes=(32,), epochs=300,
                             l2=0.02, random_state=0).fit(U_tr, y_tr)

r = dm_test(y_te - preds, y_te - ffnn.forecast(U_te), h=ds.h)
print(f"DM = {r.statistic:.2f}, p = {r.p_value:.4f} {r.stars()}")
```

The full script, with figures, is [`examples/quickstart.py`](https://github.com/merwanroudane/tvwnn/blob/main/examples/quickstart.py):

```bash
python examples/quickstart.py
```

```
Out-of-sample accuracy
----------------------------------------------------
model              MSE     vs AR      DM p
AR              2.9388     1.000
FFNN            2.7071     0.921    0.2899
TVNN            0.2284     0.078    0.0000   ***
----------------------------------------------------
TVNN vs FFNN: DM = -8.278, p = 0.0000 ***
```

---

## 4. Real economic data in ten lines

```python
from tvwnn import fetch_fred_md, build_fred_dataset, TVNN

panel, tcodes = fetch_fred_md(cache_dir="data")      # downloads & caches FRED-MD

ds = build_fred_dataset(target="UNRATE", h=12, n_factors=5,   # P = 24
                        start="1960-01-01", end="2014-12-01",
                        data=panel, tcodes=tcodes)

model = TVNN(d=2, hidden_sizes=(32,), l2=0.02, random_state=0).fit(ds.U, ds.y)
print(model.summary())
```

`n_factors` controls the predictor set, reproducing the paper's three configurations:

| `n_factors` | Design | `P` | Paper |
|---|---|---|---|
| `5` | 4 target lags + 4 lags × 5 principal components | 24 | `P = 24` |
| `15` | 4 target lags + 4 lags × 15 principal components | 64 | `P = 64` |
| `None` | 4 target lags + 4 lags × every series, no PCA | ~460 | `P = 424` |
| `0` | 4 target lags only | 4 | the AR benchmark |

Targets follow section 4 of the paper: **INDPRO** gets the log `h`-th difference `log y_{t+h} − log y_t`, **UNRATE** the aggregate `h`-month change `y_{t+h} − y_t`.

> **On the vintage.** The paper never says which FRED-MD vintage it used, and the database is revised monthly, so its "105 series after removing missing entries" cannot be reproduced. Pass `vintage="2015-01"` to pin one. A recent vintage yields ~115 complete series over 1960–2014, so the high-dimensional setting becomes `P = 460` instead of 424 — same arithmetic, different vintage.

> **One trap this library handles for you.** FRED-MD's last one or two months are ragged: at the time of writing, the latest vintage has 47 of 126 series filled in for its final month. Applying `dropna(axis=1)` to the raw file therefore throws away three quarters of the database — INDPRO and UNRATE included. `transform_panel` trims ragged edges *before* dropping columns.

---

## 5. How this code was written, step by step

This section is the build guide: what each module does, why it exists, the equation it implements, and the trap it avoids. Read it in order and you can rewrite the library from scratch.

### Step 1 — Decide the backend split

The paper's Algorithm 1 has two loops over different objectives, and only one of them needs derivatives.

* The **EM loop** runs with the network frozen. It is pure recursion over tiny matrices — no autodiff required.
* The **Adam loop** runs with the states frozen. Gradients flow only through `z_t`.

So: **NumPy for the state space, PyTorch for the network.** That mirrors the algorithm exactly, keeps the filter readable, and avoids fighting an autodiff framework over a sequential scan. Anything the network touches is a `torch` tensor; anything the filter touches is a `float64` array.

> Why not JAX? `lax.scan` would be faster in principle, but at `d ∈ {1, 2}` the filter is already ~0.9 ms per pass (see step 3) and the Adam steps dominate. PyTorch installs cleanly everywhere, which matters more here.

### Step 2 — Write the Kalman filter ([`tvwnn/kalman.py`](https://github.com/merwanroudane/tvwnn/blob/main/tvwnn/kalman.py))

The state space is a *local level model with a scalar observation*:

```
predict:     a_{t|t-1} = a_{t-1|t-1}          P_{t|t-1} = P_{t-1|t-1} + Ω
innovation:  v_t = y_t − z_t'a_{t|t-1}        F_t = z_t'P_{t|t-1}z_t + σ²
gain:        K_t = P_{t|t-1} z_t / F_t
update:      a_{t|t} = a_{t|t-1} + K_t v_t    P_{t|t} = P_{t|t-1} − K_t z_t' P_{t|t-1}

loglik = −½ Σ_t [ log 2π + log F_t + v_t²/F_t ]
```

Initialisation: `β_0 ~ N(b_0, V_0)` and `β_1 = β_0 + ξ_1`, so the prediction for the first observation is `N(b_0, V_0 + Ω)`.

Two details that are easy to get wrong:

* **Return both `a_pred` and `a_filt`.** They are *not* interchangeable. `a_pred` has not seen observation `t`; `a_filt` has. Which one enters eq. (8) decides whether the objective is a genuine prediction error decomposition — see [§8](#8-three-things-the-paper-leaves-open).
* **Symmetrise `P` after every update.** Over 700 iterations, floating-point drift makes the covariance non-symmetric and the smoother gains go subtly wrong.

### Step 3 — Write the RTS smoother **and the lag-one covariance**

This is the step most likely to be done wrong, silently.

The standard RTS recursion gives `β̂_t` and `Σ̂_{β_t}`:

```
J_t = P_{t|t} (P_{t+1|t})⁻¹
a_{t|n} = a_{t|t} + J_t (a_{t+1|n} − a_{t+1|t})
P_{t|n} = P_{t|t} + J_t (P_{t+1|n} − P_{t+1|t}) J_t'
```

But eq. (7) of the paper also needs `ξ̂_t` and `Σ̂_{ξ_t}`, and for a random walk `ξ_t = β_t − β_{t−1}`, so

```
Var[ξ_t | y] = P_{t|n} + P_{t−1|n} − P_{t,t−1|n} − P_{t,t−1|n}'
```

**`P_{t,t−1|n}` — the lag-one smoothed covariance — is not returned by a plain RTS smoother.** It needs its own recursion (Shumway & Stoffer Property 6.3, with transition matrix `Φ = I`):

```
P_{n,n−1|n} = (I − K_n z_n') P_{n−1|n−1}
P_{t,t−1|n} = P_{t|t} J_{t−1}' + J_t (P_{t+1,t|n} − P_{t|t}) J_{t−1}'
```

Omit it and you compute `Var[ξ_t] = P_t + P_{t−1}`, which **inflates `Ω̂` and raises no error at all**. Nothing crashes; the model just concludes the coefficients drift much more than they do. `tests/test_em.py::test_loglikelihood_increases_monotonically` is the tripwire: dropping the cross term breaks EM's monotonicity guarantee.

**Fast path.** Since `d ∈ {1, 2}`, `_filter_scalar` and `_smoother_scalar` handle `d = 1` in pure Python floats. NumPy's per-call overhead on 1×1 arrays dominated everything: this took a filter+smoother pass from 11.3 ms to 0.86 ms, a 13× speedup, and `tests/test_kalman.py` asserts it matches the matrix path to 1e-10.

### Step 4 — Write the M step ([`tvwnn/em.py`](https://github.com/merwanroudane/tvwnn/blob/main/tvwnn/em.py))

Differentiate `Q` (eq. 6) and set to zero. The paper prints none of these — it delegates to Holmes (2013) by citation — so they come from that technical report:

```
σ̂²  = (1/T) Σ_t [ (y_{t+h} − z_t'β̂_t)² + z_t' Σ̂_{β_t} z_t ]
Ω̂   = (1/T) Σ_t [ Σ̂_{ξ_t} + ξ̂_t ξ̂_t' ]
b̂_0 = β̂_0        (the smoothed *initial* state, so smooth back to index 0)
```

Note the `z_t' Σ̂_{β_t} z_t` term in `σ̂²`: it is the state-uncertainty correction, and it is what stops `σ̂²` collapsing when the state is poorly determined.

Floor `σ²` and the diagonal of `Ω` at ~1e-10, or a degenerate series divides by zero in the filter.

**Validate here, before writing any network code.** With a *fixed* design matrix the M step is an exact maximiser, so EM must recover known-truth parameters and the log-likelihood must increase monotonically. It does — `σ² = 0.25` recovered as `0.2606`, `Ω = diag(0.010, 0.004)` as `diag(0.0105, 0.0025)` on `n = 800`.

### Step 5 — Cross-check against an independent implementation

Before trusting anything, check the filter against code you did not write. `statsmodels`' `MLEModel` implements the same state space in Durbin–Koopman conventions:

```
our loglik    -219.6990712607
statsmodels   -219.6990712601
max |v diff|      5.8e-10
max |F diff|      1.6e-08
max |filt diff|   1.7e-10
```

This is the strongest available correctness argument for the half of the model that contains no novel mathematics. Whatever goes wrong later is in the alternation, not the filter. It lives in `tests/test_kalman.py::test_filter_matches_statsmodels`.

### Step 6 — Write the network ([`tvwnn/network.py`](https://github.com/merwanroudane/tvwnn/blob/main/tvwnn/network.py))

Appendix A.3 pins everything down:

```
u_t → Dense(32, tanh) → [Dense(16, tanh)] → Dense(d) → z_t
```

Glorot/Xavier uniform init, zero biases. **`d ∈ {1, 2}`.**

The reading that matters: the paper's "final hidden layer `z_t ∈ ℝ^d`" combined with "first hidden layer of size 32" means `z_t` is a **bottleneck output layer of width 1 or 2**, not the 32-unit hidden layer. Get this wrong and you build a 32-dimensional state — a completely different, far more expensive model.

Dropout is applied only in training mode. `basis_numpy()` forces eval mode so the basis handed to the Kalman recursions is deterministic; a stochastic `z_t` would make the filter meaningless.

### Step 7 — Assemble Algorithm 1 ([`tvwnn/model.py`](https://github.com/merwanroudane/tvwnn/blob/main/tvwnn/model.py))

```python
for _ in range(n_outer):
    Z, z_tilde = net.basis_numpy(U_t)           # network frozen
    theta, _   = run_em(target, Z, theta, ...)  # inner EM to convergence
    filt       = kalman_filter(target, Z, ...)
    b, V       = filt.a_pred, filt.P_pred       # states frozen

    for _ in range(grad_steps):                 # Adam, states constant
        z, zt = net(U_t)                        # z recomputed: w changes
        resid = y_t - einsum("td,td->t", z, b_t)
        F     = sigma2 + einsum("td,tde,te->t", z, V_t, z)
        loss  = sum(resid**2 / F) + l2 * net.l2_penalty()
        loss.backward(); optimiser.step()
```

Three points worth stating explicitly:

* `b` and `V` are **frozen** inside the inner loop. That is the paper's computational claim, and it is only true under freezing.
* `z` is **recomputed each Adam step** (the weights are changing) and appears in *both* the mean and the variance — the gradient flows through both.
* **Momentum is applied to `b`, the filtered states, not to the weights** (Adam handles those). It damps oscillation between the two loops.

**The loss is averaged over `t`, and that decides what `l2` means.** Algorithm 1 writes the objective as `log p(y | θ, w) − α‖w‖²`, and a log-likelihood is a *sum*. But dividing it by `n` is a positive rescaling that leaves its maximiser unchanged while completely changing what the penalty `α` is worth — so the convention has to be pinned down, and the paper does not state it.

Appendix A.3 does, indirectly. It says the networks use **"heavy regularization"**, with `α = 0.01 × 2^s`, `s ∈ {1..6}`, i.e. 0.02 to 0.64. With Glorot initialisation a 24→32→1 network has `‖w‖² ≈ 28`, so the penalty is roughly 0.6 to 18. Against a *summed* loss of order 600 that is 0.1% — not heavy, not anything. Against an *averaged* loss of order 1 it is dominant, which is what "heavy" means.

So this library averages, for the TVNN and for every baseline, and `l2` therefore means the same thing in all of them — which is what makes "similarly tuned" literally true rather than an assumption. It is not a cosmetic choice: on FRED-MD unemployment at `h = 12`, switching the plain FFNN baseline from summed to averaged moved its MSE from 1.60× the AR benchmark to 1.25×, and the TVNN from 1.56× to 1.47×.

**Standardisation.** Not mentioned anywhere in the paper, but a `tanh` network on raw macro data will not train. Inputs and target are standardised internally and the scaling is inverted on output, so it is invisible to the caller. It does mean `σ̂²` and `Ω̂` are reported on the standardised axis.

### Step 8 — Get the `h`-step alignment right ([`tvwnn/data.py`](https://github.com/merwanroudane/tvwnn/blob/main/tvwnn/data.py))

A supervised pair is `(u_t, y_{t+h})`. At forecast origin `o` only observations dated `≤ o` exist, so the last **usable** training pair is `(u_{o−h}, y_o)`.

That means the state at index `o` is `h` random-walk steps beyond the last one the filter saw:

```
E[β_o | data] = β_{T|T}
Var[β_o | data] = P_{T|T} + h·Ω
```

which is exactly what `forecast(..., gap=h)` does. Skip it and you either leak future information or understate the forecast variance by `h·Ω`. `tests/test_model.py::test_forecast_gap_reproduces_random_walk_inflation` asserts the inflation is exactly `h·Ω`.

One more alignment note: the paper prints the lag range in eq. (1) as `x'_{t−kx:t}`, which spans `kx + 1` lags and contradicts its own `P = p·kx + ky`. The empirical `P = 24 = 4 + 5×4` confirms exactly `kx` lags are intended. `build_lag_matrix` uses `t−kx+1:t`.

### Step 9 — Build the evaluation loop ([`tvwnn/evaluate.py`](https://github.com/merwanroudane/tvwnn/blob/main/tvwnn/evaluate.py))

Expanding window from 1960, re-estimate every 30 months, POOS 1990–2014, hyperparameters from A.3.

Two things `poos_experiment` does that the paper does not describe:

* **PCA inside the training window.** Estimate the diffusion indices on training data only and *project* the evaluation period through those loadings (`PCAProjector`). Fitting the PCA once on the full sample leaks the future into every forecast.
* **Re-filter at every origin.** The network is re-estimated only every 30 months, but the Kalman filter absorbs each new observation at almost no cost. Holding the state fixed between re-estimations wastes the main advantage of the state-space layer. Pass `refilter_each_origin=False` to disable.

### Step 10 — Report like a journal ([`tvwnn/tables.py`](https://github.com/merwanroudane/tvwnn/blob/main/tvwnn/tables.py), [`tvwnn/plots.py`](https://github.com/merwanroudane/tvwnn/blob/main/tvwnn/plots.py))

Tables render twice — fixed-width console with rules, and `booktabs` LaTeX — with block minima marked and a significance footer, following the layout of Tables 1 and 2.

Figures are **light only**: white background, Okabe–Ito colour-blind-safe palette, serif stack, no top/right spines, 300 dpi PNG + vector PDF. There is no dark variant, by design — dark figures do not print.

### Step 11 — Run it on real data, and believe what you see

Everything up to here passes on simulated data. Then FRED-MD breaks it twice, and both failures are worth the time it takes to find them:

* `|z|` drifts to **42** because the basis layer is linear and nothing pins its scale;
* the EM concludes `tr(Ω)/σ² ≈ 11` — "all drift, no noise" — because at `h = 12` the target `y_{t+h} − y_t` is an overlapping difference whose MA(11) errors the state is delighted to absorb.

Together they produced out-of-sample forecasts spanning `[−18.4, 9.8]` against a realised `[−1.5, 4.0]`. The fixes are `normalize_basis` and `sn_cap`, both documented in [§8b](#8b-two-things-that-only-show-up-on-real-data), both one keyword from the paper's literal behaviour.

The lesson generalises: a component-verified implementation can still be useless on the data it was written for. Simulate to check the mathematics, then run the real thing and look at the *range* of the forecasts before you look at their mean squared error.

### Step 12 — Add what the paper omits

Three gaps worth filling, each clearly labelled as an extension rather than a reproduction:

* **`dm_matrix`** — all pairwise Diebold–Mariano tests. The paper's headline is that the TVNN beats a similarly tuned FFNN, but every star in Tables 1–2 is against the *AR* benchmark; TVNN vs FFNN is never tested.
* **`model_confidence_set`** — Hansen–Lunde–Nason MCS. The paper reports 2 targets × 4 horizons × 3 predictor sets × 6 models of pairwise tests with no multiplicity control.
* **`forecast(..., return_variance=True)`** — the Kalman machinery hands you `F_t = σ² + z_t'V_t z_t` for free. The paper says uncertainty "cannot be propagated through these models in the usual way" and reports point forecasts only. This is a predictive variance *conditional on `ŵ`*; it ignores uncertainty in the network weights, and is labelled as such.

---

## 6. Syntax reference

### `TVNN` — the estimator

```python
TVNN(d=1, hidden_sizes=(32,), activation="tanh", dropout=0.0, fixed_effect=False,
     n_outer=30, grad_steps=10, em_iter=50, em_tol=1e-6,
     learning_rate=0.01, l2=0.32, momentum=0.0,
     loss="weighted_sse", state="predicted", omega_structure="full",
     sn_cap=0.01, v0_scale=1e6, standardize=True, normalize_basis=True,
     random_state=None, device=None, verbose=False)
```

| Argument | Default | Meaning |
|---|---|---|
| `d` | `1` | Basis functions = state dimension. A.3 selects from `{1, 2}`. |
| `hidden_sizes` | `(32,)` | `(32,)` or `(32, 16)` in the paper. |
| `activation` | `"tanh"` | A.3 uses `tanh` throughout. |
| `dropout` | `0.0` | A.3 says "dropout with default parameters" without naming a rate; off by default. |
| `fixed_effect` | `False` | Add the `z̃_t` output node of eq. (4). |
| `n_outer` | `30` | `N`, outer iterations. |
| `grad_steps` | `10` | `M`, Adam steps per outer iteration. `30 × 10 = 300` epochs, A.3's budget. |
| `em_iter` | `50` | Cap `K` on inner EM ("run to convergence"). |
| `learning_rate` | `0.01` | A.3 draws from `0.01 × 2^s`, `s ∈ {0,1,2}`. |
| `l2` | `0.32` | `α`. A.3 draws from `0.01 × 2^s`, `s ∈ {1..6}`; TVNN selected `s=5` for INDPRO, `s=1` for UNRATE. |
| `momentum` | `0.0` | `γ` on the *filtered states*, not on weights. |
| `loss` | `"weighted_sse"` | `"weighted_sse"` = eq. (8) as printed; `"exact"` adds back `log F_t`. See [§8](#8-three-things-the-paper-leaves-open). |
| `state` | `"predicted"` | `β_{t\|t-1}` (a genuine prediction error decomposition) or `"filtered"` `β_{t\|t}`. See [§8](#8-three-things-the-paper-leaves-open). |
| `omega_structure` | `"full"` | `"full"` or `"diagonal"`. At `d ≤ 2` this is at most one parameter. |
| `sn_cap` | `0.01` | Bound on `tr(Ω)/(d·σ²)`. **Not in the paper** — see [§8b.1](#8b1-the-signal-to-noise-ratio-runs-away--sn_cap). `None` reproduces it literally. |
| `v0_scale` | `1e6` | Diagonal of `V_0`. `0.0` gives the paper's literal `β_0 = b_0`. |
| `standardize` | `True` | Internal scaling of `U` and `y`, inverted on output. |
| `normalize_basis` | `True` | Unit-variance columns of `z_t`. **Not in the paper** — see [§8b.2](#8b2-the-learned-basis-is-unbounded--normalize_basis). |

**Methods**

| Call | Returns |
|---|---|
| `.fit(U, y)` | `self`. `U[t] = u_t`, `y[t] = y_{t+h}` — use `build_supervised`. |
| `.predict()` | In-sample one-step-ahead fitted values, original scale. |
| `.forecast(u_new, gap=0, return_variance=False, U_hist=None, y_hist=None)` | Point forecast(s). **Set `gap=h`.** |
| `.summary()` | `dict` of `sigma2`, `Omega`, `omega_trace`, `b0`, `final_loglik`, settings. |

**Attributes after `fit`:** `theta_` (`StateParams`), `history_` (`FitHistory`), `smoothed_states_` `(n, d)`, `smoothed_state_cov_` `(n, d, d)`.

### `TVNNEnsemble`

```python
TVNNEnsemble(n_members=10, random_state=None, **tvnn_kwargs)
  .fit(U, y) · .forecast(u_new, gap=0) · .forecast_spread(u_new, gap=0)
```

A.3: "ensembles of 10 networks were used for each evaluation period." **Tables 1 and 2 of the paper are ensemble forecasts** — a single network will not reproduce them.

### Baselines

```python
ARBenchmark()                                      # OLS on target lags (P = 4)
RidgeARX(alphas=None)                              # RidgeCV
FeedForwardForecaster(hidden_sizes=(32,), activation="tanh", dropout=0.0,
                      epochs=300, learning_rate=0.01, l2=0.32, random_state=None)
RecurrentForecaster(cell="rnn"|"lstm", kx=4, ky=4, n_factors=5,
                    hidden_size=32, epochs=300, learning_rate=0.01,
                    l2=0.64, random_state=None)
```

All expose `.fit(U, y)` and `.forecast(u_new, gap=0)`; `gap` is ignored by the fixed-parameter models.

### Data

```python
fetch_fred_md(vintage=None, cache_dir="data", force_download=False) -> (DataFrame, Series)
fetch_fred_qd(vintage=None, cache_dir="data", force_download=False) -> (DataFrame, Series)
target_kind_from_tcode(tcode) -> "diff" | "logdiff"
transform_panel(data, tcodes, drop_incomplete=True, trim_ragged_edges=True, min_coverage=0.95)
apply_transformation(series, tcode)                # McCracken-Ng codes 1-7
diffusion_indices(X, n_factors, standardize=True) -> (factors, evr)
PCAProjector(n_factors).fit(X_train).transform(X)  # for POOS work
make_target(level, h, kind="diff"|"logdiff")
build_lag_matrix(values, n_lags, prefix="x")
build_supervised(factors, target_lagged, target_future, dates, h, kx=4, ky=4) -> Dataset
build_fred_dataset(target="INDPRO", h=12, n_factors=5, kx=4, ky=4,
                   start="1960-01-01", end=None, vintage=None,
                   target_kind=None, ...) -> Dataset
simulate_tvnn(n=400, p=3, h=1, d_true=1, sigma=0.3, omega=0.02, seed=0) -> (Dataset, dict)
```

`Dataset` carries `.U`, `.y`, `.dates` (forecast origins), `.target_dates`, `.h`, `.feature_names`.

### Evaluation

```python
poos_experiment(panel, tcodes, model_factories, target="INDPRO", h=12, n_factors=5,
                kx=4, ky=4, poos_start="1990-01-01", poos_end="2014-12-01",
                train_start="1960-01-01", retrain_every=30,
                refilter_each_origin=True, verbose=True) -> PoosResult

dm_test(errors_model, errors_benchmark, h=1, loss="squared",
        harvey_adjustment=True, alternative="two-sided") -> DMResult

model_confidence_set(losses, alpha=0.10, n_boot=1000, block_size=None, seed=0)
mse_table(results, benchmark="AR")

# Appendix A.3's grid and the per-period random search of section 4
A3_GRID, A3_GRID_RECURRENT
sample_hyperparameters(rng, grid=None)
select_hyperparameters(make_model, U, y, dates, valid_months=90, n_draws=8,
                       h=1, grid=None, seed=0)
```

`PoosResult` exposes `.forecasts`, `.errors`, `.squared_errors`, `.mse()`, `.relative_mse(benchmark)`, `.dm_against(benchmark)`.

`model_factories` maps a name to a **callable returning a fresh unfitted model** — a factory, not an instance, because every re-estimation date needs a new one. Pass `tune={"TVNN": True, ...}` and each listed factory is called with keyword arguments drawn by random search, so its signature must accept them.

### Reporting

```python
forecast_accuracy_table(long, value="relative_mse", benchmark="AR", model_order=None)
render_console(table, title="", note=None, bold_min_within="P", float_fmt="{:.3f}")
render_latex(table, caption="", label="tab:tvnn", note=None, bold_min_within="P")
model_summary_table(summary, title="TVNN estimates")
dm_matrix(errors, h=1)

set_journal_style(font_scale=1.0)
plot_mse_by_horizon(mse_by_model, title="", ylabel="MSE", highlight="TVNN")
plot_time_varying_weights(states, dates=None, state_cov=None, shade_recessions=True)
plot_forecast_vs_actual(forecasts, actual_col="actual", models=None)
plot_cssed(errors, benchmark, models=None)
plot_convergence(history)
plot_scree(explained_variance_ratio)
save_figure(fig, path, formats=("png", "pdf"))
```

---

## 7. Faithfulness: code step → paper equation

| Paper | Code |
|---|---|
| eq. (1) `u_t` | `data.build_lag_matrix`, `data.build_supervised` |
| eq. (2)–(3) linear TVP | `kalman.kalman_filter` with `Z = U` |
| eq. (4) TVNN measurement | `model.TVNN.fit`, `network.BasisNetwork` |
| eq. (4) fixed effect `z̃_t` | `network.BasisNetwork(fixed_effect=True)` |
| eq. (5) marginal likelihood | `kalman.FilterResult.loglik` |
| eq. (6) / App. A.1 eq. (9) `Q` | `em.expected_loglik` |
| eq. (7) E-step quantities | `kalman.kalman_smoother` → `a_smooth`, `P_smooth`, `xi_mean`, `xi_var` |
| M step (Holmes 2013) | `em.m_step` |
| eq. (8) / App. A.2 | the Adam block of `model.TVNN.fit` |
| Algorithm 1, outer loop | `model.TVNN.fit` |
| Algorithm 1, `for k = 0:K` | `em.run_em` |
| Algorithm 1, momentum on `b` | `model.TVNN(momentum=γ)` |
| Algorithm 1, `−α‖w‖²` | `network.BasisNetwork.l2_penalty` |
| §4 targets | `data.make_target` |
| §4 diffusion indices | `data.diffusion_indices`, `data.PCAProjector` |
| §4 POOS protocol | `evaluate.poos_experiment` |
| §4 Diebold–Mariano | `evaluate.dm_test` |
| App. A.3 architecture & grids | `network.BasisNetwork` defaults, `examples/run_empirical_study.py` |
| App. A.3 ensembles of 10 | `model.TVNNEnsemble` |
| Tables 1–2 | `tables.forecast_accuracy_table` |
| Figures 1–2 | `plots.plot_mse_by_horizon` |

### Typographical errors in the published equations

Found during the read, confirmed against the Supporting Information, and corrected here:

| # | Where | Issue | Confirmed by |
|---|---|---|---|
| T1 | eq. (1) | lag range printed `t−kx:t` (that is `kx+1` lags) and the target block subscripted `kx` not `ky` | `P = 24 = 4 + 5×4` |
| T2 | eq. (3) | `−½(log\|Ω\| + log σ²)` missing the `T` multiplier | App. A.1 derives `−T/2` in eq. (9) |
| T3 | eq. (8) | spurious leading minus; should be `= Σ log p(·)` | App. A.2 |
| T4 | eq. (8) | sum printed `Σ_{t=1}^{h}`; should run to `T` | App. A.2 |
| T5 | eq. (8) | **`log F_t` term missing** | *not resolved* — see §8 |
| T6 | Algorithm 1 | `arg min_θ Q` should be `arg max` | `Q` is an expected log-likelihood |
| T7 | Algorithm 1 | penalty written inside the gradient | should be `∇_w[log p − α‖w‖²]` |

---

## 8. Three things the paper leaves open

Rather than guess, the library implements both branches of each and documents the default.

### 8.1 The missing `log F_t` in eq. (8) — `loss=`

Eq. (8) as printed is

```
log p ∝ − Σ_t (y_{t+h} − z_t'b_t)² (σ² + z_t'V_t z_t)⁻¹
```

But the Gaussian log-density is `−½[log(2πF_t) + v_t²/F_t]`. Eq. (8) has **neither the `log F_t` nor the ½**. Appendix A.2 sets the decomposition up correctly, states `E[y_{t+h}|y_t] = z_t'b_t` and `Var = σ² + z_t'V_t z_t`, then writes *"By normality, (8) follows"* — asserting rather than deriving. Since `F_t` depends on `w` through `z_t`, the omission **changes the gradient**.

| `loss` | Objective | Note |
|---|---|---|
| `"weighted_sse"` *(default)* | `Σ v_t²/F_t` | Eq. (8) as printed. The prose — "gradient ascent … scaled by a filtered estimate of the variance" — says this is what was coded. |
| `"exact"` | `½ Σ [log(2πF_t) + v_t²/F_t]` | The true Gaussian marginal likelihood. |

Only the authors' `functions_lib.py` (listed in the Supporting Information manifest but distributed as a separate file) can settle which was used. Report both.

### 8.2 Which filtered state enters eq. (8) — `state=`

Algorithm 1 says `b_t = E[β_t | y_t]`, the *updated* filtered state. But observation `t` in this state space **is** `y_{t+h}`, so `β_{t|t}` has already absorbed the very value it is being used to predict — which would drive `F_t → 0`.

The textbook prediction error decomposition (Durbin & Koopman ch. 7.2, the reference Appendix A.2 follows) uses the **one-step-predicted** state `β_{t|t−1}`, which has not seen observation `t`. That is `state="predicted"`, the default, and it is the only reading under which eq. (8) is a prediction-error decomposition at all. `state="filtered"` is provided for comparison.

Note this is about the *training objective*. The out-of-sample side is unambiguous and handled by `gap=h` (step 8): you cannot use observations you do not have.

### 8.3 Identification — read shapes, not levels

The factorisation `(z_t, β_t)` is **not identified**. For any invertible `A ∈ ℝ^{d×d}`,

```
z → A'z ,   β → A⁻¹β ,   Ω → A⁻¹ Ω A⁻ᵀ
```

leaves the observation equation and the entire predictive distribution unchanged. Consequences:

* `β̂_t` carries no "the coefficient on feature *j* drifted" reading. **Read the shape** — when it moves and by how much — not the level or sign.
* `Ω̂` is defined relative to whatever basis the network happened to settle on. **Do not compare it across runs or seeds.**
* Point forecasting is completely unaffected, which is why the paper's results stand.
* At `d = 1` this collapses to a sign/scale indeterminacy, which is mild.

The paper concedes the symptom ("the black box nature of neural networks renders them difficult to interpret") without naming the cause. `plot_time_varying_weights` repeats the warning in its docstring.

---

## 8b. Two things that only show up on real data

Neither of these is visible on simulated data, and neither is mentioned in the paper. Both were found by running the estimator on FRED-MD and watching it fail. They are on by default, each is one keyword away from the paper's literal behaviour, and the numbers below are reproducible with `examples/run_empirical_study.py`.

### 8b.1 The signal-to-noise ratio runs away — `sn_cap`

Unconstrained maximum likelihood for a random-walk-coefficient model is free to attribute *all* persistence in the data to coefficient drift rather than to noise. On real macro data it does.

The paper's own design makes this sharper than usual. At horizon `h` the target is an **overlapping** difference, `y_{t+h} − y_t`, so its errors are MA(`h−1`) *by construction*. A drifting coefficient is an excellent way to absorb that serial correlation, and the likelihood happily takes it.

On FRED-MD unemployment at `h = 12`, the unconstrained EM returns

```
sigma2 = 0.0082 ,  Omega = 0.0905   ->   tr(Omega)/sigma2 ≈ 11
```

i.e. the coefficient fully re-randomises within a year. The Kalman gain approaches one, the state chases `y_t / z_t`, and propagating it `h` steps ahead produces this:

| `sn_cap` | `d` | POOS MSE | Forecast range |
|---|---|---|---|
| `None` (paper) | 1 | 3.13 | `[−9.60, 3.92]` |
| `None` (paper) | 2 | 10.11 | `[−18.41, 9.77]` |
| `0.01` (default) | 1 | **2.09** | `[−1.45, 2.83]` |
| `0.01` (default) | 2 | **2.03** | `[−2.41, 3.84]` |

*(UNRATE, `h = 12`, `P = 24`, POOS 2005–2014. Realised range `[−1.50, 4.00]`; AR benchmark MSE 1.52.)*

Bounding `tr(Ω)/(d·σ²)` is the standard control and is the frequentist analogue of the shrinkage prior on the signal-to-noise ratio in Goulet Coulombe (2020), *Time-Varying Parameters as Ridge Regressions*. It cuts the error roughly five-fold and brings the forecast range back inside the realised one.

The cap is **on in `TVNN` (0.01) and off in `em.run_em`** — the bare EM keeps its textbook guarantees, including the monotone likelihood, because a constrained M step no longer has them. Set `sn_cap=None` for the paper's literal model.

### 8b.2 The learned basis is unbounded — `normalize_basis`

The basis layer is linear, so `‖z‖` is unconstrained. Because the model is invariant to `z → A'z`, `β → A⁻¹β` (§8.3), nothing pins the scale down — and on FRED-MD the network drifts to `|z|` up to **42** while `β` shrinks to `0.02` to compensate. Mathematically identical; numerically awful, since the Kalman gain `K_t = P z_t / (z_t'P z_t + σ²)` becomes enormous.

`normalize_basis=True` rescales each column of `z_t` to unit standard deviation over the training sample. It is a **reparameterisation, not a restriction** — it picks a well-conditioned point on the same orbit. `|z|max` drops from 42.2 to 2.5. Set `False` for the literal reading.

### 8b.3 A fourth open question: how is the objective normalised?

Algorithm 1 optimises `log p(y | θ, w) − α‖w‖²`. A log-likelihood is a sum, but any positive rescaling of the data term leaves its maximiser unchanged while completely changing what `α` is worth — so the convention is load-bearing and **the paper never states it**. Three readings, all defensible, with measured consequences:

| Reading | Data term | Effect |
|---|---|---|
| Summed | `Σ_t v_t²/F_t` | `α ∈ [0.02, 0.64]` regularises by ~0.1% — not the "heavy regularization" A.3 describes. FFNN lands at 1.60× the AR benchmark (UNRATE, `h=12`). |
| **Averaged** *(used here)* | `mean_t v_t²/F_t` | Makes A.3's grid heavy, as described. FFNN improves to **1.25×**, TVNN 1.56× → **1.47×**. |
| Averaged and rescaled by `σ²` | `σ² · mean_t v_t²/F_t` | Puts the TVNN's term on the baselines' scale so `α` bites equally. Tried, and **worse**: INDPRO `h=6` went 1.59× → 1.83×. Rejected. |

The averaged reading is used because it is the only one under which A.3's own description of its penalty grid is true. One caveat remains and is not resolved: the `1/F` weighting still leaves the TVNN's objective on a different scale from the baselines' plain squared error, by roughly `1/σ²`, so `α` does not bite equally hard across models. Equalising it made things worse, so it is left alone and flagged here.

Only the authors' `functions_lib.py` can settle this.

### 8b.4 What this means for the paper

Every component is verified independently — the filter matches `statsmodels` to 1e-9, the EM recovers known-truth `σ²` and `Ω`, the alignment carries no look-ahead and is unit-tested — and on simulated data with genuine coefficient drift the TVNN beats an identically tuned FFNN by a factor of twelve (`DM = −8.3`, `p < 0.001`).

**On real FRED-MD data, this implementation does not reproduce the paper's result.** In the default configuration the TVNN is the *worst* of the six models on both targets, at 1.20–1.59× the AR benchmark on industrial production and 1.47–2.06× on unemployment, where the paper has it beating the benchmark at long horizons. The result is also unstable under the normalisation choice above: an earlier convention had the TVNN as the *best* neural model, significantly ahead of the FFNN (`DM p = 0.034`). A finding that flips sign under an unstated convention is not a finding, and it is reported here as such rather than presented as a reproduction.

Two explanations were tested and **ruled out**:

* *Undertrained networks.* "300 epochs" in Keras means ~5,700 minibatch updates, not 300 full-batch steps, so the networks here could have been 19× undertrained. Raising the budget makes them **worse** (UNRATE `h=12`, FFNN: 1.25× at 300 steps, 1.39× at 5,700). They are overfitting, not undertrained.
* *Too little regularisation.* Sweeping Appendix A.3's entire penalty grid rescues the FFNN — 1.251, 1.176, 1.049, **1.008**, 1.042, 1.084 for `α = 0.02 … 0.64` — bringing it to parity with the AR benchmark. It does nothing for the TVNN, which stays flat at 1.45–1.59 throughout. So the residual gap on UNRATE is not a tuning problem.

Three parts of the paper's protocol are implemented but off by default, because each multiplies runtime. They narrow the gap without closing it:

| Missing piece | Flag | Measured effect |
|---|---|---|
| 10-network ensembles (A.3) | `--ensemble 10` | UNRATE `h=12`: 1.70× → **1.41×** (`DM p = 0.075` vs a single network) |
| Per-period random search over the A.3 grid, validated on the previous 90 months | `--tune` | implemented; not measured at full scale |
| A pinned FRED-MD vintage | `--vintage 2015-01` | unknown; the paper's is unstated |

Note also that this negative result is specific to the paper's own two targets. On inflation and on the federal funds rate the method works, and on the latter it beats the fixed-weight network decisively — see [§8c](#8c-where-the-tvnn-actually-helps-other-targets-other-data).

What this library does claim, and tests, is narrower and firmer: each component is correct in isolation, the `h`-step alignment carries no look-ahead, and on data generated by the paper's own model the estimator recovers it. Reproducing the published forecast accuracy needs `functions_lib.py`.

---

## 8c. Where the TVNN actually helps: other targets, other data

The paper studies two series at four horizons, all with `h > 1`. That is a narrow slice, and running the estimator outside it changes the conclusion substantially. [`examples/other_targets_study.py`](https://github.com/merwanroudane/tvwnn/blob/main/examples/other_targets_study.py) sweeps six FRED-MD targets at `h ∈ {1, 3, 12}` with a single fixed configuration (`d=1`, `l2=0.02`, `P=24`, POOS 1990–2014):

```
POOS MSE relative to the AR benchmark, P=24, 1990-01 to 2014-12
==================================================
    target  model      h = 1      h = 3     h = 12
--------------------------------------------------
  CPIAUCSL     AR      1.000      1.000      1.000
  CPIAUCSL    ARX  0.815***<   0.788***      0.732
  CPIAUCSL   FFNN    0.869**  0.683***<  0.469***<
  CPIAUCSL   TVNN      0.951      0.874   0.500***
--------------------------------------------------
  FEDFUNDS     AR     1.000<     1.000<     1.000<
  FEDFUNDS    ARX   2.003***      1.518      1.018
  FEDFUNDS   FFNN   3.105***      1.887      1.722
  FEDFUNDS   TVNN   1.483***      1.037      1.191
--------------------------------------------------
    INDPRO     AR     1.000<     1.000<     1.000<
    INDPRO   FFNN      1.105     1.355*    1.340**
    INDPRO   TVNN    1.207**      1.490    1.395**
--------------------------------------------------
    PAYEMS     AR     1.000<     1.000<     1.000<
    PAYEMS   FFNN   1.585***   1.516***      1.465
    PAYEMS   TVNN   2.495***   2.031***    2.434**
--------------------------------------------------
   S&P 500     AR     1.000<     1.000<     1.000<
   S&P 500   FFNN    1.211**    1.391**      1.238
   S&P 500   TVNN    1.137**      1.096    1.528**
--------------------------------------------------
    UNRATE     AR      1.000      1.000     1.000<
    UNRATE    ARX     0.939<     0.923<      1.090
    UNRATE   FFNN      1.105      1.269      1.251
    UNRATE   TVNN      0.998      1.173      1.727
==================================================
```

*(ARX rows trimmed for INDPRO/PAYEMS/S&P 500; full table in `tables/other_targets.txt`.)*

Three things come out of this that the paper's two-series design cannot show.

**1. On inflation the method works, and works well.** On CPIAUCSL at `h = 12` the TVNN cuts the AR benchmark's error in half — `0.500`, `p < 0.01` — and every nonlinear model beats the benchmark at every horizon. This is consistent with Medeiros et al. (2021), who find machine learning pays off specifically for inflation. The paper never tries it.

**2. The paper's own claim reproduces — on a series with a real regime break.** On FEDFUNDS at `h = 1` the TVNN more than halves the fixed-weight FFNN's error, `1.483` against `3.105`, `DM p = 0.017`. That is exactly the "time-varying weights beat fixed weights" result the paper asserts and never tests, and it shows up where theory says it should: the federal funds rate over 1990–2014 contains the zero lower bound from December 2008, about as clean a structural break as macroeconomics offers. Where the parameters genuinely move, modelling them as moving helps.

**3. The overlapping target really is part of the problem.** On INDPRO the TVNN goes from `1.207` at `h = 1` to `1.490` at `h = 3`. At `h = 1` the target does not overlap and there is no induced MA structure for the random-walk state to absorb; the degradation as `h` grows is the mechanism of [§8b.1](#8b1-the-signal-to-noise-ratio-runs-away--sn_cap) showing up in forecast accuracy. It is not the whole story — the TVNN still loses to AR at `h = 1` on four of six targets — but it is a real component.

**Where it clearly does not help:** PAYEMS, badly and significantly (2.0–2.5× the benchmark), and the paper's own INDPRO and UNRATE. Payroll employment is smooth and strongly autoregressive; there is little parameter instability to exploit and a drifting coefficient is pure added variance.

The honest summary is that **the TVNN is not a general-purpose improvement over a fixed-weight network — it is a tool for series whose conditional relationship actually shifts.** That is a more useful statement than the paper's, and it is only visible once you look past two targets.

### Quarterly data

`fetch_fred_qd()` loads FRED-QD (McCracken & Ng 2020), the quarterly companion: ~233 complete series over 1960–2014, including the national-accounts aggregates that have no monthly counterpart (real GDP `GDPC1`, consumption `PCECC96`, the GDP deflator `GDPCTPI`). The file carries two metadata rows rather than one, which the loader handles; everything downstream is frequency-agnostic.

```python
from tvwnn import fetch_fred_qd, build_fred_dataset, TVNN

panel, tcodes = fetch_fred_qd(cache_dir="data")
ds = build_fred_dataset("GDPC1", h=4, n_factors=5,     # 4 quarters = 1 year
                        data=panel, tcodes=tcodes,
                        start="1960-01-01", end="2014-12-01")
TVNN(d=1, l2=0.02, random_state=0).fit(ds.U, ds.y)
```

`h = 4` quarters is the same economic horizon as `h = 12` months but only a third of the overlap, which makes it a natural further test of the mechanism in point 3.

### A note on target transforms

Section 4 of the paper defines targets only for its own two series. `target_kind_from_tcode` generalises the rule: series the McCracken–Ng codes treat in logs (4, 5, 6) get the log `h`-th difference, series in levels or differences (1, 2, 3, 7) get the plain `h`-period change. That reproduces the paper's two choices exactly — INDPRO is code 5, UNRATE is code 2 — and extends them to any target. Override with `target_kind=`.


---

## 9. Reproducing the paper's tables and figures

```bash
# ~2 minutes: one target, h = 12, P = 24
python examples/run_empirical_study.py --quick

# default: both targets, h in {3,6,9,12}, P in {24,64}, single networks
python examples/run_empirical_study.py

# the paper's full configuration — expect hours
python examples/run_empirical_study.py --predictor-sets 24 64 424 --ensemble 10
```

```bash
# the paper's own protocol: per-period random search over the Appendix A.3 grid,
# validated on the previous 90 months, redone at every re-estimation date
python examples/run_empirical_study.py --tune --tune-draws 8 --ensemble 10
```

Useful flags: `--targets`, `--horizons`, `--predictor-sets`, `--ensemble`, `--tune`, `--tune-draws`, `--poos-start/--poos-end`, `--retrain-every`, `--vintage`, `--seed`.

Without `--tune` a run uses one fixed configuration throughout — the A.3 *modal* selections, which is not the same thing as the paper's protocol. `--tune` costs roughly `tune_draws + 1` fits per model per re-estimation date.

Writes to `tables/` (`.txt`, `.tex`, `.csv`, plus pairwise DM and MCS) and `figures/` (`.png` + `.pdf` at 300 dpi).

**What will and will not match the paper.** The protocol, architecture, grids and alignment follow the paper and Appendix A.3. The *numbers* will differ, for reasons that are the paper's, not this library's: the FRED-MD vintage is unstated; hyperparameters were drawn by random search per evaluation period rather than fixed; and Tables 1–2 average 10-network ensembles. Expect the qualitative findings — the TVNN's edge concentrating at long horizons and small, PCA-compressed predictor sets — rather than three matching decimals.

---

## 10. Output gallery

**Tables** render to console and LaTeX from the same object. The layout — block rules, the horizon columns, the block minimum marked, the significance footer — follows Tables 1 and 2 of the paper:

```
POOS MSE relative to the AR benchmark: <target>, <start> to <end>
==========================================================================
    P     model        h = 3      h = 6      h = 9     h = 12
--------------------------------------------------------------------------
   24        AR        1.000      1.000      1.000      1.000
   24       ARX        <mse>      <mse>      <mse>      <mse>
   24      FFNN        <mse>      <mse>      <mse>      <mse>
   24       RNN        <mse>      <mse>      <mse>      <mse>
   24      LSTM        <mse>      <mse>      <mse>      <mse>
   24      TVNN        <mse>      <mse>      <mse>      <mse>
==========================================================================
Note: Significance of the Diebold-Mariano test against the AR benchmark:
* p<0.10, ** p<0.05, *** p<0.01.
'<' marks the lowest value in each block.
```

Entries carry Diebold–Mariano stars (`0.812**`), the block minimum is flagged with `<` in the console and bolded in LaTeX. **The numbers above are placeholders for the layout, not results** — run `examples/run_empirical_study.py` and read `tables/`. Actual figures for this implementation, and how they compare with the paper's, are discussed honestly in [§8b.4](#8b4-what-this-means-for-the-paper).

**Figures**, all on white, 300 dpi, PNG + vector PDF:

| Function | Shows |
|---|---|
| `plot_mse_by_horizon` | Figures 1–2: MSE against `h`, one line per model, TVNN highlighted |
| `plot_time_varying_weights` | `β̂_t` with a ±2 s.e. band and NBER recession shading |
| `plot_cssed` | cumulative squared error difference vs. benchmark — shows *when* the gain accrues, which period-average MSE hides |
| `plot_forecast_vs_actual` | forecasts over realisations, recessions shaded |
| `plot_convergence` | the EM/Adam alternation: objective, log-likelihood, variance components — the fastest way to spot a runaway signal-to-noise ratio |
| `plot_scree` | variance explained by the diffusion indices |

---

## 11. Testing

```bash
pytest tests/ -v
```

28 tests, ~25 s. The ones that carry weight:

| Test | Establishes |
|---|---|
| `test_filter_matches_statsmodels` | log-likelihood, innovations and filtered states match an independent implementation to ~1e-9 |
| `test_scalar_fast_path_matches_general_path` | the `d=1` optimisation is exact to 1e-10 |
| `test_em_recovers_the_truth` | EM recovers known `σ²` and `Ω` on a fixed design |
| `test_loglikelihood_increases_monotonically` | EM's defining guarantee — and the tripwire for a dropped lag-one covariance |
| `test_lag_one_covariance_is_not_negligible` | the cross term is material, not rounding |
| `test_supervised_pair_uses_only_past_predictors` | no look-ahead in the design matrix |
| `test_forecast_gap_reproduces_random_walk_inflation` | `gap=h` inflates the variance by exactly `h·Ω` |
| `test_tvnn_beats_a_fixed_weight_network_on_drifting_data` | the paper's central claim, on a DGP where it must hold |
| `test_dm_sign_convention_and_stars` | a negative DM statistic favours the first argument |

---

## 12. Practical guidance and troubleshooting

**Start small.** `d=1`, one hidden layer of 32, `n_outer=30`, `grad_steps=10`. That *is* the paper's selected configuration for most cases — this model is not improved by making it bigger.

**Set `gap=h` in every `forecast` call.** It is the single most consequential argument in the library. `poos_experiment` does it for you.

**Long-horizon forecasts explode.** The signal-to-noise ratio has run away; see [§8b.1](#8b1-the-signal-to-noise-ratio-runs-away--sn_cap). Check `summary()["omega_trace"] / summary()["sigma2"]` — anything above ~1 means the state is absorbing the overlapping target's serial correlation. Keep `sn_cap=0.01`, or lower it.

**The weights barely move (`tr(Ω̂)` ≈ 0).** Either there genuinely is no drift, or `l2` is so high the basis has collapsed toward a constant. Try `l2=0.02`; check `plot_convergence`.

**Sawtooth log-likelihood in `plot_convergence`.** The two loops are fighting. Lower `learning_rate`, or lower `grad_steps` so `θ` is refreshed more often.

**Training is slow.** `em_iter=50` × `n_outer=30` is up to 1500 filter+smoother passes. Lower `em_iter` to 20 — EM usually converges in far fewer — or keep `d=1` for the scalar fast path. `d=2` is roughly 10× slower per pass.

**Forecasts look implausibly good.** Check the alignment first. `y[t]` must be `y_{t+h}`, and the training set must stop `h` periods before the forecast origin. Use `build_supervised` / `poos_experiment` rather than hand-rolling it.

**`σ̂²` and `Ω̂` don't match my simulation's truth.** Expected: `standardize=True` puts them on a rescaled axis, and `Ω̂` lives in the learned basis (§8.3). To check variance components against truth, feed a *fixed* design to `run_em` directly — that is what `tests/test_em.py` does.

**FRED-MD gives me only ~40 series.** You are dropping columns before trimming the ragged final months. Use `transform_panel`, which handles it.

---

## 13. Citation

Cite the paper:

```bibtex
@article{rudd2026tvnn,
  author  = {Rudd, William and Bondell, Howard and Silver, Jeremy},
  title   = {Augmenting Neural Networks With Time-Varying Weights},
  journal = {Journal of Forecasting},
  volume  = {45},
  number  = {1},
  pages   = {22--28},
  year    = {2026},
  doi     = {10.1002/for.70014}
}
```

And, if this implementation was useful:

```bibtex
@software{roudane2026tvwnn,
  author  = {Roudane, Merwan},
  title   = {tvwnn: Time-Varying Weight Neural Networks in Python},
  year    = {2026},
  version = {1.0.0},
  url     = {https://pypi.org/project/tvwnn/},
  note    = {Python package; source at https://github.com/merwanroudane/tvwnn}
}
```

### Background references

* Holmes, E. E. (2013). *Derivation of an EM algorithm for constrained and unconstrained MARSS models.* [arXiv:1302.3919](https://arxiv.org/abs/1302.3919) — the M step.
* Durbin, J. & S. J. Koopman (2012). *Time Series Analysis by State Space Methods*, 2nd ed. OUP — filter, smoother, prediction error decomposition.
* McCracken, M. & S. Ng (2016). "FRED-MD: A Monthly Database for Macroeconomic Research." *JBES* 34(4): 574–589.
* Stock, J. & M. Watson (2002). "Macroeconomic Forecasting Using Diffusion Indices." *JBES* 20(2): 147–162.
* Diebold, F. X. & R. S. Mariano (1995). "Comparing Predictive Accuracy." *JBES* 13(3): 253–263.
* Harvey, D., S. Leybourne & P. Newbold (1997). "Testing the equality of prediction mean squared errors." *IJF* 13(2): 281–291.
* Hansen, P. R., A. Lunde & J. M. Nason (2011). "The Model Confidence Set." *Econometrica* 79(2): 453–497.
* Tran, M.-N., N. Nguyen, D. Nott & R. Kohn (2020). "Bayesian Deep Net GLM and GLMM." *JCGS* 29(1): 97–113.

---

## Licence

MIT. See [LICENSE](https://github.com/merwanroudane/tvwnn/blob/main/LICENSE).

© Dr Merwan Roudane · [merwanroudane920@gmail.com](mailto:merwanroudane920@gmail.com)
[github.com/merwanroudane/tvwnn](https://github.com/merwanroudane/tvwnn) · [pypi.org/project/tvwnn](https://pypi.org/project/tvwnn/)
