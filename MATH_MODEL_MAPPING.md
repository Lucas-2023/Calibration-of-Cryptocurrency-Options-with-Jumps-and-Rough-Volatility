# Math model ↔ code mapping

This table shows which part of the **session-notes math model** is implemented in which **file** and **symbol**.

---

## Equation / concept → file(s) and code

| Math (session notes) | File(s) | Code location / symbol |
|----------------------|---------|-------------------------|
| **Eq. 1** — Asset price: \(dS_t = S_t\sqrt{V_t}\,dB_t\) | — | Not simulated; only the resulting **characteristic function** of \(\log S_T\) is used for pricing (Eq. 5–7). |
| **Eq. 2** — Variance: \(V_t = V_0 e^{-kt} + \bar{V}(1-e^{-kt}) + X_t^{(h)}\) | `model_full.py` | **Mean-reverting part** \(V_0 e^{-kt} + \bar{V}(1-e^{-kt})\) → `mean_reversion_A()`; **\(X_t^{(h)}\)** is encoded in the kernel integral inside **Eq. 5** (see below). |
| **Eq. 3** — \(X_t^{(h)} = \int_0^t h(t,s)\,dX_s\) (kernel-weighted Lévy) | `model_full.py` | The **kernel** \(h(s,r)\) is used in the double integral in **Eq. 5**: \(\xi(r) = \int_r^T h(s,r)\,ds\), then \(\int_0^T \log\phi_{X_1}(\cdots)\,dr\). Implemented in `characteristic_function_full()` (inner loop over `r`, integral over `s`). |
| **Eq. 4** — Tempered stable: \(\log\phi_{X_1}(u) = a\Gamma(-c)\bigl[(b-iu)^c - b^c\bigr]\) | `model_full.py` | `log_phi_X1_tempered_stable()`; helper `_complex_power_real_exp()` for \((b-iz)^c\). Parameters \(a,b,c\) are trainable in `FullOptionModel`. |
| **Eq. 6 (revised)** — \(\Phi_T(z)= e^{\,i z\log S_0-\frac{iz+z^2}{2}A(T)}\exp\bigl(\int_0^{\tau_n}\!\log\phi_{X_1}(z\,(-\,i\,)\,\xi(r))\,dr\bigr)\), \(\xi(r)=\int_r^{T_{\mathrm{n}}} h(s,r)\,ds\) | `model_full.py` | `characteristic_function_full()`: deterministic uses \(w=\tfrac{iz+z^2}{2}\); Lévy leg is **linear in** \(z\): `z_im*xi`, `-z_re*xi` \(\equiv\Re/\Im\{z\cdot(-\,i\,)\cdot\xi\}\). A common wrong draft \(\log\phi_{X_1}(\frac{-u+iu^2}{2}\cdots)\) is **not** implemented (see `PROJECT_PSEUDOCODE.md` §4.4). \(A\) from `mean_reversion_A()`. |
| **Eq. 6** — Carr–Madan: \(C(K,T) = \frac{e^{-\alpha\log K}}{\pi}\int_0^\infty e^{-iv\log K}\rho(v)\,dv\) | `model.py`, `model_full.py` | `model.py`: `call_price_carr_madan()` (Gaussian CF). `model_full.py`: `call_price_carr_madan_full()` (uses full \(\phi_T\) for \(\rho\)). |
| **Eq. 7** — \(\rho(v) = \frac{\phi_T(v-(\alpha+1)i)}{\alpha^2+\alpha-v^2+i(2\alpha+1)v}\) (with \(e^{-rT}\) discount) | `model.py`, `model_full.py` | Same denominator and discount in both; `model.py` uses Gaussian \(\phi\); `model_full.py` uses \(\phi_T\) from `characteristic_function_full()`. |
| **Kernel** \(h(t,s)\) / \(h(s,r)\) | `model.py`, `model_full.py` | **NN**: `model.KernelNet` — `forward(tau, s)` with causal mask. In the full model, \(h(s,r)\) is evaluated as `net(r, s)` for \(s>r\) inside `characteristic_function_full()`. |

---

## File → math / role

| File | Role |
|------|------|
| **model.py** | **KernelNet** (kernel \(h\)), **Gaussian** variance \(\sigma^2(\tau)=\int h^2\), **Carr–Madan** with \(\phi(u)=e^{-\sigma^2 u^2/2}\), and **model_call_prices()** (simplified model). |
| **model_full.py** | **Full math model**: tempered stable (Eq. 4), mean reversion \(A(T)\) (Eq. 2), double integral \(\xi(r)\) and full \(\phi_T\) (Eq. 3 + 5), **Carr–Madan** with full \(\phi_T\) (Eq. 6–7), **FullOptionModel** (trainable kernel + \(V_0,\bar{V},k,a,b,c\)). |
| **preprocess.py** | Data → \(\tau\), \(K\), mid price, \(S_0\); no equations. |
| **train.py** | Trains **simplified** model: loss = MSE(model_call_prices, market). |
| **train_full.py** | Trains **full** model: loss = MSE(FullOptionModel(...), market). |
| **evaluate.py** | Loads checkpoint, computes metrics and plots (kernel heatmap, market vs model); works with either simplified or full checkpoint if you add full-model loading. |

---

## Summary

- **Simplified (Gaussian) path:** `model.py` + `train.py` — kernel \(h\) → \(\sigma^2\) → Gaussian \(\phi\) → Carr–Madan.
- **Full math path:** `model_full.py` + `train_full.py` — kernel \(h\) + mean reversion + tempered stable → full \(\phi_T\) (Eq. 5) → Carr–Madan (Eq. 6–7).

All of the session-notes equations (2–7) are reflected in **model_full.py** and **train_full.py** as in the table above.

---

## Implementation fixes (consistency pass)

| Issue | Fix |
|--------|-----|
| **τ double-scaling** | **preprocess.py:** `tau = T - t0` only (both already in years). |
| **Full model scalars not in autograd** | **model_full.py:** `params_tensors()` returns 0-dim tensors; **no `.item()`** in `forward`. `get_params()` is for logging only. `Gamma(-c)` still uses `math.gamma` from detached `c` (no grad through Γ). |
| **Best checkpoint** | **train.py** / **train_full.py:** save `best_state` when metric improves; reload before final eval and save file. |
| **Train/val/test split** | **preprocess.split_by_quote_date**; **train.py** / **train_full.py** `--time-split` (default 70/15/15); best on val when split enabled; test metrics at end. |
| **Evaluate full model** | **evaluate.py** `--full-model` loads `full_model_state` and prices with `FullOptionModel`. |
