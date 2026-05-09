# Kernel pipeline — comprehensive pseudocode

Single reference for data flow, models, training, and evaluation (aligned with current code).

**Revision notes:** Simplified path uses `SimplifiedPricer` + physical `tau_scale` + optional `StrikeVolScale`. Full path uses **Eq. 6 (revised)** for the characteristic function \(z=v+i u_{\mathrm{im}}\) along the Carr–Madan contour:  
\[
\boxed{\Phi_T(z)= e^{\,i\,z\log S_0}\cdot e^{\,-\,\tfrac{i z+z^2}{2}\,A(T)}\cdot \exp\!\Bigl(\int_0^T \log \phi_{X_1}(\zeta(r))\,dr\Bigr),}
\]
with \(\zeta(r)=\bigl(z\,(-\,i\,)\,\bigr)\,\xi(r)\) (**linear** in \(z\)) and \(\xi(r)=\int_r^T h(s,r)\,ds\).  
**Correction (common slip):** the Lévy branch is **not** \(\log\phi_{X_1}\bigl(\frac{-u+i u^2}{2}\!\int_r^T h\,ds\bigr)\); that swaps \(\Re/\Im\) of \(z\) versus the Gaussian exponent and wrongly puts a **quadratic-in-\(z\)** term inside \(\phi_{X_{1}}\). The code matches the boxed \(\zeta(r)=z\,(-\,i\,)\,\xi(r)\). See §4.4.

---

## 1. Data loading and features (`preprocess.py`)

```
FUNCTION load_options_data(data_dir):
    FOR each CSV matching UnderlyingOptionsIntervals_*.csv:
        READ rows; PARSE quote_datetime, expiration as datetimes
    CONCATENATE all files
    FILTER option_type == "C" (calls)
    COMPUTE mid = (bid + ask) / 2 where both > 0; ELSE use close if configured
    DROP rows with missing mid
    RETURN DataFrame

FUNCTION to_training_arrays(df, S0_ref optional):
    t0 = (quote_datetime - min_quote_datetime) in years
    T  = (expiration - min_quote_datetime) in years
    tau = T - t0                              # years, time to maturity
    COMPUTE per-row underlying mid if available; BUILD S0_used per row
    k = log(strike / S0_row)                    # log-moneyness
    S0_global = mean(S0_used)                   # single spot for pricing
    RETURN arrays: tau, k, K, price_mid, S0_global

FUNCTION normalize_tau_for_net(tau, tau_max):
    RETURN tau / tau_max                        # in [0, 1] for the NNs

FUNCTION split_by_quote_date(df, train/val/test fractions):
    SPLIT by calendar day into train / val / test DataFrames
```

---

## 2. Kernel networks (`model.py`)

```
CLASS KernelNet:
    INIT(hidden_dims, input_scale, output_scale, nonnegative):
        MLP: R^2 -> hidden ReLU layers -> R^1

    FORWARD(first_arg, second_arg):   # parameter names "tau", "s" in code
        x = concat([first * input_scale, second * input_scale])
        raw = MLP(x)
        out = softplus(raw) IF nonnegative ELSE raw
        mask = 1 IF second > first ELSE 0       # causal / Volterra support
        RETURN out * mask * output_scale

CLASS StructuredKernelNetPaper7:     # optional --kernel-type paper7
    u = (second - first).clamp(floor)
    base = u^(d-1) * exp(-kappa * u)            # learnable d, kappa
    g = softplus(MLP([first, second])) + eps
    RETURN base * g * g_scale * mask(second > first)

CLASS StrikeVolScale:                # optional; disabled with --no-strike-scale
    INIT: small MLP: R^2 -> R^1
    FORWARD(tau_norm, log_k):
        RETURN softplus(MLP([tau_norm, log_k])) + 0.25   # positive multiplier

FUNCTION build_kernel_net(kernel_type, output_scale, ...):
    IF kernel_type == "paper7": RETURN StructuredKernelNetPaper7(...)
    ELSE: RETURN KernelNet(..., output_scale=output_scale, ...)

CLASS SimplifiedPricer:              # default training path in train.py
    INIT(...):
        self.kernel_net = build_kernel_net(...)
        self.strike_scale = StrikeVolScale() OR None

    FORWARD(tau_norm, K, S0, r, tau_scale, n_grid, n_u):
        RETURN model_call_prices(kernel_net, tau_norm, K, r, S0,
               n_grid, n_u, tau_scale, strike_vol_scale=self.strike_scale)
```

### 2.1 Kernel constraints (how structure is implemented)

These are the mechanisms that correspond to “constraints” on \(h(s,t)\) in code—not a separate constrained optimization layer:

- **Causal / Volterra support:** Both `KernelNet` and `StructuredKernelNetPaper7` multiply by a mask so \(h(s,t)=0\) when \(s \ge t\) (integration only over \(s < t\)).
- **Nonnegativity (default):** `KernelNet` applies `softplus` to the MLP output unless `--signed-kernel`. `paper7` uses a nonnegative base times nonnegative `g`.
- **Structural tail (`--kernel-type paper7`):** Wang–Xia-style factor \(u^{d-1} e^{-\kappa u}\) on \(u = t - s\), with learnable \(d,\kappa\) and positive \(g\) from an MLP (see `StructuredKernelNetPaper7` above).
- **\(u \to \infty\) / full half-line:** Not enforced by an extra penalty. Carr–Madan and kernel quadrature use **finite** \(u\) grids only; behavior at infinity is implicit in the chosen parameterization and grid extent.

---

## 3. Variance and Carr–Madan (`model.py`)

```
FUNCTION variance_from_kernel(kernel_net, tau_norm, n_grid):
    # Trapezoid over u in [0,1]: s = tau_norm * u, maturity = tau_norm
    FOR each batch element:
        h = kernel_net(s_flat, t_flat)   # order net(s, τ) so τ > s on [0, τ_norm]
    sigma_sq_norm = trapezoid( h^2 along u ) * tau_norm
    RETURN sigma_sq_norm  (batch vector)

FUNCTION model_call_prices(kernel_net, tau_norm, K, r, S0, n_grid, n_u,
                           tau_scale, strike_vol_scale optional):

    sigma_sq = variance_from_kernel(kernel_net, tau_norm, n_grid)
    sigma_sq = sigma_sq * tau_scale            # physical variance scale (τ_max in years)

    IF strike_vol_scale is not None:
        log_k = log(K / S0)
        m = strike_vol_scale(tau_norm, log_k)
        sigma_sq = sigma_sq * m

    tau_phys = tau_norm * tau_scale            # years, for discount & CF
    RETURN call_price_carr_madan(sigma_sq, K, tau_phys, r, S0, n_u)

FUNCTION call_price_carr_madan(sigma_sq, K, tau_phys, r, S0, alpha, u_max, n_u):
    k = log(K / S0)
    # Gaussian CF in Carr–Madan; see implementation for z_sq, denom, psi
    FOR u on grid [small, u_max]:
        BUILD complex integrand; take real part
    C = (exp(-alpha * k) / pi) * trapezoid(integrand)
    RETURN max(C, 0)
```

---

## 4. Full structural model (`model_full.py`)

### 4.1 `FullOptionModel`

```
CLASS FullOptionModel:
    CONTAINS kernel_net + learnable positive scalars (V0, V_bar, k, a, b, c)

    FORWARD(tau_norm, K, r, S0, tau_scale, n_r, n_s, n_u):
        RETURN call_price_carr_madan_full(
            kernel_net, tau_norm, K, V0, V_bar, k, a, b, c,
            r, S0, n_r, n_s, n_u, tau_scale)
```

### 4.2 Mean reversion (Eq. 2)

```
FUNCTION mean_reversion_A(tau_phys, V0, V_bar, k):
    RETURN V0/k * (1 - exp(-k*tau_phys)) + V_bar * (tau_phys - (1/k)*(1 - exp(-k*tau_phys)))
```

### 4.3 Tempered stable `log phi_{X_1}` (Eq. 4)

```
FUNCTION log_phi_X1_tempered_stable(z_re, z_im, a, b, c, Gamma(-c)):
    # log phi = a * Gamma(-c) * ((b - i*z)^c - b^c)
    COMPUTE (b - i*z)^c in complex; subtract b^c; multiply by a * Gamma(-c)
    RETURN (log_phi_re, log_phi_im)
```

### 4.4 Full characteristic function — **Eq. 6 (revised)**

**Symbols.** Write the contour point \(z=v+i u_{\mathrm{im}}\) (= `u_re` + \(i\cdot\)`u_im`) along the truncated \(v\)-line in Carr–Madan. **Do not** reuse the shorthand “\(iu+u^2\)” from a **real** Fourier argument inside \(\phi_{X_1}\); the Lévy leg always uses the **linear** complex argument \(\zeta(r)=z\,(-\,i\,)\,\xi(r)\).

**Wrong (misaligned \(\Re/\Im\) vs.\ \(\frac{iz+z^2}{2}\), and quadratic inside \(\phi_{X_{1}}\)):**

\[\exp\!\Bigl(\int_{0}^{T_{\mathrm{n}}}\log \phi_{X_1}\Bigl(\tfrac{-u+i u^{2}}{2}\!\int_{r}^{T_{\mathrm{n}}}h(s,r)\,ds\Bigr)\,dr\Bigr)\]

This does **not** match `characteristic_function_full`: it swaps real/imag parts relative to \(\tfrac{iz+z^2}{2}\) and puts a **quadratic** term into \(\phi_{X_{1}}\), whereas the Gaussian factor already carries \(\tfrac{iz+z^{2}}{2}\,A(T)\); \(\phi_{X_{1}}\) is evaluated at \(\zeta(r)=\mathcal{O}(z)\cdot\xi\) only.

**Correct (matches `model_full.characteristic_function_full`):**

\[
\Phi_T(z)=\underbrace{e^{\,i\,z\log S_0}}_{\texttt{t1}}\underbrace{e^{\,-\,w\,A(T)}}_{\texttt{t2}},\quad w=\tfrac{iz+z^{2}}{2},\quad 
\underbrace{e^{\,\int_{0}^{T_{\mathrm{n}}}\log \phi_{X_1}(\,z\,(-\,i\,)\,\xi(r)\,)\,dr}}_{\texttt{exp\_outer}} .
\]

**Implementation detail:** \(\phi_{X_{1}}\) expects input as real/imag pairs; `z_im * xi`, `-z_re * xi` is exactly \(z\cdot(-\,i\,)\cdot\xi\) in \(\mathbb{C}\).

```
FUNCTION characteristic_function_full(net, u_re, u_im, tau_norm, V0, V_bar, k, a, b, c,
                                      n_r, n_s, tau_scale, S0):

    tau_phys = tau_norm * tau_scale          # horizon in years (scalar per batch row in caller)
    A_T = mean_reversion_A(tau_phys, V0, V_bar, k)

    # --- Lévy / kernel leg ---
    FOR r on grid [0 .. tau_norm]:
        xi(r) = trapezoid( net(r, s) for s from r to tau_norm )   # h(s,r), s > r
    BUILD xi vector aligned with r-grid
    # Lévy argument zeta(r) = z * (-i) * xi(r)  with z = u_re + i*u_im  (complex multiply)
    # As real/imag: Re(z(-i))=u_im, Im(z(-i))=-u_re  multiplied by xi
    z_arg_re = u_im.unsqueeze(1) * xi.unsqueeze(0)
    z_arg_im = -u_re.unsqueeze(1) * xi.unsqueeze(0)
    outer = integral over r of log_phi_X1_tempered_stable(z_arg_re, z_arg_im, ...)
    exp_outer = exp(outer)   # complex

    # --- Deterministic Gaussian / mean-reversion leg ---
    # z = u_re + i*u_im  (Carr–Madan contour)
    t1 = exp(i * z * log(S0))
    w  = (i*z + z^2) / 2
    t2 = exp(-w * A_T)

    det = t1 * t2
    phi = det * exp_outer
    RETURN (phi_re, phi_im)
```

### 4.5 Carr–Madan with full `phi_T`

```
FUNCTION call_price_carr_madan_full(net, tau_norm, K, ..., S0, tau_scale, ...):
    FOR each batch index i:
        u_grid real part; u_im = -(alpha+1)   # standard Carr–Madan shift
        phi_re, phi_im = characteristic_function_full(..., S0=S0, tau=tau_norm[i], ...)
        rho = exp(-r * tau_phys) * phi_T / (denominator(u))    # complex algebra as in code
        integrand = Re( exp(-i * u * log(K/S0)) * rho )
        C_i = exp(-alpha * log(K/S0)) / pi * trapezoid(integrand)
    RETURN stack(C_i)
```

---

## 5. Training — simplified Gaussian path (`train.py`)

**Calibration objective (price term):** minimize one of:

- `mse`: \(\frac{1}{N}\sum (C - C^{\mathrm{mkt}})^2\)
- `relative_mse`: \(\frac{1}{N}\sum \bigl((C - C^{\mathrm{mkt}}) / (|C^{\mathrm{mkt}}|+\varepsilon)\bigr)^2\)
- `relative_mae`: \(\frac{1}{N}\sum \bigl|C - C^{\mathrm{mkt}}\bigr| / (|C^{\mathrm{mkt}}|+\varepsilon)\) — matches \(\sum_{K,T} |\cdot|\) up to batch stochasticity (full pass \(\approx\) global mean over quotes).

**CLI (selected):** `--loss {mse,relative_mse,relative_mae}`, `--relative-loss` (alias for `relative_mse`), `--output-scale`, `--signed-kernel`, `--no-strike-scale`, `--kernel-type {mlp,paper7}`, `--time-split`, `--max-samples`, `--save`.

```
MAIN:
    LOAD options CSVs -> DataFrame
    IF --time-split:
        split_by_quote_date -> train / val / test
        tau_max = max tau across splits
    ELSE:
        to_training_arrays -> tau, k, K, price; tau_max = max(tau)

    tau_norm = normalize_tau_for_net(tau, tau_max)
    BUILD TensorDataset(tau_norm, K, price)
    BUILD SimplifiedPricer(hidden, kernel_type, output_scale, use_strike_scale, ...)
    OPTIMIZER = Adam(pricer.parameters(), lr)

    FOR epoch = 1 .. epochs:
        FOR each batch (tau_b, K_b, price_b):
            C_model = pricer(tau_b, K_b, S0, r, tau_max, n_grid, n_u)
            loss = compute_price_loss(C_model, price_b, loss_kind)   # mse | relative_mse | relative_mae
            OPTIONAL: convexity / monotonicity penalties on sorted K within batch
            BACKPROP; STEP

        IF val set exists: metric = eval_loss on val
        ELSE: metric = train loss
        TRACK best_metric; SAVE best pricer.state_dict

    SAVE checkpoint:
        simplified_pricer_state, net_state (kernel only), S0, tau_max, args, best_metric
```

---

## 6. Training — full model (`train_full.py`)

Same `--loss` / `--relative-loss` options as `train.py` (shared `compute_price_loss`).

```
MAIN:
    SAME preprocess / tau_norm / tau_max as above
    kernel_net = build_kernel_net(...)
    full_model = FullOptionModel(kernel_net)
    OPTIMIZER = Adam(full_model.parameters())

    FOR epoch ...:
        FOR each batch:
            C_model = full_model(tau_b, K_b, r, S0, tau_scale=tau_max, n_r, n_s, n_u)
            loss = compute_price_loss(C_model, price_b, loss_kind)
            BACKPROP; STEP

    SAVE full_model_state, kernel_net_state, args, ...
```

---

## 7. Evaluation (`evaluate.py`)

```
MAIN:
    IF --full-model:
        LOAD FullOptionModel checkpoint
        C_model = full_model(tau_norm, K, ..., tau_scale=tau_max)
    ELSE:
        IF simplified_pricer_state EXISTS:
            LOAD SimplifiedPricer; C_model = pricer(tau_norm, K, S0, r, tau_max, ...)
        ELSE:  # legacy KernelNet only
            C_model = model_call_prices(net, tau_norm, K, ..., tau_scale=tau_max)

    COMPUTE metrics (MAE, RMSE, relative, max_ae) vs market mid
    PLOT market vs model; error by moneyness
    PLOT kernel heatmap; EXPORT kernel_grid.csv (kernel_net only)
```

---

## 8. Optional tools

```
inspect_kernel.py:
    LOAD kernel_net from checkpoint (unwrap SimplifiedPricer if needed)
    PRINT parameters; EVALUATE net(s, tau) at example normalized (s, tau)

run_pipeline.py:
    SUBPROCESS train.py [--epochs ...]
    SUBPROCESS evaluate.py --checkpoint ...
```

---

## 9. End-to-end data flow

```
CSV rows (calls)
    -> tau, k, K, mid price, S0
    -> tau_norm = tau / tau_max

SimplifiedPricer(tau_norm, K)
    -> kernel net on (s, tau_norm) grid -> sigma_sq_norm
    -> sigma_sq *= tau_max
    -> sigma_sq *= strike_scale(tau_norm, log(K/S0))   [if enabled]
    -> Carr–Madan(sigma_sq, K, tau_phys = tau_norm * tau_max)
    -> model call price C

LOSS: compare C to mid; BACKPROP through kernel_net + StrikeVolScale (if any)
```

---

## File map

| File | Role |
|------|------|
| `preprocess.py` | Load CSVs, mid, τ, k, K, normalize τ |
| `model.py` | KernelNet, Paper7, StrikeVolScale, SimplifiedPricer, variance, Gaussian Carr–Madan |
| `model_full.py` | FullOptionModel, **Eq. 6** `characteristic_function_full`, tempered-stable, full Carr–Madan |
| `train.py` | Fit `SimplifiedPricer`; save `simplified_pricer_state` |
| `train_full.py` | Fit `FullOptionModel` |
| `evaluate.py` | Metrics, plots, kernel grid; `--full-model` for full checkpoint |
| `inspect_kernel.py` | Print / probe `kernel_net` (unwraps `SimplifiedPricer` if needed) |
| `MATH_MODEL_MAPPING.md` | Equation ↔ code table |

---

## 10. One-page summary

```
DATA:   CSV -> mid, tau_years, K, k=log(K/S0), S0_global
        tau_norm = tau_years / tau_max

SIMPLIFIED (train.py):
        C = SimplifiedPricer(tau_norm, K; S0, r, tau_max)
            = Carr–Madan(  sigma^2 * tau_max * strike_scale(.) * 1_{kernel integral},
                           K, tau_phys = tau_norm * tau_max  )
        LOSS = MSE(C, market_mid)

FULL (train_full.py):
        phi_T from Eq.6 deterministic factor * Lévy factor kernel integral
        C = Carr–Madan full with phi_T
        LOSS = MSE(C, market_mid)
```
