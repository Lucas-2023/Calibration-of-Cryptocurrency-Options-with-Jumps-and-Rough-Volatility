# Option pricing kernel learning pipeline

This directory implements a pipeline to **learn the kernel \(h(\tau, s)\) of an option pricing model** from option data (e.g. VIX options in `data/`). The NN approximates the kernel; the pricing operator converts kernel → variance → characteristic function → Carr–Madan → call price. Loss is the error between model and market prices.

## Pipeline steps (overview)

1. **Preprocess** — Load CSVs, mid price = (bid+ask)/2, time to maturity **τ = (T−t₀) in years** (t₀ and T are already converted from seconds to years; do not divide τ by 365.25 again), log-moneyness k = log(K/S₀), reference spot S₀.
2. **Model** — Neural net: input (τ, s) → output h(τ, s) with causal mask (h = 0 for s ≤ τ).
3. **Pricing** — h → σ²(τ) = ∫₀^τ h(τ,s)² ds → Gaussian CF → Carr–Madan → C(K, τ).
4. **Training** — Loss = MSE (or relative MSE) between model and market prices; optional no-arbitrage penalties (convexity, monotonicity).
5. **Evaluation** — Metrics (MAE, RMSE, relative error), plots: market vs model price, error by moneyness, kernel heatmap.

## Setup

From the project root or from `kernel_pipeline`:

```bash
pip install -r kernel_pipeline/requirements.txt
```

Data: place `UnderlyingOptionsIntervals_*.csv` in `data/` (or set `--data-dir`).

## Usage

### Full pipeline (train then evaluate)

```bash
cd kernel_pipeline
python run_pipeline.py --data-dir ../data --epochs 2000
```

Output: `output/best_kernel.pt`, `output/market_vs_model.png`, `output/error_by_moneyness.png`, `output/kernel_heatmap.png`, `output/metrics.txt`.

### Train only

```bash
python train.py --data-dir ../data --epochs 2000 --batch-size 256 --lr 1e-3 --save output/best_kernel.pt
```

Use `--relative-loss` for relative MSE. Optional penalties: `--lambda-convex`, `--lambda-monotone`.

**Kernel types (`--kernel-type`):**

- **`mlp`** (default) — `KernelNet`; optional nonnegative output via softplus.
- **`paper7`** — Wang–Xia-style: `h = u^{d-1} e^{-κ u} · g(r,s)` with nonnegative `g` from an MLP, learnable `κ > 0`, and `d` either learned in `(paper7_d_low, paper7_d_high)` or fixed with **`--paper7-d-fixed`**.
- **`constant`** — fixed kernel value on the active mask: **`h = constant_h`** where `s > r` (full model) or `τ > s` (Gaussian path in `train.py`); no kernel NN parameters. Use **`--constant-h`** (default `0.6`).

**Positive MLP weights/biases (optional):** **`--positive-linear-wb`** uses a softplus parametrization so effective linear weights and biases in the kernel (and strike-scale MLP in `train.py`) are **> 0** in the forward map. Combine with **`--check-linear-wb-positive`** to print minima at log epochs.

**Strict 10-day call exports:** **`--strict-10days-data`** loads only `final_call_no_madan_strict_10days_*.csv` from `--data-dir`.

**Time-based train/val/test** (by quote date, default 70/15/15): add **`--time-split`**. The **best** weights (lowest val loss) are saved, not the last epoch.

**Recording convergence (`train_full.py`):** **`--convergence-csv output/full_model_convergence.csv`** appends **every epoch**: train loss, val loss (if time-split), selection metric, best metric so far, and scalars `V0`, `V_bar`, `k`, `a`, `b`, `c`. Plot with:

```bash
python plot_full_model_convergence.py --csv output/full_model_convergence.csv --out output/convergence_plot.png
```

**Kernel `h` movement (empirical sensitivity):** with **`--log-h-sensitivity-every N`** (and epoch 1), logs how much `h` changes on a fixed triangle over one epoch; optional **`--h-sensitivity-csv path`** writes the same metrics to CSV for plotting. Resolution: **`--h-sensitivity-n-r`**, **`--h-sensitivity-n-s`**. Implemented in `sensitivity_h.py` (full model: `net(r,s)` with `s > r`; `train.py`: `net(s,τ)` with `τ > s`).

### Full math model (Lévy / tempered stable)

```bash
python train_full.py --data-dir ../data --epochs 100 --save output/full_model.pt
```

Scalar parameters (V₀, V̄, k, a, b, c) are trained **in the autograd graph** (no `.item()` in the forward). Use **`--time-split`** like `train.py`.

**Checkpoint paths:** relative **`--save`** / **`--convergence-csv`** / **`--h-sensitivity-csv`** paths are resolved from the directory containing `train_full.py` (so they work even if your shell cwd differs). `torch.save` uses a resolved string path for compatibility on Windows.

**Example (strict 10-day data, constant kernel, convergence log):**

```bash
python train_full.py --data-dir . --strict-10days-data --kernel-type constant --constant-h 0.6 \
  --time-split --loss relative_mae --epochs 1200 --convergence-csv output/full_model_convergence.csv \
  --save output/full_model.pt
```

### Evaluate a saved model

```bash
python evaluate.py --checkpoint output/best_kernel.pt --data-dir ../data --out-dir output
```

**Full-model checkpoint** (from `train_full.py`):

```bash
python evaluate.py --full-model --checkpoint output/full_model.pt --data-dir ../data --out-dir output
```

Plots/metrics get a `_full` suffix when using `--full-model`.

### Preprocess only (sanity check)

```bash
python preprocess.py
```

(Expects `data/` one level up; or set `DATA_DIR` in the script.)

## Files

| File | Role |
|------|------|
| `preprocess.py` | Load CSVs, compute mid, τ, k, S₀; `to_training_arrays()` |
| `model.py` | `KernelNet`, `StructuredKernelNetPaper7`, `ConstantKernelNet`, `PositiveLinear`, `variance_from_kernel`, Carr–Madan, `model_call_prices()`, `build_kernel_net()` |
| `model_full.py` | Full φ_T, tempered stable, `FullOptionModel`, `call_price_carr_madan_full()` |
| `sensitivity_h.py` | Fixed-grid `h` metrics between epochs; optional CSV append for `train_full` / `train` |
| `train.py` | Training loop, best-checkpoint save, optional `--time-split`, kernel types above |
| `train_full.py` | Full model training, best checkpoint, `--convergence-csv`, h-sensitivity flags |
| `evaluate.py` | Simplified or `--full-model` evaluation, metrics, plots (loads `kernel_type`, `constant_h`, `positive_linear_wb` from checkpoint) |
| `plot_full_model_convergence.py` | Plots loss + scalars from `--convergence-csv` |
| `inspect_kernel.py` | Print kernel structure / weights (respects checkpoint args) |
| `MATH_MODEL_MAPPING.md` | Equation ↔ code map + fix notes |
| `run_pipeline.py` | Runs train then evaluate |
| `requirements.txt` | pandas, numpy, torch, matplotlib |

## Hyperparameters (typical)

- **Optimizer:** Adam, lr = 1e-3  
- **Batch size:** 256  
- **Epochs:** 500–2000  
- **Architecture:** FC(2→64)→ReLU→FC(64→64)→ReLU→FC(64→64)→ReLU→FC(64→1)  
- **n_grid:** 64 (variance integral), **n_u:** 256 (Carr–Madan)

## What the learned object represents

The NN learns **h(τ, s)** which drives:

- **σ²(τ)** — variance of log-return to maturity  
- **Risk-neutral distribution** via Gaussian CF and Carr–Madan  
- **Option surface** — model prices C(K, τ) consistent with the kernel  

So: **market option prices → kernel h → full option surface**.
