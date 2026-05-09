"""
Step 11: Evaluation and plots — market vs model price, error metrics, kernel visualization
(h(τ,s), Type I / Eq. (7) checks, asymptotic tails, and support/nonnegativity diagnostics).

Usage:
  python evaluate.py --checkpoint path/to/checkpoint.pt --data-dir ../data --out-dir ./out
  python evaluate.py ... --timestamp-out-dir   # writes to output/eval_YYYYmmdd_HHMMSS/ (no overwrite)
"""
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import (
    load_options_data,
    to_training_arrays,
    normalize_tau_for_net,
    resolve_synthetic_call_for_puts,
    apply_strike_band_by_spot_scalar,
    DATA_DIR,
)
from model import (
    KernelNet,
    SimplifiedPricer,
    StructuredKernelNetPaper7,
    build_kernel_net,
    model_call_prices,
)
from model_full import FullOptionModel

# (tt, ss, h) from _kernel_grid_h: net(s,τ) grid, valid where τ > s (same as variance integral).
KernelGrid = tuple[np.ndarray, np.ndarray, np.ndarray]


def _mask_h_abs_leq(h: np.ndarray, h_abs_min: float) -> np.ndarray:
    """Copy of ``h`` with NaN where ``|h| <= h_abs_min`` (strict: show only ``|h| > h_abs_min``). No-op if ``h_abs_min <= 0``."""
    if h_abs_min <= 0:
        return np.asarray(h, dtype=float)
    out = np.asarray(h, dtype=float).copy()
    out[np.abs(out) <= float(h_abs_min)] = np.nan
    return out

SCRIPT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _kernel_nonnegative_from_saved_args(ckpt_args: dict) -> bool:
    """paper7 / constant h>0 are nonnegative; mlp defaults False if key missing (old checkpoints)."""
    if ckpt_args.get("kernel_type") in ("paper7", "constant"):
        return True
    return bool(ckpt_args.get("kernel_nonnegative", False))


def _paper7_learned_d_float(net: torch.nn.Module) -> float | None:
    """Current ``d`` in ``u^{d-1}`` for ``StructuredKernelNetPaper7``; else ``None``."""
    if isinstance(net, StructuredKernelNetPaper7):
        if getattr(net, "d_fixed", None) is not None:
            return float(net.d_fixed)
        with torch.no_grad():
            span = net.d_high - net.d_low
            return float(net.d_low + span * torch.sigmoid(net.raw_d))
    return None


def _paper7_d_bounds_from_saved_args(ckpt_args: dict) -> tuple[float, float]:
    """Older checkpoints omit these → default (0.5, 2), matching former hard-coded paper7 range."""
    return (
        float(ckpt_args.get("paper7_d_low", 0.5)),
        float(ckpt_args.get("paper7_d_high", 2.0)),
    )


def load_checkpoint_simple(
    path: Path,
) -> tuple[SimplifiedPricer | KernelNet, float, float, dict, dict | None]:
    """Load simplified model: new ``SimplifiedPricer`` or legacy ``KernelNet`` only.

    Returns ``strike_band`` snapshot from training (if any): rel_lo, rel_hi, spot_scalar, etc.
    """
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    S0 = ckpt.get("S0", 1.0)
    tau_max = ckpt.get("tau_max", 1.0)
    args = ckpt.get("args", {})
    hidden = args.get("hidden", [64, 64, 64])
    ktype = args.get("kernel_type", "mlp")
    out_sc = float(args.get("output_scale", 0.2))
    use_strike = bool(args.get("use_strike_scale", True))

    d_lo, d_hi = _paper7_d_bounds_from_saved_args(args)
    d_fix = args.get("paper7_d_fixed")
    d_fix_f = float(d_fix) if d_fix is not None else None
    pos_wb = bool(args.get("positive_linear_wb", False))
    c_h = float(args.get("constant_h", 0.6))
    if ckpt.get("simplified_pricer_state") is not None:
        pricer = SimplifiedPricer(
            hidden_dims=hidden,
            kernel_type=ktype,
            kernel_nonnegative=_kernel_nonnegative_from_saved_args(args),
            output_scale=out_sc,
            use_strike_scale=use_strike,
            device=DEVICE,
            paper7_d_low=d_lo,
            paper7_d_high=d_hi,
            paper7_d_fixed=d_fix_f,
            positive_linear_wb=pos_wb,
            constant_h=c_h,
        )
        pricer.load_state_dict(ckpt["simplified_pricer_state"])
        pricer.eval()
        return pricer, S0, tau_max, args, ckpt.get("strike_band")

    net = build_kernel_net(
        hidden_dims=hidden,
        kernel_type=ktype,
        kernel_nonnegative=_kernel_nonnegative_from_saved_args(args),
        output_scale=out_sc,
        device=DEVICE,
        paper7_d_low=d_lo,
        paper7_d_high=d_hi,
        paper7_d_fixed=d_fix_f,
        positive_linear_wb=pos_wb,
        constant_h=c_h,
    )
    state = ckpt.get("net_state")
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    return net, S0, tau_max, args, ckpt.get("strike_band")


def load_checkpoint_full(path: Path) -> tuple[FullOptionModel, float, float, dict, dict | None]:
    """Load full model: full_model_state (or kernel_net_state + full weights)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    S0 = ckpt.get("S0", 1.0)
    tau_max = ckpt.get("tau_max", 1.0)
    args = ckpt.get("args", {})
    hidden = args.get("hidden", [64, 64, 64])
    ktype = args.get("kernel_type", "mlp")
    d_lo, d_hi = _paper7_d_bounds_from_saved_args(args)
    d_fix = args.get("paper7_d_fixed")
    d_fix_f = float(d_fix) if d_fix is not None else None
    pos_wb = bool(args.get("positive_linear_wb", False))
    c_h = float(args.get("constant_h", 0.6))
    kernel_net = build_kernel_net(
        hidden_dims=hidden,
        kernel_type=ktype,
        kernel_nonnegative=_kernel_nonnegative_from_saved_args(args),
        output_scale=float(args.get("output_scale", 0.2)),
        device=DEVICE,
        paper7_d_low=d_lo,
        paper7_d_high=d_hi,
        paper7_d_fixed=d_fix_f,
        positive_linear_wb=pos_wb,
        constant_h=c_h,
    )
    full_model = FullOptionModel(kernel_net).to(DEVICE)
    fstate = ckpt.get("full_model_state")
    if fstate is not None:
        full_model.load_state_dict(fstate)
    else:
        kstate = ckpt.get("kernel_net_state")
        if kstate is not None:
            kernel_net.load_state_dict(kstate)
    full_model.eval()
    return full_model, S0, tau_max, args, ckpt.get("strike_band")


def compute_metrics(C_model: np.ndarray, C_market: np.ndarray) -> dict:
    """MAE, RMSE, relative MAE, relative RMSE, max absolute error."""
    diff = C_model - C_market
    denom = np.abs(C_market) + 1e-8
    return {
        "mae": np.abs(diff).mean(),
        "rmse": np.sqrt((diff ** 2).mean()),
        "rel_mae": (np.abs(diff) / denom).mean(),
        "rel_rmse": np.sqrt(((diff / denom) ** 2).mean()),
        "max_ae": np.abs(diff).max(),
    }


def plot_market_vs_model(
    K: np.ndarray,
    tau: np.ndarray,
    C_market: np.ndarray,
    C_model: np.ndarray,
    out_path: Path,
    max_points: int = 2000,
) -> None:
    """Scatter: market price vs model price; ideal = 45° line."""
    n = min(len(C_market), max_points)
    idx = np.random.choice(len(C_market), n, replace=False) if len(C_market) > n else np.arange(len(C_market))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(C_market[idx], C_model[idx], alpha=0.3, s=5, c="tab:blue", label="options")
    lim = max(C_market.max(), C_model.max(), 1e-6)
    ax.plot([0, lim], [0, lim], "k--", label="ideal")
    ax.set_xlabel("Market price")
    ax.set_ylabel("Model price")
    ax.set_title("Market vs model call price")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _subsample_strike_comparison(
    K_sorted: np.ndarray,
    cm_sorted: np.ndarray,
    cp_sorted: np.ndarray,
    max_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce points for call vs strike plots: pick at most max_n strikes, spread along K
    (after sorting) so market vs model comparison stays readable.
    """
    n = len(K_sorted)
    if n <= max_n:
        return K_sorted, cm_sorted, cp_sorted
    idx = np.linspace(0, n - 1, max_n, dtype=int)
    idx = np.unique(idx)
    return K_sorted[idx], cm_sorted[idx], cp_sorted[idx]


def plot_call_price_vs_strike(
    K: np.ndarray,
    tau_phys: np.ndarray,
    C_market: np.ndarray,
    C_model: np.ndarray,
    out_path: Path,
    n_panels: int = 3,
    min_per_panel: int = 10,
    max_points_total: int = 3000,
    tau_bucket_edges: list[float] | None = None,
    y_price_max: float | None = None,
) -> None:
    """
    Call price vs strike for a few maturity buckets (τ in years).

    Interpreting the figure: for European calls at fixed maturity, C(K) decreases in K
    and is convex (no-arbitrage). Comparing curves shows how well the model matches
    the market slice; mixing all maturities on one axis would blur the picture.

    Rows per bucket are given in titles as ``rows``; scatter points are subsampled to at most
    ``max_points_total`` split across visible panels (``plotted`` in titles), sorted by ``K``,
    so market vs model stay comparable without overcrowding huge datasets.

    tau_bucket_edges:
        If set, panels are ``[e0,e1), [e1,e2), ...`` with the last interval closed on the right.
        Values are **years**, same units as ``tau_phys``. If ``None``, edges are **quantiles**
        of ``tau_phys`` (equal-count buckets, default ``n_panels=3`` → tertiles).

    y_price_max:
        If set, all panels use ``ylim(0, y_price_max)`` (shared y) so large ITM premiums
        do not squash the rest of the curve; values above are clipped in the viewport only.
    """
    t = np.asarray(tau_phys, dtype=np.float64)
    if tau_bucket_edges is not None:
        edges = np.asarray(sorted(float(x) for x in tau_bucket_edges), dtype=np.float64)
        if edges.size < 2:
            raise ValueError("tau_bucket_edges needs at least two values")
        if edges.size > 17:
            edges = edges[:17]
        n_panels = int(edges.size - 1)
    else:
        n_panels = max(1, min(n_panels, 5))
        q = np.linspace(0, 1, n_panels + 1)
        edges = np.quantile(t, q)
        # avoid empty bins: merge tiny ranges
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([t.min(), t.max()])
        n_panels = min(n_panels, len(edges) - 1)

    panel_masks: list[np.ndarray] = []
    for j in range(n_panels):
        lo, hi = edges[j], edges[j + 1]
        if j == n_panels - 1:
            mask = (t >= lo) & (t <= hi)
        else:
            mask = (t >= lo) & (t < hi)
        panel_masks.append(mask)
    n_active = sum(int(m.sum()) >= min_per_panel for m in panel_masks)
    per_panel = max(1, max_points_total // max(n_active, 1))

    if n_panels <= 4:
        nrows, ncols = 1, max(n_panels, 1)
    else:
        ncols = min(4, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
    fig, axes_arr = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharey=True, squeeze=False,
    )
    axes = axes_arr.flatten().tolist()
    for j in range(len(axes)):
        if j >= n_panels:
            axes[j].set_visible(False)
    for ax, j in zip(axes, range(n_panels)):
        mask = panel_masks[j]
        lo, hi = edges[j], edges[j + 1]
        n_in = int(mask.sum())
        if n_in < min_per_panel:
            ax.set_visible(False)
            continue
        Kb = K[mask]
        cm = C_market[mask]
        cp = C_model[mask]
        order = np.argsort(Kb)
        Ks = Kb[order]
        cm_o = cm[order]
        cp_o = cp[order]
        Ks, cm_o, cp_o = _subsample_strike_comparison(Ks, cm_o, cp_o, per_panel)
        n_plot = int(len(Ks))
        ax.scatter(Ks, cm_o, s=14, alpha=0.65, c="tab:blue", label="Market")
        ax.scatter(Ks, cp_o, s=14, alpha=0.65, c="tab:orange", label="Model")
        t_mid = 0.5 * (float(lo) + float(hi))
        ax.set_title(
            f"τ ≈ {t_mid:.3f} y\n[{lo:.3f}, {hi:.3f}]  (rows={n_in}, plotted={n_plot})",
        )
        ax.set_xlabel("Strike K")
        if j == 0:
            ax.set_ylabel("Call price")
        ax.legend(loc="upper right", fontsize=8)
    if y_price_max is not None and float(y_price_max) > 0:
        ymax = float(y_price_max)
        for ax in axes:
            if ax.get_visible():
                ax.set_ylim(0.0, ymax)
                break

    title = "Call price vs strike (by maturity bucket)"
    if tau_bucket_edges is not None:
        title += " — fixed τ edges (years)"
    if y_price_max is not None and float(y_price_max) > 0:
        title += rf" — $0 \leq C \leq {float(y_price_max):g}$ (y-axis cap)"
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_errors_by_moneyness(
    k: np.ndarray,
    C_market: np.ndarray,
    C_model: np.ndarray,
    out_path: Path,
    n_bins: int = 30,
) -> None:
    """Mean absolute error vs log-moneyness bins."""
    rel_err = np.abs(C_model - C_market) / (np.abs(C_market) + 1e-8)
    k_min, k_max = k.min(), k.max()
    bins = np.linspace(k_min, k_max, n_bins + 1)
    bin_idx = np.searchsorted(bins[1:-1], k)  # which bin
    bin_mae = []
    bin_center = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() > 0:
            bin_mae.append(rel_err[mask].mean())
            bin_center.append((bins[i] + bins[i + 1]) / 2)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(bin_center, bin_mae, width=(k_max - k_min) / n_bins * 0.8, align="center", color="steelblue", edgecolor="navy")
    ax.set_xlabel("Log-moneyness k = log(K/S0)")
    ax.set_ylabel("Mean relative error")
    ax.set_title("Relative pricing error by log-moneyness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _kernel_grid_h(
    net: KernelNet,
    n_tau: int = 50,
    n_s: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate h on a (τ, s) grid in normalized units. Same convention as variance_from_kernel:
    net(s, τ) with τ > s on the support.
    Returns tt, ss (meshgrid arrays in [0,1]), and h shaped (n_tau, n_s).
    """
    device = next(net.parameters()).device
    tau_norm = np.linspace(0.05, 1.0, n_tau).astype(np.float32)
    s_norm = np.linspace(0.0, 1.0, n_s).astype(np.float32)
    tt, ss = np.meshgrid(tau_norm, s_norm, indexing="ij")
    tau_flat = torch.tensor(tt.ravel(), device=device, dtype=torch.float32)
    s_flat = torch.tensor(ss.ravel(), device=device, dtype=torch.float32)
    with torch.no_grad():
        h_flat = net(s_flat, tau_flat)
    h = h_flat.cpu().numpy().reshape(n_tau, n_s)
    return tt, ss, h


def export_kernel_grid(
    net: KernelNet,
    tau_max: float,
    out_path: Path,
    n_tau: int = 50,
    n_s: int = 50,
    grid: KernelGrid | None = None,
) -> None:
    """Evaluate h(τ, s) on a grid and save to CSV (tau, s, h)."""
    if grid is None:
        tt, ss, h = _kernel_grid_h(net, n_tau=n_tau, n_s=n_s)
    else:
        tt, ss, h = grid
    h_flat = h.ravel()
    tau_phys_grid = (tt.astype(np.float64) * tau_max).ravel()
    s_phys_grid = (ss.astype(np.float64) * tau_max).ravel()
    header = "tau,s,h"
    lines = [header] + [f"{a:.8f},{b:.8f},{c:.8f}" for a, b, c in zip(tau_phys_grid, s_phys_grid, h_flat)]
    out_path.write_text("\n".join(lines))
    print(f"Kernel grid saved to {out_path} ({len(h_flat)} points)")


def plot_kernel_heatmap(
    net: KernelNet,
    tau_max: float,
    out_path: Path,
    n_tau: int = 50,
    n_s: int = 50,
    grid: KernelGrid | None = None,
    h_abs_plot_min: float = 0.0,
) -> None:
    """Visualize h(τ, s) on [0, tau_max] x [0, tau_max] (using normalized τ for net)."""
    if grid is None:
        tt, ss, h = _kernel_grid_h(net, n_tau=n_tau, n_s=n_s)
    else:
        tt, ss, h = grid
    h_plot = _mask_h_abs_leq(h, h_abs_plot_min)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.pcolormesh(
        tt * tau_max, ss * tau_max, h_plot,
        shading="auto", cmap="viridis",
    )
    ax.set_xlabel("τ (time to maturity, years)")
    ax.set_ylabel("s (integration variable, years)")
    title = "Kernel h(τ, s)"
    if h_abs_plot_min > 0:
        title += rf" ($|h| > {h_abs_plot_min:g}$ only)"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="h")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_kernel_abs_h_t_plus_u(
    net: torch.nn.Module,
    tau_max: float,
    out_path: Path,
    t_fixed_years: list[float] | None = None,
    n_u: int = 512,
    h_abs_plot_min: float = 0.0,
) -> None:
    """
    Line plot of ``|h(t+u, t)|`` vs lag ``u`` in **years** (not a heatmap).

    Matches the Volterra / variance-path convention in this repo: ``net(s, τ)`` with
    ``τ > s``, ``u = τ − s``, so for fixed calendar time ``t`` (here: inner time ``s = t``)
    and maturity ``τ = t + u`` we evaluate ``net(t_norm, τ_norm)`` and plot ``|h|`` vs ``u``.

    If ``t_fixed_years`` is None, uses several default slices within ``(0, tau_max)``.
    """
    device = next(net.parameters()).device
    if t_fixed_years is None:
        fracs = (0.04, 0.10, 0.18, 0.28, 0.42)
        t_fixed_years = [float(f) * tau_max for f in fracs]
    t_years = [float(t) for t in t_fixed_years if 0 < t < tau_max * 0.98]
    if not t_years:
        t_years = [float(0.05 * tau_max)]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8))
    cmap = plt.cm.viridis(np.linspace(0.12, 0.92, len(t_years)))
    y_pos: list[np.ndarray] = []

    for j, t_y in enumerate(t_years):
        t_norm = float(t_y / tau_max)
        u_max = (tau_max - t_y) * 0.999
        u_min = max(1e-9 * tau_max, 1e-12)
        if u_max <= u_min:
            continue
        u_phys = np.logspace(np.log10(u_min), np.log10(u_max), n_u)
        u_p, h_p = _h_fixed_s_vs_u(net, device, t_norm, u_phys, tau_max)
        ha = np.abs(h_p)
        m = np.isfinite(ha) & np.isfinite(u_p) & (u_p > 0)
        y_plot = np.clip(ha[m], 1e-30, None)
        if h_abs_plot_min > 0:
            y_plot = np.where(y_plot > float(h_abs_plot_min), y_plot, np.nan)
        ax.plot(u_p[m], y_plot, color=cmap[j], lw=1.65, alpha=0.9, label=f"$t$ = {t_y:.4g} y")
        y_pos.append(y_plot[np.isfinite(y_plot) & (y_plot > 0)])

    ax.set_xlabel(r"lag $u = \tau - t$ (years), with $\tau = t+u$")
    ax.set_ylabel(r"$|h(t+u,\, t)|$")
    sub = rf", $|h| > {h_abs_plot_min:g}$" if h_abs_plot_min > 0 else ""
    ax.set_title(
        r"$|h(t+u,\, t)|$ vs $u$ at fixed $t$ "
        r"(Volterra: $\mathrm{net}(s{=}t,\,\tau{=}t{+}u)$)"
        + sub,
    )
    ax.set_xscale("log")
    if y_pos:
        y_cat = np.concatenate(y_pos)
        y_cat = y_cat[np.isfinite(y_cat) & (y_cat > 0)]
        if y_cat.size and float(np.nanmax(y_cat) / max(float(np.nanmin(y_cat)), 1e-30)) > 80.0:
            ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kernel_h_div_u_pow_dminus1(
    net: torch.nn.Module,
    tau_max: float,
    out_path: Path,
    d: float,
    t_fixed_years: list[float] | None = None,
    n_u: int = 512,
    h_abs_plot_min: float = 0.0,
) -> None:
    r"""
    Line plot of ``h(t+u,\, t) \,/\, u^{d-1}`` vs lag ``u`` in **years** (same slices as ``|h|`` plot).

    Uses the same Volterra evaluation as ``plot_kernel_abs_h_t_plus_u``; ``d`` should match
    paper7 / ``--type1-d`` (Wang–Xia factor ``u^{d-1} e^{-\kappa u}``).
    """
    device = next(net.parameters()).device
    if t_fixed_years is None:
        fracs = (0.04, 0.10, 0.18, 0.28, 0.42)
        t_fixed_years = [float(f) * tau_max for f in fracs]
    t_years = [float(t) for t in t_fixed_years if 0 < t < tau_max * 0.98]
    if not t_years:
        t_years = [float(0.05 * tau_max)]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8))
    cmap = plt.cm.viridis(np.linspace(0.12, 0.92, len(t_years)))
    y_pos: list[np.ndarray] = []

    for j, t_y in enumerate(t_years):
        t_norm = float(t_y / tau_max)
        u_max = (tau_max - t_y) * 0.999
        u_min = max(1e-9 * tau_max, 1e-12)
        if u_max <= u_min:
            continue
        u_phys = np.logspace(np.log10(u_min), np.log10(u_max), n_u)
        u_p, h_p = _h_fixed_s_vs_u(net, device, t_norm, u_phys, tau_max)
        ha = np.abs(h_p)
        if h_abs_plot_min > 0:
            h_p = np.where(ha > float(h_abs_plot_min), h_p, np.nan)
        u_safe = np.maximum(u_p, 1e-20)
        denom = np.power(u_safe, float(d) - 1.0)
        ratio = np.divide(
            h_p,
            denom,
            out=np.full_like(h_p, np.nan, dtype=np.float64),
            where=np.isfinite(h_p) & np.isfinite(denom) & (denom > 0) & (u_p > 0),
        )
        m = np.isfinite(ratio) & np.isfinite(u_p) & (u_p > 0)
        y_plot = ratio[m]
        ax.plot(u_p[m], y_plot, color=cmap[j], lw=1.65, alpha=0.9, label=f"$t$ = {t_y:.4g} y")
        y_pos.append(y_plot[np.isfinite(y_plot) & (np.abs(y_plot) > 0)])

    d_m1 = float(d) - 1.0
    ax.set_xlabel(r"lag $u = \tau - t$ (years), with $\tau = t+u$")
    ax.set_ylabel(rf"$h(t+u,\, t)\,/\,u^{{{d_m1}}}$")
    sub = rf", $|h| > {h_abs_plot_min:g}$" if h_abs_plot_min > 0 else ""
    ax.set_title(
        rf"$h(t+u,\, t)\,/\,u^{{{d_m1}}}$ vs $u$ at fixed $t$ ($d={d:g}$; same $t$-slices as $|h|$ plot)"
        + sub,
    )
    ax.set_xscale("log")
    if y_pos:
        y_cat = np.concatenate(y_pos)
        y_cat = y_cat[np.isfinite(y_cat) & (np.abs(y_cat) > 0)]
        if y_cat.size:
            rspan = float(np.nanmax(np.abs(y_cat)) / max(float(np.nanmin(np.abs(y_cat))), 1e-30))
            if rspan > 80.0 and np.all(y_cat > 0):
                ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kernel_h_div_u_pow_d(
    net: torch.nn.Module,
    tau_max: float,
    out_path: Path,
    d: float,
    t_fixed_years: list[float] | None = None,
    n_u: int = 512,
    h_abs_plot_min: float = 0.0,
) -> None:
    r"""
    Line plot of ``h(t+u,\, t) \,/\, u^{d}`` vs lag ``u`` in **years** (same slices as ``|h|`` plot).

    ``d`` defaults from ``--type1-d`` (same as other kernel diagnostic plots).
    """
    device = next(net.parameters()).device
    if t_fixed_years is None:
        fracs = (0.04, 0.10, 0.18, 0.28, 0.42)
        t_fixed_years = [float(f) * tau_max for f in fracs]
    t_years = [float(t) for t in t_fixed_years if 0 < t < tau_max * 0.98]
    if not t_years:
        t_years = [float(0.05 * tau_max)]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8))
    cmap = plt.cm.viridis(np.linspace(0.12, 0.92, len(t_years)))
    y_pos: list[np.ndarray] = []

    d_ex = float(d)
    for j, t_y in enumerate(t_years):
        t_norm = float(t_y / tau_max)
        u_max = (tau_max - t_y) * 0.999
        u_min = max(1e-9 * tau_max, 1e-12)
        if u_max <= u_min:
            continue
        u_phys = np.logspace(np.log10(u_min), np.log10(u_max), n_u)
        u_p, h_p = _h_fixed_s_vs_u(net, device, t_norm, u_phys, tau_max)
        ha = np.abs(h_p)
        if h_abs_plot_min > 0:
            h_p = np.where(ha > float(h_abs_plot_min), h_p, np.nan)
        u_safe = np.maximum(u_p, 1e-20)
        denom = np.power(u_safe, d_ex)
        ratio = np.divide(
            h_p,
            denom,
            out=np.full_like(h_p, np.nan, dtype=np.float64),
            where=np.isfinite(h_p) & np.isfinite(denom) & (denom > 0) & (u_p > 0),
        )
        m = np.isfinite(ratio) & np.isfinite(u_p) & (u_p > 0)
        y_plot = ratio[m]
        ax.plot(u_p[m], y_plot, color=cmap[j], lw=1.65, alpha=0.9, label=f"$t$ = {t_y:.4g} y")
        y_pos.append(y_plot[np.isfinite(y_plot) & (np.abs(y_plot) > 0)])

    ax.set_xlabel(r"lag $u = \tau - t$ (years), with $\tau = t+u$")
    ax.set_ylabel(rf"$h(t+u,\, t)\,/\,u^{{{d_ex}}}$")
    sub = rf", $|h| > {h_abs_plot_min:g}$" if h_abs_plot_min > 0 else ""
    ax.set_title(
        rf"$h(t+u,\, t)\,/\,u^{{{d_ex}}}$ vs $u$ at fixed $t$ ($d={d:g}$; same $t$-slices as $|h|$ plot)"
        + sub,
    )
    ax.set_xscale("log")
    if y_pos:
        y_cat = np.concatenate(y_pos)
        y_cat = y_cat[np.isfinite(y_cat) & (np.abs(y_cat) > 0)]
        if y_cat.size:
            rspan = float(np.nanmax(np.abs(y_cat)) / max(float(np.nanmin(np.abs(y_cat))), 1e-30))
            if rspan > 80.0 and np.all(y_cat > 0):
                ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kernel_h_div_u_pow_compare(
    specs: list[tuple[torch.nn.Module, float, str]],
    out_path: Path,
    u_pow: float = 0.5,
    t_fixed_years: list[float] | None = None,
    n_u: int = 512,
    h_abs_plot_min: float = 0.0,
) -> None:
    """
    Overlay ``h(t+u, t) / u^{u_pow}`` vs lag ``u`` for several kernels (e.g. paper7 trained with
    different ``d`` ranges). Each spec is ``(net, tau_max, legend_label)`` in **physical years**.
    Uses the same Volterra convention as ``plot_kernel_h_div_u_pow_dminus1``.
    """
    if len(specs) < 2:
        raise ValueError("plot_kernel_h_div_u_pow_compare needs at least two (net, tau_max, label) specs")
    tau_mins = [float(tm) for _, tm, _ in specs]
    tau_cap = min(tau_mins)
    if t_fixed_years is None:
        fracs = (0.04, 0.10, 0.18, 0.28, 0.42)
        t_fixed_years = [float(f) * tau_cap for f in fracs]
    t_years = [float(t) for t in t_fixed_years if 0 < t < tau_cap * 0.98]
    if not t_years:
        t_years = [float(0.05 * tau_cap)]

    u_p_ex = float(u_pow)
    lss = ("-", "--", "-.", ":")
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.0))
    cmap = plt.cm.viridis(np.linspace(0.12, 0.92, len(t_years)))
    y_pos: list[np.ndarray] = []

    for j, t_y in enumerate(t_years):
        u_mins = [max(1e-9 * float(tm), 1e-12) for _, tm, _ in specs]
        u_maxs = [(float(tm) - t_y) * 0.999 for _, tm, _ in specs]
        u_min = max(u_mins)
        u_max = min(u_maxs)
        if u_max <= u_min:
            continue
        u_phys = np.logspace(np.log10(u_min), np.log10(u_max), n_u)

        for mi, (net, tau_max_i, label) in enumerate(specs):
            device = next(net.parameters()).device
            t_norm = float(t_y / tau_max_i)
            u_p, h_p = _h_fixed_s_vs_u(net, device, t_norm, u_phys, tau_max_i)
            ha = np.abs(h_p)
            if h_abs_plot_min > 0:
                h_p = np.where(ha > float(h_abs_plot_min), h_p, np.nan)
            u_safe = np.maximum(u_p, 1e-20)
            denom = np.power(u_safe, u_p_ex)
            ratio = np.divide(
                h_p,
                denom,
                out=np.full_like(h_p, np.nan, dtype=np.float64),
                where=np.isfinite(h_p) & np.isfinite(denom) & (denom > 0) & (u_p > 0),
            )
            m = np.isfinite(ratio) & np.isfinite(u_p) & (u_p > 0)
            y_plot = ratio[m]
            ls = lss[mi % len(lss)]
            leg = f"{label}, $t$ = {t_y:.4g} y"
            ax.plot(
                u_p[m],
                y_plot,
                color=cmap[j],
                linestyle=ls,
                lw=1.65,
                alpha=0.92,
                label=leg,
            )
            y_pos.append(y_plot[np.isfinite(y_plot) & (np.abs(y_plot) > 0)])

    pow_tex = "1/2" if abs(u_p_ex - 0.5) < 1e-9 else f"{u_p_ex:g}"
    ax.set_xlabel(r"lag $u = \tau - t$ (years), with $\tau = t+u$")
    ax.set_ylabel(rf"$h(t+u,\, t)\,/\,u^{{{pow_tex}}}$")
    sub = rf", $|h| > {h_abs_plot_min:g}$" if h_abs_plot_min > 0 else ""
    ax.set_title(
        rf"Compare $h/u^{{{pow_tex}}}$ (fixed power; dashed vs solid = different training)" + sub,
    )
    ax.set_xscale("log")
    if y_pos:
        y_cat = np.concatenate(y_pos)
        y_cat = y_cat[np.isfinite(y_cat) & (np.abs(y_cat) > 0)]
        if y_cat.size:
            rspan = float(np.nanmax(np.abs(y_cat)) / max(float(np.nanmin(np.abs(y_cat))), 1e-30))
            if rspan > 80.0 and np.all(y_cat > 0):
                ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=7.5, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _tex_u_exponent_for_title(x: float) -> str:
    """Mathtext fragment for ``u`` raised to ``x`` (title only)."""
    mapping = {
        -0.5: r"-1/2",
        -0.25: r"-1/4",
        0.25: r"1/4",
        0.5: r"1/2",
        1.0: r"1",
    }
    fx = float(x)
    for k, v in mapping.items():
        if abs(fx - k) < 1e-12:
            return v
    return f"{fx:g}"


def plot_kernel_h_t_plus_u(
    net: torch.nn.Module,
    tau_max: float,
    out_path: Path,
    t_fixed_years: list[float] | None = None,
    n_u: int = 512,
) -> None:
    """
    Line plot of **signed** ``h(t+u, t)`` vs lag ``u`` (same convention as ``plot_kernel_abs_h_t_plus_u``).
    For nonnegative kernels this coincides with ``|h|``; title uses ``h`` for clarity.
    """
    device = next(net.parameters()).device
    if t_fixed_years is None:
        fracs = (0.04, 0.10, 0.18, 0.28, 0.42)
        t_fixed_years = [float(f) * tau_max for f in fracs]
    t_years = [float(t) for t in t_fixed_years if 0 < t < tau_max * 0.98]
    if not t_years:
        t_years = [float(0.05 * tau_max)]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8))
    cmap = plt.cm.viridis(np.linspace(0.12, 0.92, len(t_years)))
    y_pos: list[np.ndarray] = []

    for j, t_y in enumerate(t_years):
        t_norm = float(t_y / tau_max)
        u_max = (tau_max - t_y) * 0.999
        u_min = max(1e-9 * tau_max, 1e-12)
        if u_max <= u_min:
            continue
        u_phys = np.logspace(np.log10(u_min), np.log10(u_max), n_u)
        u_p, h_p = _h_fixed_s_vs_u(net, device, t_norm, u_phys, tau_max)
        m = np.isfinite(h_p) & np.isfinite(u_p) & (u_p > 0)
        y_plot = h_p[m]
        ax.plot(u_p[m], y_plot, color=cmap[j], lw=1.65, alpha=0.9, label=f"$t$ = {t_y:.4g} y")
        y_pos.append(np.abs(y_plot[np.isfinite(y_plot)]))

    ax.set_xlabel(r"lag $u = \tau - t$ (years), with $\tau = t+u$")
    ax.set_ylabel(r"$h(t+u,\, t)$")
    ax.set_title(
        r"$h(t+u,\, t)$ vs $u$ at fixed $t$ "
        r"(Volterra: $\mathrm{net}(s{=}t,\,\tau{=}t{+}u)$)",
    )
    ax.set_xscale("log")
    if y_pos:
        y_cat = np.concatenate(y_pos)
        y_cat = y_cat[np.isfinite(y_cat) & (y_cat > 0)]
        if y_cat.size and float(np.nanmax(y_cat) / max(float(np.nanmin(y_cat)), 1e-30)) > 80.0:
            ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kernel_h_div_u_pow_subplots(
    net: torch.nn.Module,
    tau_max: float,
    out_path: Path,
    exponents: list[float],
    t_fixed_years: list[float] | None = None,
    n_u: int = 512,
    ncols: int = 3,
    h_abs_plot_min: float = 0.0,
) -> None:
    """
    Grid of ``h(t+u,t) / u^x`` vs ``u`` for several exponents ``x`` (same ``t``-slices as other line plots).
    Use negative ``x`` to emphasize behaviour vs ``u^{-|x|}`` (i.e. multiply by ``u^{|x|}``).
    """
    device = next(net.parameters()).device
    if t_fixed_years is None:
        fracs = (0.04, 0.10, 0.18, 0.28, 0.42)
        t_fixed_years = [float(f) * tau_max for f in fracs]
    t_years = [float(t) for t in t_fixed_years if 0 < t < tau_max * 0.98]
    if not t_years:
        t_years = [float(0.05 * tau_max)]

    n_exp = len(exponents)
    nrows = int(np.ceil(n_exp / max(1, ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.9 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ei, x_exp in enumerate(exponents):
        ax = axes_flat[ei]
        cmap = plt.cm.viridis(np.linspace(0.12, 0.92, len(t_years)))
        x_ex = float(x_exp)
        tex = _tex_u_exponent_for_title(x_ex)
        y_pos: list[np.ndarray] = []

        for j, t_y in enumerate(t_years):
            t_norm = float(t_y / tau_max)
            u_max = (tau_max - t_y) * 0.999
            u_min = max(1e-9 * tau_max, 1e-12)
            if u_max <= u_min:
                continue
            u_phys = np.logspace(np.log10(u_min), np.log10(u_max), n_u)
            u_p, h_p = _h_fixed_s_vs_u(net, device, t_norm, u_phys, tau_max)
            ha = np.abs(h_p)
            if h_abs_plot_min > 0:
                h_p = np.where(ha > float(h_abs_plot_min), h_p, np.nan)
            u_safe = np.maximum(u_p, 1e-24)
            denom = np.power(u_safe, x_ex)
            ratio = np.divide(
                h_p,
                denom,
                out=np.full_like(h_p, np.nan, dtype=np.float64),
                where=np.isfinite(h_p) & np.isfinite(denom) & (u_p > 0),
            )
            m = np.isfinite(ratio) & np.isfinite(u_p) & (u_p > 0)
            y_plot = ratio[m]
            ax.plot(u_p[m], y_plot, color=cmap[j], lw=1.45, alpha=0.9, label=f"$t$ = {t_y:.4g} y")
            y_pos.append(y_plot[np.isfinite(y_plot) & (np.abs(y_plot) > 0)])

        ax.set_xlabel(r"$u$ (years)")
        ax.set_ylabel(rf"$h\,/\,u^{{{tex}}}$")
        sub = rf", $|h|>{h_abs_plot_min:g}$" if h_abs_plot_min > 0 else ""
        ax.set_title(rf"$h\,/\,u^{{{tex}}}$ vs $u${sub}", fontsize=10)
        ax.set_xscale("log")
        if y_pos:
            y_cat = np.concatenate(y_pos)
            y_cat = y_cat[np.isfinite(y_cat) & (np.abs(y_cat) > 0)]
            if y_cat.size:
                rspan = float(np.nanmax(np.abs(y_cat)) / max(float(np.nanmin(np.abs(y_cat))), 1e-30))
                if rspan > 80.0 and np.all(y_cat > 0):
                    ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=6.5, framealpha=0.88)

    for k in range(n_exp, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.suptitle(
        r"Scaled slices $h(t{+}u,t)/u^x$ for several $x$ (same $t$-lines as $|h|$ plots)",
        y=1.02,
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kernel_heatmap_s_minus_tau(
    net: KernelNet,
    tau_max: float,
    out_path: Path,
    n_tau: int = 50,
    n_s: int = 50,
    grid: KernelGrid | None = None,
    h_abs_plot_min: float = 0.0,
) -> None:
    """Same kernel as h(τ, s), but axes τ and s − τ (years). On the Gaussian path τ > s, s − τ < 0."""
    if grid is None:
        tt, ss, h = _kernel_grid_h(net, n_tau=n_tau, n_s=n_s)
    else:
        tt, ss, h = grid
    lag = ss - tt
    h_plot = _mask_h_abs_leq(h, h_abs_plot_min)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.pcolormesh(
        tt * tau_max, lag * tau_max, h_plot,
        shading="auto", cmap="viridis",
    )
    ax.set_xlabel("τ (time to maturity, years)")
    ax.set_ylabel("s − τ (years)")
    title = "Kernel h(τ, s) vs (τ, s − τ)"
    if h_abs_plot_min > 0:
        title += rf" ($|h| > {h_abs_plot_min:g}$ only)"
    ax.set_title(title)
    ax.axhline(0.0, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.colorbar(im, ax=ax, label="h")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def type1_reference_h(
    tt: np.ndarray,
    ss: np.ndarray,
    tau_max: float,
    d: float,
    kappa: float,
) -> np.ndarray:
    """
    Wang–Xia / ``paper7`` admissible envelope on lag u = τ − s (years), masked to τ > s.
    Same structural factor as ``StructuredKernelNetPaper7`` (without learned g).
    """
    u = (tt - ss) * float(tau_max)
    u = np.maximum(u, 1e-8)
    h = (u ** (d - 1.0)) * np.exp(-kappa * u)
    mask = tt > ss
    return np.where(mask, h, 0.0)


def plot_kernel_learned_vs_type1(
    net: KernelNet,
    tau_max: float,
    out_path: Path,
    n_tau: int = 50,
    n_s: int = 50,
    type1_d: float = 0.75,
    type1_kappa: float = 1.0,
    grid: KernelGrid | None = None,
    h_abs_plot_min: float = 0.0,
) -> None:
    """Side-by-side heatmaps: learned h vs Type I reference u^{d-1} e^{-κ u} (u = τ − s)."""
    if grid is None:
        tt, ss, h_learned = _kernel_grid_h(net, n_tau=n_tau, n_s=n_s)
    else:
        tt, ss, h_learned = grid
    h_type1 = type1_reference_h(tt, ss, tau_max, d=type1_d, kappa=type1_kappa)
    h_learned_plot = _mask_h_abs_leq(h_learned, h_abs_plot_min)
    tau_phys = tt * tau_max
    s_phys = ss * tau_max
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    t0 = "Learned kernel h(τ, s)"
    if h_abs_plot_min > 0:
        t0 += rf" ($|h| > {h_abs_plot_min:g}$ only)"
    titles = (
        t0,
        f"Type I reference: u^(d−1) e^(−κu), u = τ − s\n(d={type1_d}, κ={type1_kappa})",
    )
    for ax, h, title in zip(axes, (h_learned_plot, h_type1), titles):
        im = ax.pcolormesh(tau_phys, s_phys, h, shading="auto", cmap="viridis")
        plt.colorbar(im, ax=ax, label="h")
        ax.set_xlabel("τ (years)")
        ax.set_ylabel("s (years)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Kernel: learned vs Type I (Wang–Xia admissible envelope)", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kernel_constraint_diagnostics(
    tt: np.ndarray,
    ss: np.ndarray,
    h: np.ndarray,
    out_path: Path,
    expect_nonnegative: bool,
) -> None:
    """
    Visual checks for Volterra support (h = 0 if τ ≤ s) and optional nonnegativity (h ≥ 0 on τ > s),
    as in PROJECT_PSEUDOCODE section 2.1.
    """
    invalid = tt <= ss
    valid = ~invalid
    h_inv = h[invalid]
    h_val = h[valid]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].hist(h_val.ravel(), bins=40, color="steelblue", edgecolor="white", alpha=0.9)
    axes[0].axvline(0.0, color="crimson", linestyle="--", linewidth=1.0, label="h = 0")
    axes[0].set_xlabel("h")
    axes[0].set_ylabel("count (valid: τ > s)")
    axes[0].set_title("Learned h on valid region")
    axes[0].legend(loc="upper right", fontsize=8)

    max_abs_inv = float(np.max(np.abs(h_inv))) if h_inv.size else 0.0
    frac_inv = float(np.mean(np.abs(h_inv) > 1e-6)) if h_inv.size else 0.0
    min_valid = float(np.min(h_val)) if h_val.size else float("nan")
    n_neg = int(np.sum(h_val < 0)) if h_val.size else 0
    n_val = int(h_val.size)

    lines = [
        "Kernel constraints (code / notes)",
        "",
        "Causal Volterra support: h = 0 when τ ≤ s",
        f"  max |h| on τ ≤ s: {max_abs_inv:.6e}",
        f"  P(|h| > 1e-6 on τ ≤ s): {frac_inv:.6f}",
        "",
        "Nonnegativity" + (" (expected: softplus / paper7)" if expect_nonnegative else " (signed kernel: negatives allowed)"),
        f"  min(h) on τ > s: {min_valid:.6e}",
        f"  h < 0 on valid: {n_neg} / {n_val}",
    ]
    axes[1].axis("off")
    axes[1].text(
        0.04, 0.97, "\n".join(lines),
        transform=axes[1].transAxes, va="top", ha="left",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )
    plt.suptitle("Constraint check: support + nonnegativity", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kernel_eq7_constraint(
    tt: np.ndarray,
    ss: np.ndarray,
    h: np.ndarray,
    tau_max: float,
    out_path: Path,
    d: float,
    kappa: float,
) -> dict[str, float]:
    """
    Wang–Xia Eq. (7)-style structure: h(τ,s) = u^{d-1} e^{-κ u} g(τ,s) with u = τ − s (years).

    Reports implied g = h / (u^{d-1} e^{-κ u}) and a small-u log–log slope vs target (d−1).
    """
    u = (tt - ss) * float(tau_max)
    base = type1_reference_h(tt, ss, tau_max, d=d, kappa=kappa)
    valid = tt > ss
    eps = 1e-18
    g_hat = np.where(valid & (base > eps), h / (base + eps), np.nan)

    u_v = u[valid].ravel()
    h_v = h[valid].ravel()
    g_v = g_hat[valid].ravel()
    g_finite = g_v[np.isfinite(g_v)]

    # Small-u log-log slope (lower quartile of u on valid set)
    slope_est = float("nan")
    if u_v.size > 15:
        u_pos = u_v[u_v > 1e-12]
        h_pos = h_v[u_v > 1e-12]
        if u_pos.size > 10:
            thresh = float(np.percentile(u_pos, 25))
            sub = u_pos <= max(thresh, 1e-10)
            if np.sum(sub) > 5:
                lu = np.log(u_pos[sub])
                lh = np.log(np.maximum(h_pos[sub], 1e-30))
                slope_est, _intercept = np.polyfit(lu, lh, 1)

    target_slope = d - 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    tau_p = tt * tau_max
    u_p = u

    g_plot = np.clip(g_hat, np.nanpercentile(g_finite, 2), np.nanpercentile(g_finite, 98)) if g_finite.size else g_hat
    ax = axes[0, 0]
    im = ax.pcolormesh(tau_p, u_p, g_plot, shading="auto", cmap="cividis")
    ax.set_xlabel("τ (years)")
    ax.set_ylabel("u = τ − s (years)")
    ax.set_title(r"Implied $g$ = $h\,/\,(u^{d-1} e^{-\kappa u})$")
    plt.colorbar(im, ax=ax, label="g (clipped 2–98%)")
    ax.set_ylim(bottom=0)

    ax = axes[0, 1]
    n_max = 4000
    idx = np.arange(u_v.size)
    if idx.size > n_max:
        idx = np.random.choice(idx.size, n_max, replace=False)
    else:
        idx = np.arange(idx.size)
    u_s = u_v[idx]
    h_s = h_v[idx]
    m = (u_s > 1e-12) & (h_s > 1e-30)
    if np.any(m):
        ax.scatter(np.log10(u_s[m]), np.log10(h_s[m]), s=4, alpha=0.25, c="tab:blue", edgecolors="none")
        u_line = np.logspace(np.log10(max(float(u_s[m].min()), 1e-8)), np.log10(float(u_s[m].max())), 50)
        tmpl = (u_s[m] ** (d - 1.0)) * np.exp(-kappa * u_s[m])
        scale = np.nanmedian(h_s[m] / (tmpl + 1e-30))
        h_line = (u_line ** (d - 1.0)) * np.exp(-kappa * u_line) * scale
        ax.plot(np.log10(u_line), np.log10(np.maximum(h_line, 1e-30)), "r-", lw=1.5, label="Eq.(7) factor × med(g)")
    ax.set_xlabel("log10(u), u = τ − s")
    ax.set_ylabel("log10(h)")
    ax.set_title("log–log: learned h vs u (red = template × scale)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    if u_v.size > 0 and np.any(u_v > 0):
        u_min = float(np.percentile(u_v[u_v > 0], 5))
        u_max = float(np.percentile(u_v[u_v > 0], 95))
        bins = np.linspace(u_min, u_max, 22)
        med_g, xb = [], []
        for i in range(len(bins) - 1):
            sel = (u_v >= bins[i]) & (u_v < bins[i + 1])
            if np.any(sel) and np.any(np.isfinite(g_v[sel])):
                xb.append(0.5 * (bins[i] + bins[i + 1]))
                med_g.append(float(np.nanmedian(g_v[sel])))
        if xb:
            ax.plot(xb, med_g, "o-", color="darkgreen", ms=4)
    ax.set_xlabel("u = τ − s (years)")
    ax.set_ylabel("median g in bin")
    ax.set_title("g vs u (binned; flat g ≈ Eq.(7) holds up to envelope)")
    ax.grid(True, alpha=0.3)

    median_rel_err = float("nan")
    base_v = base[valid].ravel()
    if g_finite.size and base_v.size == h_v.size:
        med_g_all = float(np.nanmedian(g_finite))
        rel = np.abs(h_v - base_v * med_g_all) / (np.abs(h_v) + 1e-12)
        median_rel_err = float(np.nanmedian(rel))

    lines = [
        "Eq. (7) style: h = u^{d-1} e^{-κ u} · g(τ,s),  u = τ − s",
        f"  reference d = {d}, κ = {kappa}  (match to paper7 / notes)",
        "",
        f"  g = h / base:  median={np.nanmedian(g_finite) if g_finite.size else float('nan'):.6e}",
        f"                   p05–p95=({np.nanpercentile(g_finite,5) if g_finite.size else float('nan'):.4e}, "
        f"{np.nanpercentile(g_finite,95) if g_finite.size else float('nan'):.4e})",
        "",
        "Small-u log–log slope (lower quartile of u):",
        f"  estimated = {slope_est:.4f}   target ≈ d−1 = {target_slope:.4f}",
        "",
        "(If h follows Eq.(7) with slowly varying g, slope ≈ d−1 for tiny u; curvature from e^{-κu}.)",
    ]
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.03, 0.97, "\n".join(lines),
        transform=axes[1, 1].transAxes, va="top", ha="left",
        family="monospace", fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.45),
    )

    plt.suptitle("Eq. (7) kernel structure check (Wang–Xia / paper7 factor)", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "g_median": float(np.nanmedian(g_finite)) if g_finite.size else float("nan"),
        "g_p05": float(np.nanpercentile(g_finite, 5)) if g_finite.size else float("nan"),
        "g_p95": float(np.nanpercentile(g_finite, 95)) if g_finite.size else float("nan"),
        "loglog_slope_small_u": slope_est,
        "target_slope_d_minus_1": target_slope,
        "median_rel_err_vs_scaled_base": median_rel_err,
    }


def _h_fixed_s_vs_u(
    net: torch.nn.Module,
    device: torch.device,
    s_norm: float,
    u_phys: np.ndarray,
    tau_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate h(s, τ) with τ = s + u (physical u in years). Returns u_phys, h (invalid nan)."""
    u_phys = np.asarray(u_phys, dtype=np.float64)
    tau_norm = s_norm + u_phys / tau_max
    s_t = torch.full((len(u_phys),), float(s_norm), device=device, dtype=torch.float32)
    tau_t = torch.tensor(tau_norm, device=device, dtype=torch.float32)
    with torch.no_grad():
        h = net(s_t, tau_t).cpu().numpy().astype(np.float64)
    bad = (tau_norm <= s_norm + 1e-10) | (tau_norm > 1.0 + 1e-6) | (u_phys <= 0)
    h = np.where(bad, np.nan, h)
    return u_phys, h


def _h_fixed_tau_vs_u(
    net: torch.nn.Module,
    device: torch.device,
    tau_norm: float,
    u_phys: np.ndarray,
    tau_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate h(s, τ) with fixed τ, u = τ - s in years (s = τ - u)."""
    u_phys = np.asarray(u_phys, dtype=np.float64)
    s_norm = tau_norm - u_phys / tau_max
    s_t = torch.tensor(s_norm, device=device, dtype=torch.float32)
    tau_t = torch.full((len(u_phys),), float(tau_norm), device=device, dtype=torch.float32)
    with torch.no_grad():
        h = net(s_t, tau_t).cpu().numpy().astype(np.float64)
    bad = (s_norm < -1e-10) | (tau_norm <= s_norm + 1e-10) | (u_phys <= 0)
    h = np.where(bad, np.nan, h)
    return u_phys, h


def plot_kernel_eq7_asymptotic_tails(
    net: torch.nn.Module,
    tau_max: float,
    out_path: Path,
    d: float,
    kappa: float,
    n_s_anchors: int = 10,
    n_tau_anchors: int = 8,
    n_u: int = 400,
    u_small_frac: float = 0.06,
    u_large_frac: float = 0.12,
    h_abs_plot_min: float = 0.0,
) -> dict[str, float]:
    """
    Eq. (7) tails: h vs lag u = τ − s for fixed s and fixed τ; small-u log-log; large-u Option A/B;
    g(s,u) = h / (u^{d-1} e^{-κ u}) collapse over s.
    """
    device = next(net.parameters()).device
    rng = np.random.default_rng(0)

    s_list = np.linspace(0.04, 0.82, n_s_anchors)
    tau_list = np.linspace(0.18, 0.98, n_tau_anchors)

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.0))

    # --- (0,0) h vs u for fixed s (anchor s, varying u)
    ax = axes[0, 0]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(s_list)))
    for j, s_n in enumerate(s_list):
        u_max = (1.0 - s_n) * tau_max * 0.999
        u_min = max(1e-7 * tau_max, 1e-12)
        u_phys = np.logspace(np.log10(u_min), np.log10(u_max), n_u)
        u_p, h_p = _h_fixed_s_vs_u(net, device, float(s_n), u_phys, tau_max)
        h_p = _mask_h_abs_leq(h_p, h_abs_plot_min)
        ax.plot(u_p, h_p, color=cmap[j], lw=1.2, alpha=0.85, label=f"s={s_n:.2f}")
    ax.set_xlabel(r"lag $u = \tau - s$ (years)")
    ax.set_ylabel(r"$h(s,\tau)$")
    title0 = "Fixed $s$: curves $u \\mapsto h(s, s+u)$"
    if h_abs_plot_min > 0:
        title0 += rf" ($|h| > {h_abs_plot_min:g}$)"
    ax.set_title(title0)
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=6, ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    # --- (0,1) h vs u for fixed τ
    ax = axes[0, 1]
    cmap2 = plt.cm.plasma(np.linspace(0.15, 0.95, len(tau_list)))
    for j, t_n in enumerate(tau_list):
        u_max = t_n * tau_max * 0.999
        u_min = max(1e-7 * tau_max, 1e-12)
        u_phys = np.logspace(np.log10(u_min), np.log10(u_max), n_u)
        u_p, h_p = _h_fixed_tau_vs_u(net, device, float(t_n), u_phys, tau_max)
        h_p = _mask_h_abs_leq(h_p, h_abs_plot_min)
        ax.plot(u_p, h_p, color=cmap2[j], lw=1.2, alpha=0.85, label=f"τ={t_n:.2f}")
    ax.set_xlabel(r"lag $u = \tau - s$ (years)")
    ax.set_ylabel(r"$h(s,\tau)$")
    title1 = "Fixed $\\tau$: curves $u \\mapsto h(\\tau-u,\\tau)$"
    if h_abs_plot_min > 0:
        title1 += rf" ($|h| > {h_abs_plot_min:g}$)"
    ax.set_title(title1)
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=6, ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    # --- (1,0) small-u log-log, pooled over s
    ax = axes[1, 0]
    u_cut = u_small_frac * tau_max
    pu, ph = [], []
    for s_n in s_list:
        u_phys = np.logspace(np.log10(max(1e-9 * tau_max, 1e-15)), np.log10(u_cut * 0.999), 120)
        u_p, h_p = _h_fixed_s_vs_u(net, device, float(s_n), u_phys, tau_max)
        h_p = _mask_h_abs_leq(h_p, h_abs_plot_min)
        m = np.isfinite(h_p) & (h_p > 1e-30) & (u_p > 0)
        pu.extend(u_p[m].tolist())
        ph.extend(h_p[m].tolist())
    pu = np.array(pu)
    ph = np.array(ph)
    slope_small_val = float("nan")
    d_hat_small = float("nan")
    if pu.size > 30:
        sub = rng.choice(pu.size, min(2500, pu.size), replace=False)
        pu_s, ph_s = pu[sub], ph[sub]
        ax.scatter(np.log10(pu_s), np.log10(ph_s), s=3, alpha=0.2, c="tab:blue", edgecolors="none")
        lo, hi = np.percentile(np.log10(pu_s), [5, 95])
        m_fit = (np.log10(pu_s) >= lo) & (np.log10(pu_s) <= hi) & np.isfinite(np.log10(ph_s))
        if m_fit.sum() > 15:
            lx = np.log10(pu_s[m_fit])
            ly = np.log10(ph_s[m_fit])
            slope_small, icept = np.polyfit(lx, ly, 1)
            slope_small_val = float(slope_small)
            d_hat_small = slope_small_val + 1.0
            xx = np.linspace(lx.min(), lx.max(), 50)
            ax.plot(xx, slope_small * xx + icept, "r-", lw=2, label=f"fit slope={slope_small:.3f}")
            ref = (d - 1.0)
            ax.plot(xx, ref * xx + (ly.mean() - ref * lx.mean()), "k--", lw=1.2, label=f"target $d-1$={ref:.3f}")
    ax.set_xlabel(r"$\log_{10} u$")
    ax.set_ylabel(r"$\log_{10} h$")
    ax.set_title(rf"Small-$u$ log-log (pooled, $u < {u_small_frac:.2f}\,\tau_{{\max}}$)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,1) Option A: log(h u^{1-d}) vs u (large u)
    ax = axes[1, 1]
    u_lo = u_large_frac * tau_max
    kappa_hat = float("nan")
    cmap_a = plt.cm.tab10(np.linspace(0, 1, 10))
    for j, s_n in enumerate(s_list[:: max(1, len(s_list) // 5)]):
        u_max = (1.0 - s_n) * tau_max * 0.999
        u_phys = np.linspace(u_lo * 1.001, u_max, 200)
        u_p, h_p = _h_fixed_s_vs_u(net, device, float(s_n), u_phys, tau_max)
        h_p = _mask_h_abs_leq(h_p, h_abs_plot_min)
        m = np.isfinite(h_p) & (h_p > 0) & (u_p > u_lo)
        z = h_p[m] * np.power(np.maximum(u_p[m], 1e-20), 1.0 - d)
        if np.sum(m) > 5 and np.all(z > 0):
            ax.plot(u_p[m], np.log(z), color=cmap_a[j % 10], lw=1.0, alpha=0.75)
    ax.set_xlabel(r"$u$ (years)")
    ax.set_ylabel(r"$\log(h\, u^{1-d})$")
    ax.set_title("Option A (large $u$): slope $\\approx -\\kappa$ if Eq.(7) tail holds")
    ax.grid(True, alpha=0.3)
    # pooled κ fit from median s
    s_mid = float(np.median(s_list))
    u_max = (1.0 - s_mid) * tau_max * 0.999
    u_phys = np.linspace(max(u_lo, 1e-6), u_max, 300)
    u_p, h_p = _h_fixed_s_vs_u(net, device, s_mid, u_phys, tau_max)
    h_p = _mask_h_abs_leq(h_p, h_abs_plot_min)
    m = np.isfinite(h_p) & (h_p > 0) & (u_p > u_lo)
    z = h_p[m] * np.power(np.maximum(u_p[m], 1e-20), 1.0 - d)
    if m.sum() > 8 and np.all(z > 0):
        slope_k, icept_k = np.polyfit(u_p[m], np.log(z), 1)
        kappa_hat = float(-slope_k)
        ax.plot(
            u_p[m],
            slope_k * u_p[m] + icept_k,
            "k--",
            lw=1.5,
            label=f"linear fit, $\\hat{{\\kappa}}={kappa_hat:.3f}$ (target $\\kappa$={kappa})",
        )
        ax.legend(loc="best", fontsize=8)

    # --- (2,0) Option B / collapse: g = h / (u^{d-1} e^{-κ u})
    ax = axes[2, 0]
    base_ref = lambda uu: np.power(np.maximum(uu, 1e-20), d - 1.0) * np.exp(-kappa * uu)
    for j, s_n in enumerate(s_list):
        u_max = (1.0 - s_n) * tau_max * 0.999
        u_phys = np.logspace(np.log10(max(1e-8 * tau_max, 1e-15)), np.log10(u_max), n_u)
        u_p, h_p = _h_fixed_s_vs_u(net, device, float(s_n), u_phys, tau_max)
        h_p = _mask_h_abs_leq(h_p, h_abs_plot_min)
        denom = base_ref(u_p)
        g = np.divide(
            h_p,
            denom,
            out=np.full_like(h_p, np.nan),
            where=np.isfinite(h_p) & (denom > 1e-30),
        )
        ax.plot(u_p, g, color=cmap[j], lw=1.0, alpha=0.7)
    ax.set_xlabel(r"$u$ (years)")
    ax.set_ylabel(r"$g(s,u)=h\,/\,(u^{d-1} e^{-\kappa u})$")
    ax.set_title("Option B / collapse: $g(s,u)$ vs $u$ for many $s$")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # --- (2,1) summary text
    ax = axes[2, 1]
    ax.axis("off")
    in_half_one = (0.5 < d_hat_small < 1.0) if np.isfinite(d_hat_small) else False
    lines = [
        "Eq. (7) asymptotic checks (same d, κ as --type1-d, --type1-kappa)",
        "",
        f"Small-u: fitted log-log slope ≈ d−1  →  d̂ = slope + 1 = {d_hat_small:.4f}" if np.isfinite(d_hat_small) else "Small-u: (insufficient points)",
        f"  Target d−1 = {d - 1:.4f};  check d̂ ∈ (1/2, 1): {in_half_one} (paper7 uses d ∈ (1/2, 1))",
        "",
        f"Large-u Option A: κ̂ = −d/du log(h u^(1−d)) ≈ {kappa_hat:.4f}" if np.isfinite(kappa_hat) else "Large-u: (insufficient)",
        f"  Target κ = {kappa}",
        "",
        "Interpretation:",
        "  • Similar h-vs-u shapes across s → lag-dominated kernel.",
        "  • Small-u line ~ slope d−1; large-u Option A ~ slope −κ; Option B ~ flat tail.",
        "  • g(s,u) curves stack → Eq.(7) up to vertical scale; spread → extra s-structure.",
    ]
    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left", family="monospace", fontsize=8.2,
        bbox=dict(boxstyle="round", facecolor="honeydew", alpha=0.5),
    )

    plt.suptitle("Eq. (7): lag structure, small-$u$ power law, large-$u$ exponential tail", y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "eq7_small_u_loglog_slope": slope_small_val,
        "eq7_d_hat_from_slope": d_hat_small,
        "eq7_target_d_minus_1": d - 1.0,
        "eq7_kappa_hat_option_a": kappa_hat,
        "eq7_target_kappa": kappa,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Saved model .pt file")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Path to CSV data")
    parser.add_argument(
        "--strict-10days-data",
        action="store_true",
        help="Load only final_call_no_madan_strict_10days_*.csv (must match training data choice).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "output",
        help="Base directory for plots (see also --out-subdir / --timestamp-out-dir).",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help=(
            "If set, all artifacts go to ``out-dir / out-subdir /`` so existing files in "
            "``out-dir`` are left unchanged. Example: --out-subdir eval_paper7_v2"
        ),
    )
    parser.add_argument(
        "--timestamp-out-dir",
        action="store_true",
        help=(
            "Save under ``out-dir / eval_YYYYmmdd_HHMMSS/`` (new folder each run). "
            "Mutually exclusive with --out-subdir."
        ),
    )
    parser.add_argument("--n-grid", type=int, default=64)
    parser.add_argument("--n-u", type=int, default=256)
    parser.add_argument("--n-r", type=int, default=20)
    parser.add_argument("--n-s", type=int, default=20)
    parser.add_argument("--r", type=float, default=0.0)
    parser.add_argument("--max-eval", type=int, default=5000, help="Max samples for eval (for speed)")
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Use full_model_state (train_full.py checkpoint); else simplified net_state",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=None,
        help="Filter option mids below this (default: value saved in checkpoint, else 0).",
    )
    parser.add_argument(
        "--type1-d",
        type=float,
        default=0.75,
        help="Type I / Eq.(7) reference exponent d in u^(d-1) e^(-κ u); must lie in (1/2, 2). "
        "For paper7, h/u^{d-1} plots use the learned d from the checkpoint when available.",
    )
    parser.add_argument(
        "--type1-kappa",
        type=float,
        default=1.0,
        help="Type I reference kernel: decay κ in u^(d-1) e^(-κ u).",
    )
    parser.add_argument(
        "--option-type",
        type=str,
        choices=("C", "P"),
        default="C",
        help="Which options to load (aligned puts: P).",
    )
    parser.add_argument(
        "--synthetic-call-from-puts",
        action="store_true",
        help="Apply same put→synthetic-call mapping as training (uses --r).",
    )
    parser.add_argument(
        "--raw-put-targets",
        action="store_true",
        help="With --option-type P, skip auto parity (must match training intent).",
    )
    parser.add_argument(
        "--s0-fallback",
        type=str,
        choices=("mean_strike", "daily_median_strike", "daily_vwap_strike"),
        default="mean_strike",
        help="Must match training when no underlying bid/ask in CSV.",
    )
    parser.add_argument(
        "--tau-bucket-edges",
        type=str,
        default=None,
        help=(
            "Optional comma-separated τ edges in **years** for call_price_vs_strike.png, "
            "e.g. 0,0.05,0.25,1.0 → panels [0,0.05), [0.05,0.25), [0.25,1.0]. "
            "Up to 16 panels (17 edges). Omit for default **quantile** buckets (tertiles if n=3)."
        ),
    )
    parser.add_argument(
        "--call-vs-strike-max-points",
        type=int,
        default=3000,
        help=(
            "Max subsampled strike points **across all** τ panels in call_price_vs_strike.png "
            "(split evenly per active panel). Default 3000 shows full buckets for typical CSV sizes; "
            "lower (e.g. 48) thins the scatter for huge files."
        ),
    )
    parser.add_argument(
        "--call-vs-strike-min-per-panel",
        type=int,
        default=10,
        help=(
            "Minimum rows in a τ bucket to draw a panel in call_price_vs_strike. "
            "Lower values show more maturity slices when data are sparse (default was 30 in-code)."
        ),
    )
    parser.add_argument(
        "--call-vs-strike-y-max",
        type=float,
        default=None,
        help=(
            "Shared y-axis 'Call price' uses [0, this value] so extreme premiums do not compress "
            "the rest of the scatter. Omit: no cap for simplified eval; with --full-model, "
            "defaults to 8000 unless you set this. Use a very large value to disable the full-model default."
        ),
    )
    parser.add_argument(
        "--kernel-h-t-plus-u-t-years",
        type=str,
        default=None,
        help=(
            "Comma-separated times t in **years** for kernel_abs_h_t_plus_u*.png "
            "(line plot |h(t+u,t)| vs lag u, not a heatmap). Omit for default slices."
        ),
    )
    parser.add_argument(
        "--kernel-h-plot-min",
        type=float,
        default=0.0,
        help=(
            "If > 0, mask kernel values with |h| <= this threshold in h heatmaps, "
            "|h(t+u,t)| line plot, learned-vs-type1 (learned panel only), and eq.7 asymptotic h curves; "
            "those figure filenames get suffix '_habsgt<value>' (e.g. 0.0001 → …_habsgt0.0001.png). "
            "Default 0: no masking."
        ),
    )
    parser.add_argument(
        "--strike-band",
        type=float,
        nargs=2,
        metavar=("REL_LOW", "REL_HIGH"),
        default=None,
        help="Optional override: ATM band vs checkpoint S0 (otherwise replay ``strike_band`` from checkpoint).",
    )
    args = parser.parse_args()

    root_out = Path(args.out_dir).resolve()
    if args.timestamp_out_dir and args.out_subdir:
        parser.error("Use only one of --timestamp-out-dir and --out-subdir.")
    if args.timestamp_out_dir:
        args.out_dir = root_out / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    elif args.out_subdir is not None and str(args.out_subdir).strip() != "":
        sub = str(args.out_subdir).strip()
        if ".." in sub.replace("\\", "/"):
            parser.error("--out-subdir must not contain '..'")
        args.out_dir = root_out / sub
    else:
        args.out_dir = root_out

    _td = float(args.type1_d)
    if not (0.5 < _td < 2.0):
        parser.error(f"--type1-d must lie strictly in (1/2, 2); got {_td}")

    args.synthetic_call_from_puts, _syn_msg = resolve_synthetic_call_for_puts(
        args.option_type, args.synthetic_call_from_puts, args.raw_put_targets
    )
    if _syn_msg:
        print(_syn_msg)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        print(f"Error: checkpoint not found: {ckpt_path}")
        print(
            "Train first (e.g. train_full.py --save output/your_model.pt) or fix --checkpoint. "
            "The file is only written after training finishes."
        )
        raise SystemExit(1)
    args.checkpoint = ckpt_path

    print("Loading checkpoint...")
    if args.full_model:
        full_model, S0, tau_max, ckpt_args, ckpt_strike_band = load_checkpoint_full(args.checkpoint)
        pricing = full_model.kernel_net
        r = ckpt_args.get("r", args.r)
        n_r = ckpt_args.get("n_r", args.n_r)
        n_s = ckpt_args.get("n_s", args.n_s)
        n_u_full = ckpt_args.get("n_u", args.n_u)
    else:
        pricing, S0, tau_max, ckpt_args, ckpt_strike_band = load_checkpoint_simple(args.checkpoint)
        full_model = None
        r = ckpt_args.get("r", args.r)
        n_r = n_s = n_u_full = None

    kernel_for_plots = (
        full_model.kernel_net
        if args.full_model and full_model is not None
        else (pricing.kernel_net if isinstance(pricing, SimplifiedPricer) else pricing)
    )
    d_hdiv = _paper7_learned_d_float(kernel_for_plots)
    if d_hdiv is None:
        d_hdiv = float(args.type1_d)
    elif isinstance(kernel_for_plots, StructuredKernelNetPaper7):
        if getattr(kernel_for_plots, "d_fixed", None) is not None:
            print(f"Kernel h/u^(d-1) plots use fixed paper7 d = {d_hdiv:.6f}")
        else:
            print(f"Kernel h/u^(d-1) plots use learned paper7 d = {d_hdiv:.6f} (not --type1-d).")

    loss_tr = ckpt_args.get("loss")
    if loss_tr is None and ckpt_args.get("relative_loss"):
        loss_tr = "relative_mse"
    if loss_tr:
        print(f"Checkpoint training objective (loss): {loss_tr}")

    min_price = (
        float(args.min_price)
        if args.min_price is not None
        else float(ckpt_args.get("min_price", 0.0))
    )
    if min_price > 0:
        print(f"Evaluation filter: option mid >= {min_price}")

    print("Loading data...")
    df = load_options_data(
        data_dir=args.data_dir,
        option_type=args.option_type,
        strict_10days=args.strict_10days_data,
    )

    sb_info = ckpt_strike_band
    if args.strike_band is not None:
        rel_lo, rel_hi = float(args.strike_band[0]), float(args.strike_band[1])
        s_band = float(S0)
        n0 = len(df)
        df = apply_strike_band_by_spot_scalar(df, spot_scalar=s_band, rel_low=rel_lo, rel_high=rel_hi)
        print(
            f"Strike band (--strike-band vs checkpoint S0={s_band:g}): "
            f"{len(df)}/{n0} rows in ({rel_lo}*S, {rel_hi}*S)",
        )
    elif sb_info is not None:
        rel_lo = float(sb_info["rel_lo"])
        rel_hi = float(sb_info["rel_hi"])
        s_band = float(sb_info["spot_scalar"])
        n0 = len(df)
        df = apply_strike_band_by_spot_scalar(df, spot_scalar=s_band, rel_low=rel_lo, rel_high=rel_hi)
        print(
            f"Strike band from checkpoint ({rel_lo}*{s_band:.6g} < K < {rel_hi}*{s_band:.6g}): "
            f"{len(df)}/{n0} rows",
        )

    # Rebuild targets with the same ``S0_ref`` policy as training (``ckpt_args[''S0'']``), not the saved mean.
    s0_ref_eval = ckpt_args.get("S0")
    tau, k, K, price, _S0_mean_unused, S0_row = to_training_arrays(
        df,
        S0_ref=s0_ref_eval,
        min_price=min_price,
        r=args.r,
        synthetic_call_from_puts=args.synthetic_call_from_puts,
        s0_fallback=args.s0_fallback,
    )
    tau_norm = normalize_tau_for_net(tau, tau_max=tau_max)
    if len(tau) > args.max_eval:
        idx = np.random.choice(len(tau), args.max_eval, replace=False)
        tau_norm, k, K, price, S0_row = tau_norm[idx], k[idx], K[idx], price[idx], S0_row[idx]
    n = len(tau_norm)

    tau_t = torch.tensor(tau_norm, dtype=torch.float32, device=DEVICE)
    K_t = torch.tensor(K, dtype=torch.float32, device=DEVICE)
    s0_row_t = torch.tensor(S0_row, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        if args.full_model and full_model is not None:
            C_model = full_model(
                tau_t, K_t, r=r, S0=S0, tau_scale=tau_max,
                n_r=n_r or args.n_r, n_s=n_s or args.n_s, n_u=n_u_full or args.n_u,
            )
        elif isinstance(pricing, SimplifiedPricer):
            C_model = pricing(
                tau_t, K_t, s0_row_t, r, tau_max, n_grid=args.n_grid, n_u=args.n_u
            )
        else:
            C_model = model_call_prices(
                pricing,
                tau_t,
                K_t,
                r=r,
                S0=s0_row_t,
                n_grid=args.n_grid,
                n_u=args.n_u,
                tau_scale=tau_max,
            )
    C_model = C_model.cpu().numpy()
    C_market = price

    metrics = compute_metrics(C_model, C_market)
    print("Metrics:")
    for name, val in metrics.items():
        print(f"  {name}: {val:.6f}")

    suffix = "_full" if args.full_model else ""
    (args.out_dir / f"metrics{suffix}.txt").write_text(
        "\n".join(f"{k}: {v:.6f}" for k, v in metrics.items())
    )

    tau_phys = tau_norm * tau_max
    plot_market_vs_model(
        K, tau_phys, C_market, C_model,
        args.out_dir / f"market_vs_model{suffix}.png", max_points=3000,
    )
    tau_edges_list: list[float] | None = None
    if args.tau_bucket_edges:
        tau_edges_list = [float(x.strip()) for x in args.tau_bucket_edges.split(",") if x.strip()]
    strike_y_cap: float | None = args.call_vs_strike_y_max
    if strike_y_cap is None and args.full_model:
        strike_y_cap = 8000.0
    plot_call_price_vs_strike(
        K,
        tau_phys,
        C_market,
        C_model,
        args.out_dir / f"call_price_vs_strike{suffix}.png",
        min_per_panel=max(1, int(args.call_vs_strike_min_per_panel)),
        max_points_total=max(10, int(args.call_vs_strike_max_points)),
        tau_bucket_edges=tau_edges_list,
        y_price_max=strike_y_cap,
    )
    plot_errors_by_moneyness(k, C_market, C_model, args.out_dir / f"error_by_moneyness{suffix}.png")

    expect_nonnegative = _kernel_nonnegative_from_saved_args(ckpt_args)
    h_plot_min = max(0.0, float(args.kernel_h_plot_min))
    kern_fname_tag = f"_habsgt{h_plot_min:g}" if h_plot_min > 0 else ""
    kgrid = _kernel_grid_h(kernel_for_plots, n_tau=50, n_s=50)
    plot_kernel_heatmap(
        kernel_for_plots,
        tau_max,
        args.out_dir / f"kernel_heatmap{suffix}{kern_fname_tag}.png",
        grid=kgrid,
        h_abs_plot_min=h_plot_min,
    )
    t_h_list: list[float] | None = None
    if args.kernel_h_t_plus_u_t_years:
        t_h_list = [float(x.strip()) for x in args.kernel_h_t_plus_u_t_years.split(",") if x.strip()]
    plot_kernel_abs_h_t_plus_u(
        kernel_for_plots,
        tau_max,
        args.out_dir / f"kernel_abs_h_t_plus_u{suffix}{kern_fname_tag}.png",
        t_fixed_years=t_h_list,
        h_abs_plot_min=h_plot_min,
    )
    plot_kernel_h_div_u_pow_dminus1(
        kernel_for_plots,
        tau_max,
        args.out_dir / f"kernel_h_div_u_pow_dminus1{suffix}{kern_fname_tag}.png",
        d=d_hdiv,
        t_fixed_years=t_h_list,
        h_abs_plot_min=h_plot_min,
    )
    plot_kernel_h_div_u_pow_d(
        kernel_for_plots,
        tau_max,
        args.out_dir / f"kernel_h_div_u_pow_d{suffix}{kern_fname_tag}.png",
        d=d_hdiv,
        t_fixed_years=t_h_list,
        h_abs_plot_min=h_plot_min,
    )
    plot_kernel_heatmap_s_minus_tau(
        kernel_for_plots,
        tau_max,
        args.out_dir / f"kernel_heatmap_s_minus_tau{suffix}{kern_fname_tag}.png",
        grid=kgrid,
        h_abs_plot_min=h_plot_min,
    )
    plot_kernel_learned_vs_type1(
        kernel_for_plots,
        tau_max,
        args.out_dir / f"kernel_learned_vs_type1{suffix}{kern_fname_tag}.png",
        type1_d=args.type1_d,
        type1_kappa=args.type1_kappa,
        grid=kgrid,
        h_abs_plot_min=h_plot_min,
    )
    plot_kernel_constraint_diagnostics(
        kgrid[0],
        kgrid[1],
        kgrid[2],
        args.out_dir / f"kernel_constraint_check{suffix}.png",
        expect_nonnegative=expect_nonnegative,
    )
    eq7_stats = plot_kernel_eq7_constraint(
        kgrid[0],
        kgrid[1],
        kgrid[2],
        tau_max,
        args.out_dir / f"kernel_eq7_constraint{suffix}.png",
        d=args.type1_d,
        kappa=args.type1_kappa,
    )
    eq7_tail_stats = plot_kernel_eq7_asymptotic_tails(
        kernel_for_plots,
        tau_max,
        args.out_dir / f"kernel_eq7_asymptotics{suffix}{kern_fname_tag}.png",
        d=args.type1_d,
        kappa=args.type1_kappa,
        h_abs_plot_min=h_plot_min,
    )
    export_kernel_grid(kernel_for_plots, tau_max, args.out_dir / f"kernel_grid{suffix}.csv", grid=kgrid)

    print(f"Plots and kernel grid saved to {args.out_dir}")
    print(
        f"Kernel constraints: expect_nonnegative={expect_nonnegative} "
        f"(causal support + nonnegativity figure: kernel_constraint_check{suffix}.png)"
    )
    print(f"Eq. (7) structure figure: kernel_eq7_constraint{suffix}.png")
    print("Eq. (7) summary (d, kappa from --type1-d, --type1-kappa):")
    for key, val in eq7_stats.items():
        if isinstance(val, float) and not np.isnan(val):
            print(f"  {key}: {val:.6g}")
        elif isinstance(val, float):
            print(f"  {key}: nan")
    print(f"Eq. (7) asymptotic tails figure: kernel_eq7_asymptotics{suffix}{kern_fname_tag}.png")
    print("Eq. (7) asymptotic tails (small-u slope, large-u kappa):")
    for key, val in eq7_tail_stats.items():
        if isinstance(val, float) and not np.isnan(val):
            print(f"  {key}: {val:.6g}")
        elif isinstance(val, float):
            print(f"  {key}: nan")


if __name__ == "__main__":
    main()
