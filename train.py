"""
Step 6–8: Training loop. Loss = price MSE, relative MSE, or mean absolute relative error
  mean_K,T |C - market| / |market|
  + optional no-arbitrage penalties.

Usage:
  python train.py --data-dir ../data --epochs 2000 --loss relative_mae
"""
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from preprocess import (
    load_options_data,
    to_training_arrays,
    normalize_tau_for_net,
    split_by_quote_date,
    resolve_synthetic_call_for_puts,
    strike_band_scalar_for_filter,
    apply_strike_band_by_spot_scalar,
    DATA_DIR,
)
from model import (
    SimplifiedPricer,
    check_linear_weights_biases_nonnegative,
    linear_weight_bias_mins,
    model_call_prices,
)
from sensitivity_h import (
    format_h_sensitivity_log,
    kernel_h_epoch_sensitivity_postflight,
    kernel_h_epoch_sensitivity_preflight,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relative_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(1/N) * sum(((pred - target) / (|target| + eps))^2)."""
    denom = target.abs() + eps
    return ((pred - target) / denom).pow(2).mean()


def relative_mae(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(1/N) * sum |pred - target| / (|target| + eps) — calibration-style average abs relative error."""
    denom = target.abs() + eps
    return ((pred - target).abs() / denom).mean()


def compute_price_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_kind: str, eps: float = 1e-8
) -> torch.Tensor:
    if loss_kind == "mse":
        return nn.functional.mse_loss(pred, target)
    if loss_kind == "relative_mse":
        return relative_mse(pred, target, eps=eps)
    if loss_kind == "relative_mae":
        return relative_mae(pred, target, eps=eps)
    raise ValueError(f"Unknown loss_kind: {loss_kind}")


def train_epoch(
    pricer: SimplifiedPricer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tau_max: float,
    r: float,
    n_grid: int,
    n_u: int,
    loss_kind: str,
    lambda_convex: float,
    lambda_monotone: float,
) -> float:
    pricer.train()
    total_loss = 0.0
    n_batches = 0
    for tau_b, K_b, price_b, s0_b in loader:
        optimizer.zero_grad()
        C_model = pricer(tau_b, K_b, s0_b, r, tau_max, n_grid=n_grid, n_u=n_u)
        loss_price = compute_price_loss(C_model, price_b, loss_kind)

        loss = loss_price

        # Optional: no-arbitrage penalties (sample-based on model prices)
        if lambda_convex > 0 or lambda_monotone > 0:
            K_sorted, idx = torch.sort(K_b)
            tau_s = tau_b[idx]
            s0_s = s0_b[idx]
            C_s = pricer(tau_s, K_sorted, s0_s, r, tau_max, n_grid=n_grid, n_u=n_u)
            dK = K_sorted[1:] - K_sorted[:-1] + 1e-8
            dC = (C_s[1:] - C_s[:-1]) / dK
            if lambda_monotone > 0 and K_b.numel() >= 2:
                L_mono = torch.relu(dC).mean()  # penalize dC/dK > 0 (calls: dC/dK <= 0)
                loss = loss + lambda_monotone * L_mono
            if lambda_convex > 0 and K_b.numel() >= 3:
                d2C = (dC[1:] - dC[:-1]) / (dK[1:] + 1e-8)
                L_convex = torch.relu(-d2C).mean()  # penalize d²C/dK² < 0
                loss = loss + lambda_convex * L_convex

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch_loss(
    pricer: SimplifiedPricer,
    tau_t: torch.Tensor,
    K_t: torch.Tensor,
    price_t: torch.Tensor,
    s0_t: torch.Tensor,
    tau_max: float,
    r: float,
    n_grid: int,
    n_u: int,
    loss_kind: str,
) -> float:
    pricer.eval()
    C = pricer(tau_t, K_t, s0_t, r, tau_max, n_grid=n_grid, n_u=n_u)
    loss = compute_price_loss(C, price_t, loss_kind).item()
    pricer.train()
    return loss


def main():
    parser = argparse.ArgumentParser(description="Train kernel NN to fit option prices")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Path to CSV data")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (Adam)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64], help="Hidden layer sizes")
    parser.add_argument("--n-grid", type=int, default=64, help="Grid points for variance integral")
    parser.add_argument("--n-u", type=int, default=256, help="Grid points for Carr-Madan")
    parser.add_argument("--r", type=float, default=0.0, help="Risk-free rate")
    parser.add_argument("--S0", type=float, default=None, help="Spot (default: from data)")
    parser.add_argument(
        "--loss",
        type=str,
        choices=("mse", "relative_mse", "relative_mae"),
        default="mse",
        help="mse=price MSE; relative_mse=mean squared relative error; "
        "relative_mae=mean |C-market|/|market| (sum/K,T style objective, batch mean)",
    )
    parser.add_argument(
        "--relative-loss",
        action="store_true",
        help="Shortcut for --loss relative_mse (backward compatible)",
    )
    parser.add_argument("--lambda-convex", type=float, default=0.0, help="Penalty weight for convexity violation")
    parser.add_argument("--lambda-monotone", type=float, default=0.0, help="Penalty weight for dC/dK > 0")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples for speed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=Path, default=None, help="Path to save best model state")
    parser.add_argument("--log-every", type=int, default=100, help="Log loss every N epochs")
    parser.add_argument(
        "--log-h-sensitivity-every",
        type=int,
        default=0,
        help="If >0: each N epochs (and epoch 1), log functional delta of h on a fixed (s,tau) grid "
        "and first-order dh_mean vs kernel moves (Gaussian path: net(s,tau), tau>s).",
    )
    parser.add_argument(
        "--h-sensitivity-n-r",
        type=int,
        default=24,
        help="Grid resolution for --log-h-sensitivity-every (s or tau direction).",
    )
    parser.add_argument(
        "--h-sensitivity-n-s",
        type=int,
        default=24,
        help="Grid resolution for --log-h-sensitivity-every.",
    )
    parser.add_argument(
        "--time-split",
        action="store_true",
        help="Split by quote date into train/val/test (70/15/15); tune on val, report test at end",
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument(
        "--kernel-type",
        type=str,
        choices=("mlp", "paper7", "constant"),
        default="mlp",
        help="mlp=KernelNet; paper7=Wang–Xia-style u^{d-1}e^{-κu}·g; constant=fixed h (--constant-h)",
    )
    parser.add_argument(
        "--constant-h",
        type=float,
        default=0.6,
        help="kernel-type constant: fixed h on the active mask (Gaussian path); no kernel NN params.",
    )
    parser.add_argument(
        "--paper7-d-low",
        type=float,
        default=0.5,
        help="paper7: lower bound for learnable d in u^{d-1} (require < --paper7-d-high).",
    )
    parser.add_argument(
        "--paper7-d-high",
        type=float,
        default=2.0,
        help="paper7: upper bound for learnable d.",
    )
    parser.add_argument(
        "--paper7-d-fixed",
        type=float,
        default=None,
        help="paper7: fix d in u^{d-1} (no grad). Omit to learn d in (--paper7-d-low,--paper7-d-high).",
    )
    parser.add_argument(
        "--strict-10days-data",
        action="store_true",
        help="Load only final_call_no_madan_strict_10days_*.csv from --data-dir.",
    )
    parser.add_argument(
        "--check-linear-wb-positive",
        action="store_true",
        help="At logging epochs: print min(W), min(b) over nn.Linear; warn on any negative entry.",
    )
    parser.add_argument(
        "--assert-linear-wb-positive",
        action="store_true",
        help="At logging epochs: raise if any Linear weight or bias is negative.",
    )
    parser.add_argument(
        "--positive-linear-wb",
        action="store_true",
        help="Use softplus parametrization so every MLP Linear has effective W,b > 0 (kernel + strike scale).",
    )
    parser.add_argument(
        "--signed-kernel",
        action="store_true",
        help="Allow signed h (disable softplus on mlp). Default: h≥0 via softplus.",
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=0.5,
        help="Kernel MLP output scale (larger → more σ² dynamic range; paper7 uses as g_scale floor)",
    )
    parser.add_argument(
        "--no-strike-scale",
        action="store_true",
        help="Disable strike/moneyness MLP factor on σ² (τ-only variance from kernel).",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.05,
        help="Drop training rows with option mid below this (stabilizes relative losses; 0 disables).",
    )
    parser.add_argument(
        "--option-type",
        type=str,
        choices=("C", "P"),
        default="C",
        help="Which options to keep (aligned BTC exports are often puts: use P).",
    )
    parser.add_argument(
        "--synthetic-call-from-puts",
        action="store_true",
        help="Map put mids to call targets via C = P + S0 - K e^{-rτ} (model prices calls). Uses --r.",
    )
    parser.add_argument(
        "--raw-put-targets",
        action="store_true",
        help="With --option-type P, do NOT auto-enable parity: use put mids as targets (expert; "
        "inconsistent with call-valued pricer unless you change the model).",
    )
    parser.add_argument(
        "--s0-fallback",
        type=str,
        choices=("mean_strike", "daily_median_strike", "daily_vwap_strike"),
        default="mean_strike",
        help="Spot per row when CSV has no underlying bid/ask: strike statistics (not option premium).",
    )
    parser.add_argument(
        "--strike-band",
        type=float,
        nargs=2,
        metavar=("REL_LOW", "REL_HIGH"),
        default=None,
        help="Optional ATM band: keep rows with REL_LOW*S < strike < REL_HIGH*S, "
        "S = --S0 if set else mean strike (e.g. 0.6 1.4).",
    )
    args = parser.parse_args()
    if args.relative_loss:
        args.loss = "relative_mse"
    if args.paper7_d_fixed is not None and args.kernel_type != "paper7":
        parser.error("--paper7-d-fixed requires --kernel-type paper7")
    if args.kernel_type == "paper7":
        if args.paper7_d_fixed is None:
            if not (0.0 < args.paper7_d_low < args.paper7_d_high):
                parser.error(
                    "paper7: --paper7-d-low and --paper7-d-high must satisfy 0 < low < high "
                    "(or set --paper7-d-fixed)"
                )
        elif not (float(args.paper7_d_fixed) > 0.0):
            parser.error("--paper7-d-fixed must be > 0")
    if args.kernel_type == "constant" and float(args.constant_h) <= 0.0:
        parser.error("--constant-h must be > 0")
    args.kernel_nonnegative = not args.signed_kernel
    args.use_strike_scale = not args.no_strike_scale

    args.synthetic_call_from_puts, _syn_msg = resolve_synthetic_call_for_puts(
        args.option_type, args.synthetic_call_from_puts, args.raw_put_targets
    )
    if _syn_msg:
        print(_syn_msg)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load and preprocess
    print("Loading options data...")
    if args.strict_10days_data:
        print("Dataset: final_call_no_madan_strict_10days_*.csv only")
    if args.min_price > 0:
        print(f"Filtering: option mid >= {args.min_price} (for stable relative-error training)")
    df = load_options_data(
        data_dir=args.data_dir,
        option_type=args.option_type,
        strict_10days=args.strict_10days_data,
    )
    print(f"Loaded {len(df)} option rows after type/mid filters")
    strike_band_info: dict | None = None
    if args.strike_band is not None:
        rel_lo, rel_hi = float(args.strike_band[0]), float(args.strike_band[1])
        s_band = strike_band_scalar_for_filter(df, args.S0)
        n0 = len(df)
        df = apply_strike_band_by_spot_scalar(df, spot_scalar=s_band, rel_low=rel_lo, rel_high=rel_hi)
        strike_band_info = {
            "rel_lo": rel_lo,
            "rel_hi": rel_hi,
            "spot_scalar": s_band,
            "rows_before": n0,
            "rows_after": len(df),
        }
        print(
            f"Strike band ({rel_lo}*S,{rel_hi}*S), S={s_band:.6g}: "
            f"{len(df)}/{n0} rows"
        )

    test_tau_t = test_K_t = test_price_t = None
    val_tau_t = val_K_t = val_price_t = None

    if args.time_split:
        tr_df, va_df, te_df = split_by_quote_date(
            df, args.train_frac, args.val_frac, args.test_frac
        )
        t_arr_kw = dict(
            min_price=args.min_price,
            r=args.r,
            synthetic_call_from_puts=args.synthetic_call_from_puts,
            s0_fallback=args.s0_fallback,
        )
        tau_tr, k_tr, K_tr, price_tr, S0, S0_tr = to_training_arrays(
            tr_df, S0_ref=args.S0, **t_arr_kw
        )
        tau_va, _, K_va, price_va, _, S0_va = to_training_arrays(
            va_df, S0_ref=args.S0, **t_arr_kw
        )
        tau_te, _, K_te, price_te, _, S0_te = to_training_arrays(
            te_df, S0_ref=args.S0, **t_arr_kw
        )
        tau = tau_tr
        k, K, price = k_tr, K_tr, price_tr
        S0_row = S0_tr
        if args.max_samples is not None and len(tau) > args.max_samples:
            idx = np.random.choice(len(tau), args.max_samples, replace=False)
            tau, k, K, price, S0_row = tau[idx], k[idx], K[idx], price[idx], S0_row[idx]
        tau_max = float(
            max(
                tau.max() if len(tau) else 0.0,
                tau_va.max() if len(tau_va) else 0.0,
                tau_te.max() if len(tau_te) else 0.0,
            )
        )
        tau_max = max(tau_max, 1e-9)
        tau_norm = normalize_tau_for_net(tau, tau_max=tau_max)
        val_tau_t = torch.tensor(normalize_tau_for_net(tau_va, tau_max=tau_max), dtype=torch.float32, device=DEVICE)
        val_K_t = torch.tensor(K_va, dtype=torch.float32, device=DEVICE)
        val_price_t = torch.tensor(price_va, dtype=torch.float32, device=DEVICE)
        val_s0_t = torch.tensor(S0_va, dtype=torch.float32, device=DEVICE)
        test_tau_t = torch.tensor(normalize_tau_for_net(tau_te, tau_max=tau_max), dtype=torch.float32, device=DEVICE)
        test_K_t = torch.tensor(K_te, dtype=torch.float32, device=DEVICE)
        test_price_t = torch.tensor(price_te, dtype=torch.float32, device=DEVICE)
        test_s0_t = torch.tensor(S0_te, dtype=torch.float32, device=DEVICE)
        print(
            f"Time split: train={len(tau)}, val={len(tau_va)}, test={len(tau_te)}; "
            f"S0={S0}"
        )
    else:
        tau, k, K, price, S0, S0_row = to_training_arrays(
            df,
            S0_ref=args.S0,
            min_price=args.min_price,
            r=args.r,
            synthetic_call_from_puts=args.synthetic_call_from_puts,
            s0_fallback=args.s0_fallback,
        )
        if args.max_samples is not None and len(tau) > args.max_samples:
            idx = np.random.choice(len(tau), args.max_samples, replace=False)
            tau, k, K, price, S0_row = tau[idx], k[idx], K[idx], price[idx], S0_row[idx]
        tau_max = float(np.max(tau)) if len(tau) else 1.0
        tau_max = max(tau_max, 1e-9)
        tau_norm = normalize_tau_for_net(tau, tau_max=tau_max)
        print(f"Using S0 = {S0}, n_samples = {len(tau)}")

    tau_t = torch.tensor(tau_norm, dtype=torch.float32, device=DEVICE)
    K_t = torch.tensor(K, dtype=torch.float32, device=DEVICE)
    price_t = torch.tensor(price, dtype=torch.float32, device=DEVICE)
    s0_row_t = torch.tensor(S0_row, dtype=torch.float32, device=DEVICE)

    dataset = TensorDataset(tau_t, K_t, price_t, s0_row_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    pricer = SimplifiedPricer(
        hidden_dims=args.hidden,
        kernel_type=args.kernel_type,
        kernel_nonnegative=args.kernel_nonnegative,
        output_scale=args.output_scale,
        use_strike_scale=args.use_strike_scale,
        device=DEVICE,
        paper7_d_low=args.paper7_d_low,
        paper7_d_high=args.paper7_d_high,
        paper7_d_fixed=args.paper7_d_fixed,
        positive_linear_wb=args.positive_linear_wb,
        constant_h=args.constant_h,
    )
    optimizer = torch.optim.Adam(pricer.parameters(), lr=args.lr)

    best_metric = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        h_sens_pre = None
        if args.log_h_sensitivity_every > 0 and (
            epoch % args.log_h_sensitivity_every == 0 or epoch == 1
        ):
            h_sens_pre = kernel_h_epoch_sensitivity_preflight(
                pricer.kernel_net,
                device=DEVICE,
                dtype=torch.float32,
                n_r=args.h_sensitivity_n_r,
                n_s=args.h_sensitivity_n_s,
                order="st",
            )
        avg_loss = train_epoch(
            pricer, loader, optimizer,
            tau_max=tau_max, r=args.r,
            n_grid=args.n_grid, n_u=args.n_u,
            loss_kind=args.loss,
            lambda_convex=args.lambda_convex,
            lambda_monotone=args.lambda_monotone,
        )
        if h_sens_pre is not None:
            h_met = kernel_h_epoch_sensitivity_postflight(pricer.kernel_net, h_sens_pre)
            print(format_h_sensitivity_log(h_met))
        if args.time_split and val_tau_t is not None and val_tau_t.numel() > 0:
            val_loss = eval_epoch_loss(
                pricer, val_tau_t, val_K_t, val_price_t, val_s0_t, tau_max, args.r,
                args.n_grid, args.n_u, args.loss,
            )
            metric = val_loss
            if (epoch % args.log_every == 0) or epoch == 1:
                print(f"Epoch {epoch}/{args.epochs}  train_loss = {avg_loss:.6f}  val_loss = {val_loss:.6f}")
        else:
            metric = avg_loss
            if (epoch % args.log_every == 0) or epoch == 1:
                print(f"Epoch {epoch}/{args.epochs}  loss = {avg_loss:.6f}")

        if (args.check_linear_wb_positive or args.assert_linear_wb_positive) and (
            (epoch % args.log_every == 0) or epoch == 1
        ):
            wm, bm = linear_weight_bias_mins(pricer)
            if wm is not None:
                bstr = f"{bm:.6g}" if bm is not None else "nan"
                print(f"  Linear min(W)={wm:.6g} min(b)={bstr}")
            check_linear_weights_biases_nonnegative(
                pricer, strict=args.assert_linear_wb_positive, prefix=""
            )

        if metric < best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in pricer.state_dict().items()}

    if best_state is not None:
        pricer.load_state_dict(best_state)

    # Final eval (train or test set)
    pricer.eval()
    with torch.no_grad():
        if args.time_split and test_tau_t is not None and test_tau_t.numel() > 0:
            n_eval = len(test_tau_t)
            C = pricer(
                test_tau_t, test_K_t, test_s0_t, args.r, tau_max, n_grid=args.n_grid, n_u=args.n_u
            )
            rel_err = (C - test_price_t).abs() / (test_price_t.abs() + 1e-8)
            print(f"\nBest {'val' if args.time_split else 'train'} metric = {best_metric:.6f}")
            print(f"Test set (out-of-sample): n={n_eval}, rel_err mean = {rel_err.mean().item():.4f}")
        else:
            n_eval = min(2000, len(tau_t))
            C = pricer(
                tau_t[:n_eval], K_t[:n_eval], s0_row_t[:n_eval], args.r, tau_max, n_grid=args.n_grid, n_u=args.n_u
            )
            rel_err = (C - price_t[:n_eval]).abs() / (price_t[:n_eval].abs() + 1e-8)
            print(f"\nBest loss = {best_metric:.6f}")
            print(f"Sample relative error (first {n_eval}): mean = {rel_err.mean().item():.4f}, max = {rel_err.max().item():.4f}")

    if args.save is not None:
        save_path = Path(args.save).expanduser()
        if not save_path.is_absolute():
            save_path = SCRIPT_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_str = str(save_path.resolve())
        try:
            torch.save(
                {
                    "simplified_pricer_state": pricer.state_dict(),
                    "net_state": pricer.kernel_net.state_dict(),
                    "S0": S0,
                    "tau_max": tau_max,
                    "args": vars(args),
                    "best_metric": best_metric,
                    "time_split": args.time_split,
                    "strike_band": strike_band_info,
                    "use_per_row_s0": True,
                },
                save_str,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to write checkpoint to {save_str!r} (cwd={Path.cwd()!s}). "
                f"Create the folder manually or use an absolute path. Original error: {e}"
            ) from e
        print(f"Saved best model to {save_str}")


if __name__ == "__main__":
    main()
