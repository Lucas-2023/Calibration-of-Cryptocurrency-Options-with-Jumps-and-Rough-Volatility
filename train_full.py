"""
Train the FULL math model: kernel + mean reversion + tempered stable.
Saves the best checkpoint (by val loss if --time-split, else train loss).

Price objective: --loss mse | relative_mse | relative_mae (mean |C-market|/|market|).
"""
from pathlib import Path
import argparse
import copy
import csv
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from train import compute_price_loss
from preprocess import (
    load_options_data,
    to_training_arrays,
    normalize_tau_for_net,
    split_by_quote_date,
    resolve_synthetic_call_for_puts,
    DATA_DIR,
)
from model import build_kernel_net, check_linear_weights_biases_nonnegative, linear_weight_bias_mins
from model_full import FullOptionModel
from sensitivity_h import (
    append_h_sensitivity_csv,
    format_h_sensitivity_log,
    kernel_h_epoch_sensitivity_postflight,
    kernel_h_epoch_sensitivity_preflight,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_output_path(path: Path | str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p


CONVERGENCE_CSV_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "selection_metric",
    "best_metric_so_far",
    "V0",
    "V_bar",
    "k",
    "a",
    "b",
    "c",
]


def append_full_model_convergence_csv(
    csv_path: Path,
    *,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    selection_metric: float,
    best_metric_so_far: float,
    V0: float,
    V_bar: float,
    k: float,
    a: float,
    b: float,
    c: float,
) -> None:
    """Append one row per epoch for loss / scalar convergence tracking."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    row = {
        "epoch": str(int(epoch)),
        "train_loss": f"{float(train_loss):.10g}",
        "val_loss": "" if val_loss is None else f"{float(val_loss):.10g}",
        "selection_metric": f"{float(selection_metric):.10g}",
        "best_metric_so_far": f"{float(best_metric_so_far):.10g}",
        "V0": f"{float(V0):.10g}",
        "V_bar": f"{float(V_bar):.10g}",
        "k": f"{float(k):.10g}",
        "a": f"{float(a):.10g}",
        "b": f"{float(b):.10g}",
        "c": f"{float(c):.10g}",
    }
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CONVERGENCE_CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def train_epoch_full(
    full_model: FullOptionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    S0: float,
    tau_max: float,
    r: float,
    n_r: int,
    n_s: int,
    n_u: int,
    loss_kind: str,
    enforce_nonnegative: bool,
) -> float:
    full_model.train()
    total_loss = 0.0
    n_batches = 0
    for tau_b, K_b, price_b in loader:
        optimizer.zero_grad()
        C_model = full_model(
            tau_b, K_b, r=r, S0=S0, tau_scale=tau_max, n_r=n_r, n_s=n_s, n_u=n_u,
            enforce_nonnegative=enforce_nonnegative,
        )
        loss = compute_price_loss(C_model, price_b, loss_kind)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_full_loss(
    full_model: FullOptionModel,
    tau_t: torch.Tensor,
    K_t: torch.Tensor,
    price_t: torch.Tensor,
    S0: float,
    tau_max: float,
    r: float,
    n_r: int,
    n_s: int,
    n_u: int,
    loss_kind: str,
    enforce_nonnegative: bool,
) -> float:
    full_model.eval()
    C = full_model(
        tau_t, K_t, r=r, S0=S0, tau_scale=tau_max, n_r=n_r, n_s=n_s, n_u=n_u,
        enforce_nonnegative=enforce_nonnegative,
    )
    loss = compute_price_loss(C, price_t, loss_kind).item()
    full_model.train()
    return loss


def main():
    parser = argparse.ArgumentParser(description="Train full math model")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--n-r", type=int, default=20)
    parser.add_argument("--n-s", type=int, default=20)
    parser.add_argument("--n-u", type=int, default=128)
    parser.add_argument("--r", type=float, default=0.0)
    parser.add_argument("--S0", type=float, default=None)
    parser.add_argument(
        "--loss",
        type=str,
        choices=("mse", "relative_mse", "relative_mae"),
        default="mse",
        help="mse | relative_mse | relative_mae (mean abs relative price error)",
    )
    parser.add_argument(
        "--relative-loss",
        action="store_true",
        help="Shortcut for --loss relative_mse",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--log-h-sensitivity-every",
        type=int,
        default=0,
        help="If >0: each N epochs (and epoch 1), log functional delta of h on a fixed (r,s) grid "
        "and first-order dh_mean vs kernel parameter move (full model: net(r,s), s>r).",
    )
    parser.add_argument(
        "--h-sensitivity-n-r",
        type=int,
        default=24,
        help="Grid resolution (r or s direction) for --log-h-sensitivity-every.",
    )
    parser.add_argument(
        "--h-sensitivity-n-s",
        type=int,
        default=24,
        help="Grid resolution (s direction) for --log-h-sensitivity-every.",
    )
    parser.add_argument(
        "--h-sensitivity-csv",
        type=Path,
        default=None,
        help="If set (with --log-h-sensitivity-every > 0): append epoch-wise h metrics to this CSV for plotting.",
    )
    parser.add_argument(
        "--convergence-csv",
        type=Path,
        default=None,
        help="If set: append every epoch — train_loss, val_loss (if time-split), selection metric, "
        "best metric so far, and learned scalars (V0,V_bar,k,a,b,c) for convergence plots.",
    )
    parser.add_argument("--time-split", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument(
        "--kernel-type",
        type=str,
        choices=("mlp", "paper7", "constant"),
        default="mlp",
        help="mlp=KernelNet; paper7=Wang–Xia Eq.(7)-style (h≥0); constant=fixed h (see --constant-h)",
    )
    parser.add_argument(
        "--constant-h",
        type=float,
        default=0.6,
        help="kernel-type constant: fixed h on s>r (first=r, second=s); no kernel NN parameters.",
    )
    parser.add_argument(
        "--paper7-d-low",
        type=float,
        default=0.5,
        help="paper7: lower bound for learnable d in factor u^{d-1} (require < --paper7-d-high).",
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
        help="paper7: fix exponent d in u^{d-1} (no grad). Omit to learn d in (--paper7-d-low,--paper7-d-high).",
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
        help="Use softplus parametrization so kernel MLP Linears have effective W,b > 0 (mlp and paper7 g-net).",
    )
    parser.add_argument(
        "--signed-kernel",
        action="store_true",
        help="Allow signed h on mlp (disable softplus). Default: nonnegative via softplus.",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.05,
        help="Drop rows with option mid below this (0 disables).",
    )
    parser.add_argument(
        "--option-type",
        type=str,
        choices=("C", "P"),
        default="C",
        help="Which options to keep (aligned exports are often puts: use P).",
    )
    parser.add_argument(
        "--synthetic-call-from-puts",
        action="store_true",
        help="Map put mids to call targets via put-call parity (model prices calls). Uses --r.",
    )
    parser.add_argument(
        "--raw-put-targets",
        action="store_true",
        help="With --option-type P, skip auto parity; use put mids as targets (see train.py).",
    )
    parser.add_argument(
        "--s0-fallback",
        type=str,
        choices=("mean_strike", "daily_median_strike", "daily_vwap_strike"),
        default="mean_strike",
        help="Spot per row when no underlying bid/ask in CSV (see preprocess.to_training_arrays).",
    )
    parser.add_argument(
        "--disable-price-clamp",
        action="store_true",
        help="Do not clamp Carr-Madan prices at zero (debug numerical issues).",
    )
    parser.add_argument(
        "--debug-pricer",
        action="store_true",
        help="Print raw Carr-Madan C stats (min/mean/max/negative ratio) at logging steps.",
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
    enforce_nonnegative = not args.disable_price_clamp

    args.synthetic_call_from_puts, _syn_msg = resolve_synthetic_call_for_puts(
        args.option_type, args.synthetic_call_from_puts, args.raw_put_targets
    )
    if _syn_msg:
        print(_syn_msg)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading options data...")
    if args.strict_10days_data:
        print("Dataset: final_call_no_madan_strict_10days_*.csv only")
    if args.min_price > 0:
        print(f"Filtering: option mid >= {args.min_price}")
    df = load_options_data(
        data_dir=args.data_dir,
        option_type=args.option_type,
        strict_10days=args.strict_10days_data,
    )
    print(f"Loaded {len(df)} option rows after type/mid filters")
    test_tau_t = test_K_t = test_price_t = None
    val_tau_t = val_K_t = val_price_t = None

    t_arr_kw = dict(
        min_price=args.min_price,
        r=args.r,
        synthetic_call_from_puts=args.synthetic_call_from_puts,
        s0_fallback=args.s0_fallback,
    )
    if args.time_split:
        tr_df, va_df, te_df = split_by_quote_date(
            df, args.train_frac, args.val_frac, args.test_frac
        )
        tau_tr, k_tr, K_tr, price_tr, S0, _ = to_training_arrays(
            tr_df, S0_ref=args.S0, **t_arr_kw
        )
        tau_va, _, K_va, price_va, _, _ = to_training_arrays(
            va_df, S0_ref=args.S0, **t_arr_kw
        )
        tau_te, _, K_te, price_te, _, _ = to_training_arrays(
            te_df, S0_ref=args.S0, **t_arr_kw
        )
        tau, k, K, price = tau_tr, k_tr, K_tr, price_tr
        if args.max_samples is not None and len(tau) > args.max_samples:
            idx = np.random.choice(len(tau), args.max_samples, replace=False)
            tau, k, K, price = tau[idx], k[idx], K[idx], price[idx]
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
        test_tau_t = torch.tensor(normalize_tau_for_net(tau_te, tau_max=tau_max), dtype=torch.float32, device=DEVICE)
        test_K_t = torch.tensor(K_te, dtype=torch.float32, device=DEVICE)
        test_price_t = torch.tensor(price_te, dtype=torch.float32, device=DEVICE)
        print(f"Time split: train={len(tau)}, val={len(tau_va)}, test={len(tau_te)}")
    else:
        tau, k, K, price, S0, _ = to_training_arrays(df, S0_ref=args.S0, **t_arr_kw)
        if args.max_samples is not None and len(tau) > args.max_samples:
            idx = np.random.choice(len(tau), args.max_samples, replace=False)
            tau, k, K, price = tau[idx], k[idx], K[idx], price[idx]
        tau_max = float(np.max(tau)) if len(tau) else 1.0
        tau_max = max(tau_max, 1e-9)
        tau_norm = normalize_tau_for_net(tau, tau_max=tau_max)
        print(f"Using S0 = {S0}, n_samples = {len(tau)}")

    tau_t = torch.tensor(tau_norm, dtype=torch.float32, device=DEVICE)
    K_t = torch.tensor(K, dtype=torch.float32, device=DEVICE)
    price_t = torch.tensor(price, dtype=torch.float32, device=DEVICE)

    dataset = TensorDataset(tau_t, K_t, price_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    kernel_net = build_kernel_net(
        hidden_dims=args.hidden,
        kernel_type=args.kernel_type,
        kernel_nonnegative=args.kernel_nonnegative,
        device=DEVICE,
        paper7_d_low=args.paper7_d_low,
        paper7_d_high=args.paper7_d_high,
        paper7_d_fixed=args.paper7_d_fixed,
        positive_linear_wb=args.positive_linear_wb,
        constant_h=args.constant_h,
    )
    full_model = FullOptionModel(kernel_net).to(DEVICE)
    optimizer = torch.optim.Adam(full_model.parameters(), lr=args.lr)
    if args.convergence_csv is not None:
        print(f"Convergence CSV: {_resolve_output_path(args.convergence_csv)}")

    best_metric = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        h_sens_pre = None
        if args.log_h_sensitivity_every > 0 and (
            epoch % args.log_h_sensitivity_every == 0 or epoch == 1
        ):
            h_sens_pre = kernel_h_epoch_sensitivity_preflight(
                full_model.kernel_net,
                device=DEVICE,
                dtype=torch.float32,
                n_r=args.h_sensitivity_n_r,
                n_s=args.h_sensitivity_n_s,
                order="rs",
            )
        avg_loss = train_epoch_full(
            full_model, loader, optimizer,
            S0=S0, tau_max=tau_max, r=args.r,
            n_r=args.n_r, n_s=args.n_s, n_u=args.n_u,
            loss_kind=args.loss,
            enforce_nonnegative=enforce_nonnegative,
        )
        val_loss_ep: float | None = None
        if args.time_split and val_tau_t is not None and val_tau_t.numel() > 0:
            val_loss_ep = eval_full_loss(
                full_model, val_tau_t, val_K_t, val_price_t, S0, tau_max, args.r,
                args.n_r, args.n_s, args.n_u, args.loss, enforce_nonnegative,
            )
            metric = val_loss_ep
        else:
            metric = avg_loss

        if h_sens_pre is not None:
            h_met = kernel_h_epoch_sensitivity_postflight(full_model.kernel_net, h_sens_pre)
            print(format_h_sensitivity_log(h_met))
            if args.h_sensitivity_csv is not None:
                append_h_sensitivity_csv(
                    _resolve_output_path(args.h_sensitivity_csv),
                    epoch=epoch,
                    metrics=h_met,
                    train_loss=avg_loss,
                    val_loss=val_loss_ep,
                )

        if (epoch % args.log_every == 0) or epoch == 1:
            if val_loss_ep is not None:
                print(f"Epoch {epoch}/{args.epochs}  train={avg_loss:.6f}  val={val_loss_ep:.6f}")
            else:
                print(f"Epoch {epoch}/{args.epochs}  loss = {avg_loss:.6f}")

        if args.debug_pricer and ((epoch % args.log_every == 0) or epoch == 1):
            full_model.eval()
            n_dbg = min(128, len(tau_t))
            _ = full_model(
                tau_t[:n_dbg], K_t[:n_dbg], r=args.r, S0=S0, tau_scale=tau_max,
                n_r=args.n_r, n_s=args.n_s, n_u=args.n_u,
                enforce_nonnegative=False, debug_stats=True,
            )
            full_model.train()

        if (args.check_linear_wb_positive or args.assert_linear_wb_positive) and (
            (epoch % args.log_every == 0) or epoch == 1
        ):
            wm, bm = linear_weight_bias_mins(full_model)
            if wm is not None:
                print(f"  Linear min(W)={wm:.6g} min(b)={bm if bm is not None else float('nan'):.6g}")
            check_linear_weights_biases_nonnegative(
                full_model, strict=args.assert_linear_wb_positive, prefix=""
            )

        if metric < best_metric:
            best_metric = metric
            best_state = copy.deepcopy(full_model.state_dict())

        if args.convergence_csv is not None:
            V0e, Vbare, ke, ae, be, ce = full_model.get_params()
            append_full_model_convergence_csv(
                _resolve_output_path(args.convergence_csv),
                epoch=epoch,
                train_loss=avg_loss,
                val_loss=val_loss_ep,
                selection_metric=metric,
                best_metric_so_far=best_metric,
                V0=V0e,
                V_bar=Vbare,
                k=ke,
                a=ae,
                b=be,
                c=ce,
            )

    if best_state is not None:
        full_model.load_state_dict(best_state)

    full_model.eval()
    with torch.no_grad():
        if args.time_split and test_tau_t is not None and test_tau_t.numel() > 0:
            C = full_model(
                test_tau_t, test_K_t, r=args.r, S0=S0, tau_scale=tau_max,
                n_r=args.n_r, n_s=args.n_s, n_u=args.n_u,
                enforce_nonnegative=enforce_nonnegative,
            )
            rel_err = (C - test_price_t).abs() / (test_price_t.abs() + 1e-8)
            print(f"\nBest metric = {best_metric:.6f}")
            print(f"Test rel_err mean = {rel_err.mean().item():.4f}")
        else:
            n_eval = min(500, len(tau_t))
            C = full_model(
                tau_t[:n_eval], K_t[:n_eval], r=args.r, S0=S0, tau_scale=tau_max,
                n_r=args.n_r, n_s=args.n_s, n_u=args.n_u,
                enforce_nonnegative=enforce_nonnegative,
            )
            rel_err = (C - price_t[:n_eval]).abs() / (price_t[:n_eval].abs() + 1e-8)
            print(f"\nBest loss = {best_metric:.6f}")
            print(f"Sample relative error: mean = {rel_err.mean().item():.4f}")
        V0, V_bar, k, a, b, c = full_model.get_params()
        print(f"Learned params: V0={V0:.4f}, V_bar={V_bar:.4f}, k={k:.4f}, a={a:.4f}, b={b:.4f}, c={c:.4f}")

    if args.save is not None:
        save_path = Path(args.save).expanduser()
        if not save_path.is_absolute():
            save_path = SCRIPT_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_str = str(save_path.resolve())
        try:
            torch.save(
                {
                    "kernel_net_state": kernel_net.state_dict(),
                    "full_model_state": full_model.state_dict(),
                    "S0": S0,
                    "tau_max": tau_max,
                    "args": vars(args),
                    "best_metric": best_metric,
                    "time_split": args.time_split,
                },
                save_str,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to write checkpoint to {save_str!r} (cwd={Path.cwd()!s}). "
                f"Create the folder manually or use an absolute path under a short ASCII path. "
                f"Original error: {e}"
            ) from e
        print(f"Saved best model to {save_str}")


if __name__ == "__main__":
    main()
