"""
Show the "content" of the learned kernel: it has no closed form f(τ,s)=...,
but you can see its structure and evaluate it at any (τ, s).

Usage:
  python inspect_kernel.py --checkpoint output/best_kernel.pt
"""
from pathlib import Path
import argparse
import torch

from model import SimplifiedPricer, build_kernel_net

SCRIPT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")


def _kernel_nonnegative_from_saved_args(ckpt_args: dict) -> bool:
    if ckpt_args.get("kernel_type") == "paper7":
        return True
    return bool(ckpt_args.get("kernel_nonnegative", False))


def load_net(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    args = ckpt.get("args", {})
    hidden = args.get("hidden", [64, 64, 64])
    ktype = args.get("kernel_type", "mlp")
    out_sc = float(args.get("output_scale", 0.2))
    d_lo = float(args.get("paper7_d_low", 0.5))
    d_hi = float(args.get("paper7_d_high", 2.0))
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
            use_strike_scale=bool(args.get("use_strike_scale", True)),
            device=DEVICE,
            paper7_d_low=d_lo,
            paper7_d_high=d_hi,
            paper7_d_fixed=d_fix_f,
            positive_linear_wb=pos_wb,
            constant_h=c_h,
        )
        pricer.load_state_dict(ckpt["simplified_pricer_state"])
        pricer.eval()
        return pricer.kernel_net
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
    if state:
        net.load_state_dict(state)
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR / "output" / "best_kernel.pt")
    parser.add_argument("--tau", type=float, default=None, help="Example τ to evaluate")
    parser.add_argument("--s", type=float, default=None, help="Example s to evaluate")
    args = parser.parse_args()

    net = load_net(args.checkpoint)
    print("=" * 60)
    print("KERNEL — mask: second > first. Full model: net(r,s), s>r. Gaussian σ² path: net(s,τ), τ>s.")
    print("=" * 60)
    print()
    print("There is no closed-form formula like 'f(τ,s) = ...' in elementary functions.")
    print("The kernel is defined by the neural network below.")
    print()
    print("SYMBOLIC FORM (depends on checkpoint kernel_type):")
    print("  mlp:    h = mask * output_scale * softplus(MLP([·,·]))  unless trained with --signed-kernel")
    print("  paper7: h = mask * u^{d-1} e^{-κ u} * softplus(MLP([·,·])),  u = second - first,  d∈(1/2,1)")
    print("  and MLP is:")
    print("    x0 = [τ, s]  (2 inputs)")
    print("    x1 = ReLU( W0 @ x0 + b0 )           # shape: 64")
    print("    x2 = ReLU( W1 @ x1 + b1 )          # shape: 64")
    print("    x3 = ReLU( W2 @ x2 + b2 )          # shape: 64")
    print("    out = W3 @ x3 + b3                  # shape: 1")
    print("  So h = output_scale * 𝟙(second>first) * out  (names: forward(tau,s) params)")
    print()
    print("With your saved model: input_scale = 1.0, output_scale = 0.2")
    print()
    print("WEIGHT SHAPES (the 'content' of the function):")
    for name, p in net.named_parameters():
        print(f"  {name}: {tuple(p.shape)}")
    print()
    print("Gaussian σ² path: pass net(s, τ) with τ>s, e.g. s=0.3, τ=0.5 (normalized).")

    s_ex = torch.tensor([[0.3]], dtype=torch.float32)
    tau_ex = torch.tensor([[0.5]], dtype=torch.float32)
    with torch.no_grad():
        h_val = net(s_ex, tau_ex).item()
    print(f"  net(s=0.3, τ=0.5) = {h_val}")
    print()
    if args.tau is not None and args.s is not None:
        t, s_val = args.tau, args.s
        s_t = torch.tensor([[s_val]], dtype=torch.float32)
        tau_t = torch.tensor([[t]], dtype=torch.float32)
        with torch.no_grad():
            h_val = net(s_t, tau_t).item()
        print(f"  net(s={s_val}, τ={t}) = {h_val}")
    print()
    print("Full grid of values is in: output/kernel_grid.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
