"""
Overlay h / u^p for two checkpoints (e.g. paper7 with d in (0.5,1) vs (0.5,2)).

Example:
  py -3 plot_compare_h_div_u_pow.py \\
    --ckpt-narrow output/full_model_d05_1.pt \\
    --ckpt-wide output/full_model_d05_2.pt \\
    --out output/compare_h_div_u_sqrtu.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate import load_checkpoint_full, load_checkpoint_simple, plot_kernel_h_div_u_pow_compare
from model import SimplifiedPricer

SCRIPT_DIR = Path(__file__).resolve().parent


def _kernel_tau(ckpt_path: Path, full: bool) -> tuple[torch.nn.Module, float]:
    if full:
        full_model, _S0, tau_max, _args, _ = load_checkpoint_full(ckpt_path)
        return full_model.kernel_net, float(tau_max)
    net_obj, _S0, tau_max, _args, _ = load_checkpoint_simple(ckpt_path)
    if isinstance(net_obj, SimplifiedPricer):
        return net_obj.kernel_net, float(tau_max)
    return net_obj, float(tau_max)


def _is_full_checkpoint(path: Path) -> bool:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("full_model_state") is not None


def _kernel_tau_auto(ckpt_path: Path) -> tuple[torch.nn.Module, float]:
    full = _is_full_checkpoint(ckpt_path)
    return _kernel_tau(ckpt_path, full)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare h/u^p across two saved kernels")
    p.add_argument("--ckpt-narrow", type=Path, required=True, help="Checkpoint trained with d in (0.5, 1)")
    p.add_argument("--ckpt-wide", type=Path, required=True, help="Checkpoint trained with d in (0.5, 2)")
    p.add_argument("--out", type=Path, default=SCRIPT_DIR / "output" / "compare_h_div_u_sqrtu.png")
    p.add_argument("--u-pow", type=float, default=0.5, help="Fixed denominator exponent (default 0.5 → u^{1/2})")
    p.add_argument("--narrow-label", type=str, default=r"$d\in(0.5,1)$")
    p.add_argument("--wide-label", type=str, default=r"$d\in(0.5,2)$")
    args = p.parse_args()

    if not (args.u_pow > 0):
        p.error("--u-pow must be positive")

    narrow_net, tau_n = _kernel_tau_auto(args.ckpt_narrow.resolve())
    wide_net, tau_w = _kernel_tau_auto(args.ckpt_wide.resolve())

    specs = [
        (narrow_net, tau_n, args.narrow_label),
        (wide_net, tau_w, args.wide_label),
    ]
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_kernel_h_div_u_pow_compare(specs, args.out, u_pow=float(args.u_pow))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
