"""
Plot raw h vs u and a grid of h/u^x for chosen exponents (small-u scaling diagnostics).

Example:
  py -3 plot_h_scaling_survey.py --checkpoint output/full_model_d05_2.pt \\
    --out-h output/survey_h_vs_u.png --out-grid output/survey_h_div_u_pow.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate import (
    load_checkpoint_full,
    load_checkpoint_simple,
    plot_kernel_h_div_u_pow_subplots,
    plot_kernel_h_t_plus_u,
)
from model import SimplifiedPricer

SCRIPT_DIR = Path(__file__).resolve().parent


def _kernel_tau_auto(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    full = ckpt.get("full_model_state") is not None
    if full:
        fm, _s0, tau_max, _a, _ = load_checkpoint_full(path)
        return fm.kernel_net, float(tau_max)
    net_obj, _s0, tau_max, _a, _ = load_checkpoint_simple(path)
    if isinstance(net_obj, SimplifiedPricer):
        return net_obj.kernel_net, float(tau_max)
    return net_obj, float(tau_max)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot h vs u and h/u^x grid")
    p.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR / "output" / "full_model_d05_2.pt")
    p.add_argument(
        "--out-h",
        type=Path,
        default=SCRIPT_DIR / "output" / "survey_h_vs_u.png",
        help="Output path for h(t+u,t) vs u",
    )
    p.add_argument(
        "--out-grid",
        type=Path,
        default=SCRIPT_DIR / "output" / "survey_h_div_u_pow.png",
        help="Output path for subplot grid h/u^x",
    )
    p.add_argument(
        "--exponents",
        type=float,
        nargs="+",
        default=[-0.5, -0.25, 0.25, 0.5, 1.0],
        help="Exponents x in h/u^x",
    )
    args = p.parse_args()

    ckpt = args.checkpoint.resolve()
    net, tau_max = _kernel_tau_auto(ckpt)

    args.out_h.parent.mkdir(parents=True, exist_ok=True)
    args.out_grid.parent.mkdir(parents=True, exist_ok=True)

    plot_kernel_h_t_plus_u(net, tau_max, args.out_h.resolve())
    plot_kernel_h_div_u_pow_subplots(net, tau_max, args.out_grid.resolve(), list(args.exponents))

    print(f"Wrote {args.out_h.resolve()}")
    print(f"Wrote {args.out_grid.resolve()}")


if __name__ == "__main__":
    main()
