"""
Functional sensitivity of the kernel map h w.r.t. one training epoch of optimizer steps.

- **Grid finite difference:** ||h(θ_end) - h(θ_start)|| on a fixed (r,s) or (s,τ) triangle.
- **First-order scalar:** for h̄ = mean(h on the same grid), compare
  (h̄(θ_end) - h̄(θ_start)) to ⟨∇_θ h̄(θ_start), θ_end - θ_start⟩ (inner product over kernel parameters).

Call convention matches training code:
- **Full model:** ``kernel_net(r, s)`` with ``s > r`` (normalized times in ``[0, 1]``).
- **Gaussian / SimplifiedPricer:** ``kernel_net(s, τ)`` with ``τ > s`` (same units as variance integral).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

KernelOrder = Literal["rs", "st"]


def triangle_grid_rs(
    device: torch.device,
    dtype: torch.dtype,
    n_r: int,
    n_s: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower time r, upper time s with ``0 <= r < s <= 1`` (full-model convention)."""
    n_r = max(2, int(n_r))
    n_s = max(2, int(n_s))
    r_vals = torch.linspace(0.0, 1.0 - 1e-6, n_r, device=device, dtype=dtype)
    s_vals = torch.linspace(1e-6, 1.0, n_s, device=device, dtype=dtype)
    R, S = torch.meshgrid(r_vals, s_vals, indexing="ij")
    mask = S > R + 1e-9
    r_flat = R[mask].reshape(-1)
    s_flat = S[mask].reshape(-1)
    return r_flat, s_flat


def triangle_grid_st(
    device: torch.device,
    dtype: torch.dtype,
    n_r: int,
    n_s: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner time s, maturity τ with ``0 < s < τ <= 1`` (Gaussian variance path convention)."""
    n_r = max(2, int(n_r))
    n_s = max(2, int(n_s))
    s_vals = torch.linspace(1e-6, 1.0 - 1e-6, n_r, device=device, dtype=dtype)
    t_vals = torch.linspace(1e-5, 1.0, n_s, device=device, dtype=dtype)
    S, T = torch.meshgrid(s_vals, t_vals, indexing="ij")
    mask = T > S + 1e-9
    s_flat = S[mask].reshape(-1)
    t_flat = T[mask].reshape(-1)
    return s_flat, t_flat


def eval_h_on_grid(
    kernel_net: nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
    order: KernelOrder,
) -> torch.Tensor:
    """Evaluate h at paired columns ``(a, b)`` each shape ``(N,)`` → returns ``(N,)``."""
    af = a.reshape(-1, 1)
    bf = b.reshape(-1, 1)
    if order == "rs":
        return kernel_net(af, bf).reshape(-1)
    return kernel_net(af, bf).reshape(-1)


@dataclass
class KernelHSensitivityPreflight:
    """Snapshot at the start of an epoch (before minibatch updates)."""

    order: KernelOrder
    a: torch.Tensor
    b: torch.Tensor
    h_before: torch.Tensor
    h_bar_before: float
    p_start: dict[str, torch.Tensor]
    param_rows: list[tuple[str, torch.nn.Parameter]]
    grads: tuple[torch.Tensor | None, ...]


def kernel_h_epoch_sensitivity_preflight(
    kernel_net: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_r: int,
    n_s: int,
    order: KernelOrder,
) -> KernelHSensitivityPreflight:
    kernel_net.train()
    if order == "rs":
        a, b = triangle_grid_rs(device, dtype, n_r, n_s)
    else:
        a, b = triangle_grid_st(device, dtype, n_r, n_s)
    param_rows = list(kernel_net.named_parameters())
    param_list = [p for _, p in param_rows]
    h_vec = eval_h_on_grid(kernel_net, a, b, order)
    h_before = h_vec.detach().clone()
    h_bar = h_vec.mean()
    if not param_list:
        grads = tuple()
    elif h_bar.requires_grad:
        grads = torch.autograd.grad(
            h_bar,
            param_list,
            retain_graph=False,
            allow_unused=True,
        )
    else:
        # e.g. parameter-free ``ConstantKernelNet``: h̄ is constant w.r.t. θ
        grads = tuple(None for _ in param_list)
    p_start = {n: p.detach().clone() for n, p in param_rows}
    return KernelHSensitivityPreflight(
        order=order,
        a=a,
        b=b,
        h_before=h_before,
        h_bar_before=float(h_bar.detach().item()),
        p_start=p_start,
        param_rows=param_rows,
        grads=grads,
    )


def kernel_h_epoch_sensitivity_postflight(
    kernel_net: nn.Module,
    pre: KernelHSensitivityPreflight,
) -> dict[str, float]:
    """Call after ``optimizer.step()`` loops for the epoch; uses current ``kernel_net`` weights."""
    kernel_net.train()
    with torch.no_grad():
        h_after = eval_h_on_grid(kernel_net, pre.a, pre.b, pre.order).detach()
    dh = h_after - pre.h_before
    l2 = float(torch.norm(dh).item())
    linf = float(dh.abs().max().item())
    mae = float(dh.abs().mean().item())
    h_bar_after = float(h_after.mean().item())
    actual_delta_hbar = h_bar_after - pre.h_bar_before

    pred_delta_hbar = 0.0
    for g, (name, p) in zip(pre.grads, pre.param_rows):
        if g is None:
            continue
        d = p.data - pre.p_start[name]
        pred_delta_hbar += float((g * d).sum().item())

    denom = max(abs(actual_delta_hbar), abs(pred_delta_hbar), 1e-12)
    rel_err_first_order = abs(actual_delta_hbar - pred_delta_hbar) / denom

    return {
        "dh_grid_l2": l2,
        "dh_grid_max_abs": linf,
        "dh_grid_mean_abs": mae,
        "hbar_before": pre.h_bar_before,
        "hbar_after": h_bar_after,
        "dhbar_actual": actual_delta_hbar,
        "dhbar_pred_first_order": pred_delta_hbar,
        "dhbar_first_order_rel_err": rel_err_first_order,
        "n_points": float(h_after.numel()),
    }


def format_h_sensitivity_log(m: dict[str, float]) -> str:
    return (
        f"[h_sensitivity] grid: n={int(m['n_points'])}  "
        f"||dh||_2={m['dh_grid_l2']:.6g}  max|dh|={m['dh_grid_max_abs']:.6g}  "
        f"mean|dh|={m['dh_grid_mean_abs']:.6g}  |  "
        f"h_mean: {m['hbar_before']:.6g} -> {m['hbar_after']:.6g}  "
        f"dh_mean_actual={m['dhbar_actual']:.6g}  dh_mean_pred_1st={m['dhbar_pred_first_order']:.6g}  "
        f"rel_err_1st_order={m['dhbar_first_order_rel_err']:.6g}"
    )


H_SENSITIVITY_CSV_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "dh_grid_l2",
    "dh_grid_max_abs",
    "dh_grid_mean_abs",
    "h_mean_after",
    "dh_mean_actual",
    "dh_mean_pred_1st",
    "rel_err_1st_order",
    "n_grid_points",
]


def append_h_sensitivity_csv(
    csv_path: Path | str,
    *,
    epoch: int,
    metrics: dict[str, float],
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> None:
    """Append one row (create file + header if missing)."""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_file = not p.exists() or p.stat().st_size == 0
    row = {
        "epoch": str(int(epoch)),
        "train_loss": "" if train_loss is None else f"{float(train_loss):.10g}",
        "val_loss": "" if val_loss is None else f"{float(val_loss):.10g}",
        "dh_grid_l2": f"{metrics['dh_grid_l2']:.10g}",
        "dh_grid_max_abs": f"{metrics['dh_grid_max_abs']:.10g}",
        "dh_grid_mean_abs": f"{metrics['dh_grid_mean_abs']:.10g}",
        "h_mean_after": f"{metrics['hbar_after']:.10g}",
        "dh_mean_actual": f"{metrics['dhbar_actual']:.10g}",
        "dh_mean_pred_1st": f"{metrics['dhbar_pred_first_order']:.10g}",
        "rel_err_1st_order": f"{metrics['dhbar_first_order_rel_err']:.10g}",
        "n_grid_points": str(int(metrics["n_points"])),
    }
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=H_SENSITIVITY_CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
