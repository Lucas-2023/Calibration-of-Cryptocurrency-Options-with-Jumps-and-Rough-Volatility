"""
Full math model: stochastic variance with mean reversion + kernel-weighted Lévy (tempered stable).

Kernel convention (explicit): h(s, r) for integration variable s and "lower" time r, with s > r.
We evaluate as net(r, s) where KernelNet has mask (s > r) — first arg = r, second = s.
"""
import numpy as np
import torch
import torch.nn as nn
from math import gamma as math_gamma

from model import KernelNet


def _kernel_net_device(net: nn.Module, *tensor_fallbacks: torch.Tensor | None) -> torch.device:
    """Device for ``net``; supports parameter-free kernels (e.g. ``ConstantKernelNet``)."""
    for p in net.parameters():
        return p.device
    for b in net.buffers():
        return b.device
    for t in tensor_fallbacks:
        if t is not None and isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Eq. 4: Tempered stable — log phi_{X_1}(z) = a Gamma(-c)[(b - iz)^c - b^c]
# Gamma(-c) is evaluated with Python math (no grad through c); a, b, c enter powers with grad.
# -----------------------------------------------------------------------------


def _complex_power_real_exp(
    re_val: torch.Tensor,
    im_val: torch.Tensor,
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(re_val + 1j*im_val)^c; c may be 0-dim tensor (broadcast)."""
    mag_sq = re_val ** 2 + im_val ** 2
    mag = torch.clamp(mag_sq, min=1e-12) ** 0.5
    angle = torch.atan2(im_val, re_val)
    out_mag = mag ** c
    out_angle = c * angle
    return out_mag * torch.cos(out_angle), out_mag * torch.sin(out_angle)


def log_phi_X1_tempered_stable(
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    gamma_neg_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """log phi_{X_1}(z); a, b, c, gamma_neg_c are tensors on same device as z_re."""
    base_re = b + z_im
    base_im = -z_re
    pow_re, pow_im = _complex_power_real_exp(base_re, base_im, c)
    bc = torch.pow(b, c)
    diff_re = pow_re - bc
    diff_im = pow_im
    coef = a * gamma_neg_c
    return coef * diff_re, coef * diff_im


# -----------------------------------------------------------------------------
# Eq. 2: A(T) = V_0/k*(1 - e^{-kT}) + V_bar*(T - (1/k)*(1 - e^{-kT}))
# -----------------------------------------------------------------------------


def mean_reversion_A(
    tau_phys: torch.Tensor,
    V0: torch.Tensor,
    V_bar: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """tau_phys, V0, V_bar, k: broadcastable tensors (e.g. 0-dim params)."""
    exp_kt = torch.exp(-k * tau_phys)
    term1 = V0 / k * (1.0 - exp_kt)
    term2 = V_bar * (tau_phys - (1.0 / k) * (1.0 - exp_kt))
    return term1 + term2


# -----------------------------------------------------------------------------
# Eq. 6 (revised): Phi_T(z) with z = u_re + i*u_im on the Carr-Madan contour and
#   xi(r) = int_r^T h(s,r) ds:
#
#   Phi_T(z) = exp(i*z*log S_0) * exp(-(i*z+z^2)/2 * A(T)) * exp(int log phi_{X_1}(z*(-i)*xi(r)) dr)
#
# Lévy argument is LINEAR in z (z * (-i) * xi); do not use ((-u+i*u**2)/2)*integral h inside phi_{X_1}.
# A(T) = mean_reversion_A(T). Older drafts omitted the (iz+z^2)/2 factor — see PROJECT_PSEUDOCODE §4.4.
# -----------------------------------------------------------------------------


def characteristic_function_full(
    net: KernelNet,
    u_re: torch.Tensor,
    u_im: torch.Tensor,
    tau: float,
    V0: torch.Tensor,
    V_bar: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_r: int = 32,
    n_s: int = 32,
    tau_scale: float = 1.0,
    S0: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full CF at u = u_re + i u_im.

    Deterministic (mean-reversion) piece matches Eq. 6:
    exp(i u log S_0 - (i u + u^2)/2 * A(T)) with A(T) from mean_reversion_A.
    Lévy piece: exp(int_0^T log phi_{X_1}(u*(-i)*xi(r)) dr) as before.

    tau in [0,1] normalized; r,s grids in [0, tau]. A uses tau_phys = tau * tau_scale.
    """
    if device is None:
        device = _kernel_net_device(net, u_re, u_im, V0)
    c_f = float(c.detach().cpu().item())
    gamma_neg_c = torch.as_tensor(
        math_gamma(-c_f), device=device, dtype=torch.float32
    )
    tau_clip = max(min(float(tau), 1.0), 1e-9)
    r_grid = torch.linspace(0.0, tau_clip, n_r + 1, device=device, dtype=torch.float32)
    dr = tau_clip / n_r
    xi_list = []
    for j in range(r_grid.size(0) - 1):
        r = r_grid[j].item()
        s_vals = torch.linspace(r, tau_clip, n_s + 1, device=device, dtype=torch.float32)
        r_batch = torch.full((s_vals.size(0),), r, device=device, dtype=torch.float32)
        h_sr = net(r_batch, s_vals)
        ds = (tau_clip - r) / max(n_s, 1)
        integ = (h_sr[:-1] + h_sr[1:]).sum() * 0.5 * ds if n_s > 0 else h_sr[0] * ds
        xi_list.append(integ)
    xi = torch.cat([torch.stack(xi_list), torch.zeros(1, device=device)])
    if u_re.dim() == 0:
        u_re = u_re.unsqueeze(0)
        u_im = u_im.unsqueeze(0)
    z_re = u_im.unsqueeze(1) * xi.unsqueeze(0)
    z_im = -u_re.unsqueeze(1) * xi.unsqueeze(0)
    log_phi_re, log_phi_im = log_phi_X1_tempered_stable(z_re, z_im, a, b, c, gamma_neg_c)
    outer_re = (log_phi_re[:, :-1] + log_phi_re[:, 1:]).sum(dim=1) * 0.5 * dr
    outer_im = (log_phi_im[:, :-1] + log_phi_im[:, 1:]).sum(dim=1) * 0.5 * dr
    exp_outer_re = torch.exp(outer_re.clamp(max=10.0)) * torch.cos(outer_im)
    exp_outer_im = torch.exp(outer_re.clamp(max=10.0)) * torch.sin(outer_im)
    tau_phys = torch.tensor(tau * tau_scale, device=device, dtype=torch.float32)
    A_T = mean_reversion_A(tau_phys, V0, V_bar, k).squeeze()
    log_s0 = float(np.log(max(S0, 1e-12)))

    # z = u_re + i u_im
    ur = u_re.squeeze()
    ui = u_im.squeeze()
    # exp(i * z * log S_0):  i*z*logS0 = (i*ur - ui)*logS0  ->  Re = -ui*logS0,  Im = ur*logS0
    t1_mag = torch.exp(-ui * log_s0)
    t1_re = t1_mag * torch.cos(ur * log_s0)
    t1_im = t1_mag * torch.sin(ur * log_s0)

    # w = (i*z + z^2) / 2  (complex);  exp(-w * A_T)
    # i*z = -ui + i*ur
    iz_re = -ui
    iz_im = ur
    z2_re = ur * ur - ui * ui
    z2_im = 2.0 * ur * ui
    sum_re = iz_re + z2_re
    sum_im = iz_im + z2_im
    w_re = 0.5 * sum_re
    w_im = 0.5 * sum_im
    # exp(-A_T * w)
    t2_mag = torch.exp(-A_T * w_re)
    t2_re = t2_mag * torch.cos(-A_T * w_im)
    t2_im = t2_mag * torch.sin(-A_T * w_im)

    det_re = t1_re * t2_re - t1_im * t2_im
    det_im = t1_re * t2_im + t1_im * t2_re

    phi_re = det_re * exp_outer_re - det_im * exp_outer_im
    phi_im = det_re * exp_outer_im + det_im * exp_outer_re
    return phi_re, phi_im


# -----------------------------------------------------------------------------
# Carr-Madan with full phi_T
# -----------------------------------------------------------------------------


def call_price_carr_madan_full(
    net: KernelNet,
    tau: torch.Tensor,
    K: torch.Tensor,
    V0: torch.Tensor,
    V_bar: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    r: float = 0.0,
    S0: float = 1.0,
    alpha: float = 1.25,
    u_max: float = 100.0,
    n_u: int = 128,
    n_r: int = 24,
    n_s: int = 24,
    tau_scale: float = 1.0,
    device: torch.device | None = None,
    enforce_nonnegative: bool = True,
    debug_stats: bool = False,
) -> torch.Tensor:
    """European call; V0, V_bar, k, a, b, c are 0-dim tensors (keeps autograd)."""
    if device is None:
        device = _kernel_net_device(net, tau, K, V0)
    log_K = torch.log(torch.clamp(K, min=1e-12)).to(device)
    u_grid = torch.linspace(1e-6, u_max, n_u, device=device, dtype=torch.float32)
    u_re = u_grid
    u_im = torch.full_like(u_grid, -(alpha + 1.0))
    batch = tau.size(0)
    prices = []
    for i in range(batch):
        tau_i = tau[i].item()
        phi_re, phi_im = characteristic_function_full(
            net, u_re, u_im, tau_i, V0, V_bar, k, a, b, c,
            n_r=n_r, n_s=n_s, tau_scale=tau_scale, S0=S0, device=device,
        )
        discount = torch.exp(torch.tensor(-r * tau_i * tau_scale, device=device, dtype=torch.float32))
        denom_re = alpha ** 2 + alpha - u_grid ** 2
        denom_im = (2 * alpha + 1) * u_grid
        denom_sq = torch.clamp(denom_re ** 2 + denom_im ** 2, min=1e-12)
        rho_re = (discount * (phi_re * denom_re + phi_im * denom_im)) / denom_sq
        rho_im = (discount * (phi_im * denom_re - phi_re * denom_im)) / denom_sq
        phase_re = torch.cos(u_grid * log_K[i])
        phase_im = -torch.sin(u_grid * log_K[i])
        integrand = phase_re * rho_re - phase_im * rho_im
        du = u_max / (n_u - 1)
        integral = (integrand[:-1] + integrand[1:]).sum() * 0.5 * du
        C = torch.exp(-alpha * log_K[i]) / np.pi * integral
        prices.append(C)
    out = torch.stack(prices)
    if debug_stats:
        neg_ratio = float((out < 0).float().mean().item()) if out.numel() else 0.0
        print(
            "Carr-Madan raw C stats:",
            f"min={float(out.min().item()):.6g}",
            f"mean={float(out.mean().item()):.6g}",
            f"max={float(out.max().item()):.6g}",
            f"neg_ratio={neg_ratio:.4f}",
        )
    return out.clamp(min=0.0) if enforce_nonnegative else out


# -----------------------------------------------------------------------------
# Trainable full model — forward uses tensors (no .item() on learnable params)
# -----------------------------------------------------------------------------


class FullOptionModel(nn.Module):
    """
    Kernel h(s,r) via KernelNet(r,s) + mean reversion (V0, V_bar, k) + tempered stable (a, b, c).
    """

    def __init__(
        self,
        kernel_net: KernelNet,
        V0_init: float = 0.04,
        V_bar_init: float = 0.04,
        k_init: float = 1.0,
        a_init: float = 0.1,
        b_init: float = 1.0,
        c_init: float = 0.5,
    ):
        super().__init__()
        self.kernel_net = kernel_net
        self.log_V0 = nn.Parameter(torch.tensor(np.log(V0_init)))
        self.log_V_bar = nn.Parameter(torch.tensor(np.log(V_bar_init)))
        self.log_k = nn.Parameter(torch.tensor(np.log(k_init)))
        self.log_a = nn.Parameter(torch.tensor(np.log(a_init)))
        self.log_b = nn.Parameter(torch.tensor(np.log(b_init)))
        self.log_c = nn.Parameter(torch.tensor(np.log(c_init)))

    def params_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Learnable scalars as 0-dim tensors (differentiable)."""
        V0 = torch.exp(self.log_V0)
        V_bar = torch.exp(self.log_V_bar)
        k = torch.exp(self.log_k)
        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        c = torch.sigmoid(self.log_c) * 0.99 + 0.01
        return V0, V_bar, k, a, b, c

    def get_params(self) -> tuple[float, float, float, float, float, float]:
        """For logging only — detaches from graph."""
        V0, V_bar, k, a, b, c = self.params_tensors()
        return V0.item(), V_bar.item(), k.item(), a.item(), b.item(), c.item()

    def forward(
        self,
        tau: torch.Tensor,
        K: torch.Tensor,
        r: float = 0.0,
        S0: float = 1.0,
        tau_scale: float = 1.0,
        n_r: int = 24,
        n_s: int = 24,
        n_u: int = 128,
        enforce_nonnegative: bool = True,
        debug_stats: bool = False,
    ) -> torch.Tensor:
        V0, V_bar, k, a, b, c = self.params_tensors()
        return call_price_carr_madan_full(
            self.kernel_net, tau, K, V0, V_bar, k, a, b, c,
            r=r, S0=S0, n_r=n_r, n_s=n_s, n_u=n_u, tau_scale=tau_scale,
            enforce_nonnegative=enforce_nonnegative, debug_stats=debug_stats,
        )
