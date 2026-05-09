"""
Step 4–5: Neural network for kernel h(τ, s) and pricing operator kernel -> option price.

Architecture: input (first, second) times -> MLP -> output with causal mask **second > first**
(see docstrings). Pricing: h -> sigma^2(τ) = int_0^τ h^2 ds -> Gaussian CF -> Carr-Madan.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveLinear(nn.Module):
    """
    Linear map with **strictly positive** effective weights and biases:
    ``y = (softplus(W_raw) + eps) @ x + (softplus(b_raw) + eps)``.
    Raw parameters are unconstrained; forward applies softplus so W,b used in the affine map are > 0.
    """

    __constants__ = ["in_features", "out_features", "bias"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.weight) + self.eps
        b = None if self.bias is None else (F.softplus(self.bias) + self.eps)
        return F.linear(input, w, b)


def _linear_wb_mins_module(m: nn.Module) -> tuple[float, float] | None:
    """Return (min W, min b) for one linear-like module, or None if not applicable."""
    if isinstance(m, PositiveLinear):
        w_eff = F.softplus(m.weight) + m.eps
        w_min = float(w_eff.min().item())
        if m.bias is not None:
            b_eff = F.softplus(m.bias) + m.eps
            b_min = float(b_eff.min().item())
        else:
            b_min = float("inf")
        return w_min, b_min
    if isinstance(m, nn.Linear):
        w_min = float(m.weight.data.min().item())
        b_min = float(m.bias.data.min().item()) if m.bias is not None else float("inf")
        return w_min, b_min
    return None


def linear_weight_bias_mins(module: nn.Module) -> tuple[float | None, float | None]:
    """Minimum weight and minimum bias over all ``nn.Linear`` / ``PositiveLinear`` (``None`` if none)."""
    w_mins: list[float] = []
    b_mins: list[float] = []
    for m in module.modules():
        pair = _linear_wb_mins_module(m)
        if pair is not None:
            w_mins.append(pair[0])
            if pair[1] < float("inf"):
                b_mins.append(pair[1])
    if not w_mins:
        return None, None
    return min(w_mins), (min(b_mins) if b_mins else None)


def check_linear_weights_biases_nonnegative(
    module: nn.Module,
    *,
    strict: bool = False,
    prefix: str = "",
) -> bool:
    """
    If any linear layer has a negative effective weight or bias, print (or raise when ``strict``).
    ``PositiveLinear`` is checked on ``softplus(raw) + eps``; ordinary ``nn.Linear`` on stored tensors.
    Returns ``True`` iff every such layer has all effective weights and biases ``>= 0``.
    """
    ok = True
    for name, m in module.named_modules():
        pair = _linear_wb_mins_module(m)
        if pair is None:
            continue
        w_min, b_min = pair
        if w_min < 0.0 or b_min < 0.0:
            ok = False
            loc = f"{prefix}{name}" if prefix else name
            msg = f"{loc}: min(W)={w_min:.6g} min(b)={b_min:.6g}"
            if strict:
                raise RuntimeError(f"Linear W,b must be nonnegative: {msg}")
            print(f"[linear_Wb_check] {msg}")
    return ok


class KernelNet(nn.Module):
    """
    Two-time kernel network. Forward signature is ``forward(first, second)`` (parameters
    named ``tau``, ``s`` for historical reasons). **Mask: nonzero only if second > first.**

    - **Full model:** ``net(r, s)`` with ``s > r`` on ``∫_r^T`` (first = lower time ``r``).
    - **Gaussian / variance path:** ``net(s, τ)`` with ``τ > s`` on ``∫_0^τ`` — pass
      integration variable ``s`` as the **first** argument and maturity ``τ`` as the **second**
      so the mask matches the triangle ``0 ≤ s < τ``.

    If ``nonnegative=True`` (default), the last layer output is passed through ``softplus`` so
    ``h ≥ 0`` on the active mask (only affects the ``mlp`` path; use ``nonnegative=False`` to
    recover a signed kernel).
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        input_scale: float = 1.0,
        output_scale: float = 0.1,
        nonnegative: bool = True,
        positive_linear_wb: bool = False,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64, 64]
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.nonnegative = nonnegative
        dims = [2] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            lin: nn.Module
            if positive_linear_wb:
                lin = PositiveLinear(dims[i], dims[i + 1])
            else:
                lin = nn.Linear(dims[i], dims[i + 1])
            layers.append(lin)
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, tau: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        ``tau`` / ``s`` are the two time arguments (see class doc). Returns ``h``, masked
        to zero unless **s > tau** (i.e. second input > first).
        """
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        if s.dim() == 0:
            s = s.unsqueeze(0)
        if tau.dim() == 1:
            tau = tau.unsqueeze(1)
        if s.dim() == 1:
            s = s.unsqueeze(1)
        tau = tau.expand(-1, s.size(1)) if tau.size(1) == 1 and s.size(1) > 1 else tau
        s = s.expand(tau.size(0), -1) if s.size(0) == 1 and tau.size(0) > 1 else s
        x = torch.cat([tau * self.input_scale, s * self.input_scale], dim=-1)
        raw = self.net(x).squeeze(-1)
        out = F.softplus(raw) if self.nonnegative else raw
        mask = (s > tau).float().squeeze(-1) if s.dim() > 1 else (s > tau).float()
        return out * mask * self.output_scale


class StructuredKernelNetPaper7(nn.Module):
    """
    Wang–Xia–style admissible kernel satisfying asymptotic growth (their Eq. 7 sketch):

    With lag ``u = second - first`` (so ``u = s - r`` for ``net(r,s)`` on the full model, and
    ``u = τ - s`` when you call ``net(s, τ)`` for the Gaussian variance path),

    ``h ~ O(u^{d-1})`` as ``u ↘ 0``, and exponential damping as ``u → ∞``.

    **Parametrization (hard structural prior):**

    ``h = u^{d-1} · exp(-κ u) · g(r, s)``

    where ``g ≥ 0`` is a small MLP (learns residual shape), learnable ``d ∈ (d_\mathrm{low}, d_\mathrm{high})``
    unless ``d_fixed`` is set (constant ``d``), and ``κ > 0`` learnable.

    This is **not** identical to every detail of (7) (which uses two-sided asymptotics on
    ``h(t+u,t)``), but it embeds the same **power-law near the diagonal** and **mean-reversion
    damping at large lags** in code.

    Drop-in replacement for ``KernelNet`` wherever ``forward(tau, s)`` is used.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        input_scale: float = 1.0,
        g_scale: float = 1.0,
        u_floor: float = 1e-8,
        d_low: float = 0.5,
        d_high: float = 2.0,
        d_fixed: float | None = None,
        positive_linear_wb: bool = False,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64, 64]
        self.input_scale = input_scale
        self.g_scale = g_scale
        self.d_fixed: float | None = None if d_fixed is None else float(d_fixed)
        d_lo, d_hi = float(d_low), float(d_high)
        if self.d_fixed is None:
            if not (0.0 < d_lo < d_hi):
                raise ValueError("StructuredKernelNetPaper7 requires 0 < d_low < d_high when d is learnable")
        self.d_low = d_lo
        self.d_high = d_hi
        # Lower bound only for ``pow(u, d-1)`` / ``exp(-κ u)`` when ``u_raw > 0``; support uses ``u_raw > 0``.
        self.u_floor = float(u_floor)
        if self.d_fixed is not None:
            if not (float(self.d_fixed) > 0.0):
                raise ValueError("paper7 d_fixed must be > 0")
            self.register_buffer("d_const", torch.tensor([float(self.d_fixed)], dtype=torch.float32))
            self.raw_d = None  # type: ignore[assignment]
        else:
            # d in (d_low, d_high): d = d_low + (d_high - d_low) * sigmoid(raw_d)
            self.raw_d = nn.Parameter(torch.tensor(0.0))
        # κ > 0
        self.raw_kappa = nn.Parameter(torch.tensor(0.0))
        dims = [2] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            if positive_linear_wb:
                layers.append(PositiveLinear(dims[i], dims[i + 1]))
            else:
                layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.g_net = nn.Sequential(*layers)

    def forward(self, tau: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        if s.dim() == 0:
            s = s.unsqueeze(0)
        if tau.dim() == 1:
            tau = tau.unsqueeze(1)
        if s.dim() == 1:
            s = s.unsqueeze(1)
        tau = tau.expand(-1, s.size(1)) if tau.size(1) == 1 and s.size(1) > 1 else tau
        s = s.expand(tau.size(0), -1) if s.size(0) == 1 and tau.size(0) > 1 else s

        # ``u_raw = second - first`` in call-site order ``net(s_inner, τ_outer)`` (see class docstring).
        u_raw = s - tau
        mask = (u_raw > 0).float()
        if mask.dim() > 1:
            mask = mask.squeeze(-1)

        u_eps = max(self.u_floor, 1e-12)
        u_num = torch.clamp(u_raw, min=u_eps)
        if self.raw_d is not None:
            span = self.d_high - self.d_low
            d = self.d_low + span * torch.sigmoid(self.raw_d)
        else:
            d = self.d_const.to(device=u_num.device, dtype=u_num.dtype).view(())
        kappa = F.softplus(self.raw_kappa) + 1e-4
        # Base factor: u^{d-1} e^{-κ u} on interior; invalid ``u_raw <= 0`` zeroed by ``mask``.
        base = torch.pow(u_num, d - 1.0) * torch.exp(-kappa * u_num)
        base = base.squeeze(-1)

        x = torch.cat([tau * self.input_scale, s * self.input_scale], dim=-1)
        g = F.softplus(self.g_net(x).squeeze(-1)) + 1e-3
        out = base * g * self.g_scale
        return out * mask


class ConstantKernelNet(nn.Module):
    """
    Degenerate kernel: ``h = h_value`` on the causal set **second > first**, else ``0``.
    No learnable parameters (only ``FullOptionModel`` scalars train in ``train_full.py``).
    """

    def __init__(self, h_value: float = 0.6):
        super().__init__()
        hv = float(h_value)
        if hv <= 0.0:
            raise ValueError("ConstantKernelNet requires h_value > 0")
        self.h_value = hv

    def forward(self, tau: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        if s.dim() == 0:
            s = s.unsqueeze(0)
        if tau.dim() == 1:
            tau = tau.unsqueeze(1)
        if s.dim() == 1:
            s = s.unsqueeze(1)
        tau = tau.expand(-1, s.size(1)) if tau.size(1) == 1 and s.size(1) > 1 else tau
        s = s.expand(tau.size(0), -1) if s.size(0) == 1 and tau.size(0) > 1 else s
        mask = (s > tau).float().squeeze(-1) if s.dim() > 1 else (s > tau).float()
        h = tau.new_tensor(self.h_value)
        return h * mask


class StrikeVolScale(nn.Module):
    """
    Positive multiplicative factor on total variance from (normalized τ, log-moneyness).
    Lets σ² vary with strike / moneyness while the kernel still sets a τ-only baseline.
    """

    def __init__(self, hidden: int = 32, positive_linear_wb: bool = False):
        super().__init__()
        if positive_linear_wb:
            self.net = nn.Sequential(
                PositiveLinear(2, hidden),
                nn.ReLU(),
                PositiveLinear(hidden, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, tau_norm: torch.Tensor, log_k: torch.Tensor) -> torch.Tensor:
        x = torch.stack([tau_norm, log_k], dim=-1)
        return F.softplus(self.net(x).squeeze(-1)) + 0.25


def build_kernel_net(
    hidden_dims: list[int] | None = None,
    kernel_type: str = "mlp",
    kernel_nonnegative: bool = True,
    output_scale: float = 0.5,
    device: torch.device | None = None,
    paper7_d_low: float = 0.5,
    paper7_d_high: float = 2.0,
    paper7_d_fixed: float | None = None,
    positive_linear_wb: bool = False,
    constant_h: float = 0.6,
) -> nn.Module:
    """
    Factory: ``mlp`` = ``KernelNet``; ``paper7`` = structured nonnegative kernel; ``constant`` = fixed ``h``.
    """
    hidden_dims = hidden_dims or [64, 64, 64]
    dev = device or torch.device("cpu")
    if kernel_type == "constant":
        return ConstantKernelNet(h_value=float(constant_h)).to(dev)
    if kernel_type == "paper7":
        return StructuredKernelNetPaper7(
            hidden_dims=hidden_dims,
            input_scale=1.0,
            g_scale=max(0.2, output_scale),
            d_low=paper7_d_low,
            d_high=paper7_d_high,
            d_fixed=paper7_d_fixed,
            positive_linear_wb=positive_linear_wb,
        ).to(dev)
    return KernelNet(
        hidden_dims=hidden_dims,
        input_scale=1.0,
        output_scale=output_scale,
        nonnegative=kernel_nonnegative,
        positive_linear_wb=positive_linear_wb,
    ).to(dev)


class SimplifiedPricer(nn.Module):
    """
    Kernel → σ² (with physical-time scaling) → optional strike/moneyness scale → Carr–Madan.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        kernel_type: str = "mlp",
        kernel_nonnegative: bool = True,
        output_scale: float = 0.5,
        use_strike_scale: bool = True,
        device: torch.device | None = None,
        paper7_d_low: float = 0.5,
        paper7_d_high: float = 2.0,
        paper7_d_fixed: float | None = None,
        positive_linear_wb: bool = False,
        constant_h: float = 0.6,
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        self.kernel_net = build_kernel_net(
            hidden_dims=hidden_dims,
            kernel_type=kernel_type,
            kernel_nonnegative=kernel_nonnegative,
            output_scale=output_scale,
            device=dev,
            paper7_d_low=paper7_d_low,
            paper7_d_high=paper7_d_high,
            paper7_d_fixed=paper7_d_fixed,
            positive_linear_wb=positive_linear_wb,
            constant_h=constant_h,
        )
        self.strike_scale = (
            StrikeVolScale(positive_linear_wb=positive_linear_wb).to(dev)
            if use_strike_scale
            else None
        )

    def forward(
        self,
        tau_norm: torch.Tensor,
        K: torch.Tensor,
        S0: float | torch.Tensor,
        r: float,
        tau_scale: float,
        n_grid: int = 64,
        n_u: int = 256,
    ) -> torch.Tensor:
        return model_call_prices(
            self.kernel_net,
            tau_norm,
            K,
            r=r,
            S0=S0,
            n_grid=n_grid,
            n_u=n_u,
            tau_scale=tau_scale,
            strike_vol_scale=self.strike_scale,
        )


# -----------------------------------------------------------------------------
# Kernel -> variance sigma^2(τ)
# -----------------------------------------------------------------------------


def _module_device(net: nn.Module, *tensor_fallbacks: torch.Tensor | None) -> torch.device:
    """Device for ``net``; supports parameter-free modules (e.g. ``ConstantKernelNet``)."""
    for p in net.parameters():
        return p.device
    for b in net.buffers():
        return b.device
    for t in tensor_fallbacks:
        if t is not None and isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def variance_from_kernel(
    net: nn.Module,
    tau: torch.Tensor,
    n_grid: int = 64,
) -> torch.Tensor:
    """
    sigma^2(τ) = int_0^τ h(s, τ)^2 ds (trapezoidal), with ``net(s, τ)`` and ``τ > s`` on the path.

    We call ``net(s_flat, t_flat)`` so the existing mask ``second > first`` coincides with ``τ > s``.
    tau: (batch,) in same units as used in net (e.g. normalized [0,1] or years).
    """
    device = _module_device(net, tau)
    if not isinstance(tau, torch.Tensor):
        tau = torch.tensor(tau, device=device, dtype=torch.float32)
    if tau.dim() == 0:
        tau = tau.unsqueeze(0)
    if tau.dim() == 2 and tau.shape[1] == 1:
        tau = tau.squeeze(-1)
    batch = tau.size(0)
    u = torch.linspace(0.0, 1.0, n_grid + 1, device=device, dtype=torch.float32)
    s = tau.unsqueeze(1) * u.unsqueeze(0)
    t = tau.unsqueeze(1).expand(-1, n_grid + 1)
    t_flat = t.reshape(-1)
    s_flat = s.reshape(-1)
    h_flat = net(s_flat, t_flat)
    h = h_flat.reshape(batch, n_grid + 1)
    h2 = h ** 2
    du = 1.0 / n_grid
    integ = (h2[:, :-1] + h2[:, 1:]).sum(dim=1) * (du / 2.0)
    sigma_sq = integ * tau
    return sigma_sq


# -----------------------------------------------------------------------------
# Carr-Madan: sigma^2, K, τ -> call price C(K, τ)
# -----------------------------------------------------------------------------


def _s0_as_tensor(S0: float | torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar ``S0`` to ``K``'s shape, or validate / cast tensor ``S0``."""
    if isinstance(S0, torch.Tensor):
        s = S0.to(device=K.device, dtype=K.dtype)
        if s.shape != K.shape:
            raise ValueError(f"S0 tensor shape {tuple(s.shape)} must match K {tuple(K.shape)}")
        return s
    return torch.full_like(K, float(S0), dtype=K.dtype, device=K.device)


def call_price_carr_madan(
    sigma_sq: torch.Tensor,
    K: torch.Tensor,
    tau_phys: torch.Tensor,
    r: float = 0.0,
    S0: float | torch.Tensor = 1.0,
    alpha: float = 1.25,
    u_max: float = 100.0,
    n_u: int = 256,
    enforce_nonnegative: bool = True,
    sigma_sq_clip: float = 50.0,
) -> torch.Tensor:
    """
    European call via Carr-Madan using Gaussian log-price CF under risk-neutral drift.
    ``tau_phys`` = maturity in **years**. ``sigma_sq`` = total variance over ``tau_phys``.
    ``S0`` may be a scalar or a tensor matching ``K`` (per-row spot).
    """
    device = sigma_sq.device
    S0_t = _s0_as_tensor(S0, K)
    # Work in log-strike space directly to keep phase/CF normalization consistent.
    log_K = torch.log(torch.clamp(K, min=1e-12)).to(device)
    log_S0 = torch.log(torch.clamp(S0_t, min=1e-12)).to(device)
    sigma_sq = torch.clamp(sigma_sq, min=1e-12, max=float(sigma_sq_clip))
    tau_phys = torch.clamp(tau_phys, min=1e-12)
    u = torch.linspace(1e-6, u_max, n_u, device=device, dtype=torch.float32)
    a = alpha + 1.0
    v = u.unsqueeze(0).expand(sigma_sq.size(0), -1)
    z = v - 1j * a
    # log S_T = log S0 + (r*T - 0.5*sigma_sq) + sqrt(sigma_sq) * Z
    mu = r * tau_phys - 0.5 * sigma_sq
    phi_c = torch.exp(1j * z * (log_S0.unsqueeze(1) + mu.unsqueeze(1)) - 0.5 * sigma_sq.unsqueeze(1) * (z ** 2))
    denom = alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
    denom = torch.where(torch.abs(denom) < 1e-12, denom + (1e-12 + 0j), denom)
    psi = phi_c / denom.unsqueeze(0)
    phase = torch.exp(-1j * u.unsqueeze(0) * log_K.unsqueeze(1))
    integrand = (phase * psi).real
    du = u_max / (n_u - 1)
    integral = (integrand[:, :-1] + integrand[:, 1:]).sum(dim=1) * (du / 2.0)
    C = torch.exp(-r * tau_phys) * torch.exp(-alpha * log_K) / np.pi * integral
    return C.clamp(min=0.0) if enforce_nonnegative else C


# -----------------------------------------------------------------------------
# End-to-end: net + (τ, K) -> model call prices
# -----------------------------------------------------------------------------


def model_call_prices(
    net: nn.Module,
    tau: torch.Tensor,
    K: torch.Tensor,
    r: float = 0.0,
    S0: float | torch.Tensor = 1.0,
    n_grid: int = 64,
    n_u: int = 256,
    tau_scale: float = 1.0,
    strike_vol_scale: nn.Module | None = None,
    enforce_nonnegative: bool = True,
) -> torch.Tensor:
    """
    Forward: kernel net -> σ² (physical scaling) -> optional strike/moneyness factor -> Carr–Madan.

    ``tau`` is **normalized** ``τ / τ_max`` in ``[0, 1]`` (same as training). ``tau_scale`` should be
    ``τ_max`` in years so discounting and variance use physical time.
    ``S0`` may be a scalar or a tensor matching ``K`` (per-row spot).
    """
    sigma_sq = variance_from_kernel(net, tau, n_grid=n_grid)
    # ∫ h² ds_norm over [0, τ_norm] relates to physical ∫ by ds_phys = τ_max ds_norm
    sigma_sq = sigma_sq * tau_scale
    S0_t = _s0_as_tensor(S0, K)
    if strike_vol_scale is not None:
        log_k = torch.log(K / S0_t)
        m = strike_vol_scale(tau, log_k)
        sigma_sq = sigma_sq * m
    tau_phys = tau * tau_scale
    return call_price_carr_madan(
        sigma_sq,
        K,
        tau_phys,
        r=r,
        S0=S0_t,
        n_u=n_u,
        enforce_nonnegative=enforce_nonnegative,
    )
