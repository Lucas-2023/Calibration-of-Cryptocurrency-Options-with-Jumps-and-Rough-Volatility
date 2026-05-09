"""
Step 1–4: Load option dataset, compute mid prices, time to maturity τ, log-moneyness k.

Input: CSVs in data_dir:
  - UnderlyingOptionsIntervals_*.csv (original schema), and/or
  - aligned*.csv / final_cleaned*.csv (same aligned schema): columns date, strike, option_type, Bid, Ask, Mark;
    expiration from ``expiry_date`` (ISO) when present, else token ``expiry`` (e.g. 1MAY26).
  - If ``strict_10days=True``: **only** ``final_call_no_madan_strict_10days_*.csv`` (strict calendar DTE).
  - Else if any ``final_call_no_madan_tol_*.csv`` exist (direct **call** chains, aligned schema), **only** those are loaded
    (takes precedence over ``final_cleaned_no_tol_*.csv`` in the same folder).
  - Else if any ``final_cleaned_no_tol_*.csv`` exist, **only** those are loaded (avoids mixing with other ``final_cleaned*``).
    Prefer **USD** columns ``Bid_USD`` / ``Ask_USD`` / ``Mark_USD`` when present.
Output: Arrays (tau, k, K, price) and S0 for pricing.
"""
from pathlib import Path
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# Default: CSVs next to this package (e.g. final_cleaned_no_tol_*.csv in kernel_pipeline/).
DATA_DIR = SCRIPT_DIR


def _numeric_series_excel_dash(s: pd.Series) -> pd.Series:
    """Coerce to float; treat '-', '', whitespace as NaN."""
    if s is None:
        return pd.Series(np.nan, index=pd.RangeIndex(0))
    out = s.astype(str).str.strip().replace({"": np.nan, "-": np.nan, "—": np.nan})
    return pd.to_numeric(out, errors="coerce")


def _column_by_aliases(df_raw: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    """Case-insensitive match of ``df_raw`` column name to first hit in ``aliases``."""
    lower_map = {str(c).strip().lower(): c for c in df_raw.columns}
    for a in aliases:
        key = a.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _aligned_csv_to_standard(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map aligned export to interval-CSV column names.

    Prefer ``expiry_date`` (ISO calendar) when present; otherwise parse ``expiry`` tokens (``%d%b%y``).
    """
    need = {"date", "strike", "option_type"}
    missing = need - set(df_raw.columns)
    if missing:
        raise ValueError(f"aligned CSV missing columns {missing}")
    if "expiry_date" not in df_raw.columns and "expiry" not in df_raw.columns:
        raise ValueError("aligned CSV needs at least one of: expiry_date, expiry")

    df = pd.DataFrame()
    df["quote_datetime"] = pd.to_datetime(df_raw["date"])
    if "expiry_date" in df_raw.columns:
        exp_iso = pd.to_datetime(df_raw["expiry_date"], errors="coerce")
    else:
        exp_iso = pd.Series(pd.NaT, index=df_raw.index)
    if "expiry" in df_raw.columns:
        exp_tok = pd.to_datetime(
            df_raw["expiry"].astype(str).str.strip(),
            format="%d%b%y",
            errors="coerce",
        )
    else:
        exp_tok = pd.Series(pd.NaT, index=df_raw.index)
    df["expiration"] = exp_iso.where(exp_iso.notna(), exp_tok)
    df["strike"] = _numeric_series_excel_dash(df_raw["strike"])
    ot = df_raw["option_type"].astype(str).str.strip().str.upper().str[0]
    df["option_type"] = ot

    if "Bid_USD" in df_raw.columns and "Ask_USD" in df_raw.columns:
        df["bid"] = _numeric_series_excel_dash(df_raw["Bid_USD"])
        df["ask"] = _numeric_series_excel_dash(df_raw["Ask_USD"])
        mu = _column_by_aliases(df_raw, ("Mark_USD", "mark_usd"))
        if mu is not None:
            df["close"] = _numeric_series_excel_dash(df_raw[mu])
        else:
            df["close"] = np.nan
    else:
        bid_col = df_raw["Bid"] if "Bid" in df_raw.columns else df_raw.get("bid")
        ask_col = df_raw["Ask"] if "Ask" in df_raw.columns else df_raw.get("ask")
        df["bid"] = _numeric_series_excel_dash(bid_col)
        df["ask"] = _numeric_series_excel_dash(ask_col)
        if "Mark" in df_raw.columns:
            df["close"] = _numeric_series_excel_dash(df_raw["Mark"])
        elif "mark" in df_raw.columns:
            df["close"] = _numeric_series_excel_dash(df_raw["mark"])
        else:
            df["close"] = np.nan

    if "Volume" in df_raw.columns:
        vol = _numeric_series_excel_dash(df_raw["Volume"])
        df["open_interest"] = vol.fillna(0).astype(np.int64)
    else:
        df["open_interest"] = 0

    # Optional per-row **underlying spot** (not option bid/ask — those map to ``bid``/``ask`` above).
    # 1) Underlying/index bid & ask → ``get_underlying_mid`` uses (bid+ask)/2 when both > 0.
    ub_c = _column_by_aliases(df_raw, ("underlying_bid", "index_bid", "btc_index_bid"))
    ua_c = _column_by_aliases(df_raw, ("underlying_ask", "index_ask", "btc_index_ask"))
    got_underlying = False
    if ub_c is not None and ua_c is not None:
        ub = _numeric_series_excel_dash(df_raw[ub_c])
        ua = _numeric_series_excel_dash(df_raw[ua_c])
        if ((ub > 0) & (ua > 0)).any():
            df["underlying_bid"] = ub
            df["underlying_ask"] = ua
            got_underlying = True
    if not got_underlying:
        # Single spot column → duplicate so (ub+ua)/2 equals that mark
        spot_series = None
        for name in ("spot", "underlying_price", "underlying_mid", "index_price", "btc_spot", "S0"):
            c = _column_by_aliases(df_raw, (name,))
            if c is not None:
                spot_series = _numeric_series_excel_dash(df_raw[c])
                break
        if spot_series is not None and spot_series.notna().any():
            df["underlying_bid"] = spot_series
            df["underlying_ask"] = spot_series

    df = df.dropna(subset=["quote_datetime", "expiration", "strike"])
    return df


def resolve_synthetic_call_for_puts(
    option_type: str,
    synthetic_call_from_puts: bool,
    raw_put_targets: bool,
) -> tuple[bool, str | None]:
    """
    ``SimplifiedPricer`` / Carr-Madan path prices **calls**. If ``option_type`` is ``P``, training
    targets should be synthetic calls from put–call parity unless ``raw_put_targets`` is set.

    Returns:
        (effective_synthetic_call_from_puts, optional_user_message)
    """
    if option_type != "P":
        return synthetic_call_from_puts, None
    if raw_put_targets:
        if not synthetic_call_from_puts:
            return False, (
                "option_type=P with --raw-put-targets: using put mids as targets against a "
                "call pricer (generally inconsistent; use only if intentional)."
            )
        return True, None
    if not synthetic_call_from_puts:
        return True, (
            "option_type=P: enabled put->synthetic-call targets (same as --synthetic-call-from-puts): "
            "model prices calls, so targets use C = P + S0 - K*exp(-r*tau). "
            "Use --raw-put-targets to keep put mids as targets instead."
        )
    return synthetic_call_from_puts, None


def load_options_data(
    data_dir: Path | None = None,
    min_open_interest: int = 0,
    use_close_if_no_mid: bool = True,
    option_type: str = "C",
    strict_10days: bool = False,
) -> pd.DataFrame:
    """
    Load option CSVs from data_dir: UnderlyingOptionsIntervals_*.csv and/or aligned*.csv/final_cleaned*.csv,
    or **only** ``final_call_no_madan_strict_10days_*.csv`` when ``strict_10days=True``,
    or **only** ``final_call_no_madan_tol_*.csv`` when any such files exist, else **only**
    ``final_cleaned_no_tol_*.csv`` when those exist (avoids mixing with other ``final_cleaned*``).

    Mid = (bid + ask) / 2 when both > 0, else ``close``/Mark when use_close_if_no_mid.
    ``option_type``: keep rows where option_type matches ('C' or 'P', single letter).
    """
    data_dir = data_dir or DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if strict_10days:
        strict_10 = sorted(data_dir.glob("final_call_no_madan_strict_10days_*.csv"))
        if not strict_10:
            raise FileNotFoundError(
                f"No final_call_no_madan_strict_10days_*.csv in {data_dir} "
                "(use strict 10-day call exports in data_dir)."
            )
        csv_files = strict_10
        names = [fp.name for fp in csv_files[:8]]
        extra = " ..." if len(csv_files) > 8 else ""
        print(
            f"Strict 10-day mode: {len(csv_files)} CSV(s) — {', '.join(names)}{extra}"
        )
    call_madan = sorted(data_dir.glob("final_call_no_madan_tol_*.csv"))
    no_tol = sorted(data_dir.glob("final_cleaned_no_tol_*.csv"))
    if strict_10days:
        pass  # csv_files already set
    elif call_madan:
        csv_files = call_madan
    elif no_tol:
        csv_files = no_tol
    else:
        seen: set[str] = set()
        csv_files = []
        for pattern in ("UnderlyingOptionsIntervals_*.csv", "aligned*.csv", "final_cleaned*.csv"):
            for fp in sorted(data_dir.glob(pattern)):
                key = str(fp.resolve())
                if key not in seen:
                    seen.add(key)
                    csv_files.append(fp)
    if not csv_files:
        raise FileNotFoundError(
            f"No option CSVs in {data_dir} "
            "(expected final_call_no_madan_strict_10days_*.csv with strict_10days=True, else "
            "final_call_no_madan_tol_*.csv, final_cleaned_no_tol_*.csv, "
            "UnderlyingOptionsIntervals_*.csv, aligned*.csv, final_cleaned*.csv)"
        )

    dfs = []
    for fp in csv_files:
        raw = pd.read_csv(fp)
        if "quote_datetime" in raw.columns and "expiration" in raw.columns:
            df = raw
            df["quote_datetime"] = pd.to_datetime(df["quote_datetime"])
            df["expiration"] = pd.to_datetime(df["expiration"])
        elif "date" in raw.columns and ("expiry" in raw.columns or "expiry_date" in raw.columns):
            df = _aligned_csv_to_standard(raw)
        else:
            raise ValueError(
                f"Unrecognized option CSV schema in {fp}: "
                "need interval columns (quote_datetime, expiration) or aligned (date, expiry)."
            )
        if "close" not in df.columns:
            df["close"] = np.nan
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["option_type"] == option_type].copy()

    bid_ask_ok = (df["bid"] > 0) & (df["ask"] > 0)
    df["mid"] = np.where(bid_ask_ok, (df["bid"] + df["ask"]) / 2, np.nan)
    if use_close_if_no_mid:
        df.loc[df["mid"].isna(), "mid"] = df.loc[df["mid"].isna(), "close"]
    df = df.dropna(subset=["mid"])

    if min_open_interest > 0:
        df = df[df["open_interest"] >= min_open_interest]

    return df


def get_underlying_mid(df: pd.DataFrame) -> np.ndarray | None:
    """Per-row underlying mid (underlying_bid + underlying_ask)/2 when both > 0."""
    if "underlying_bid" not in df.columns or "underlying_ask" not in df.columns:
        return None
    ub = df["underlying_bid"].values
    ua = df["underlying_ask"].values
    ok = (ub > 0) & (ua > 0)
    if not ok.any():
        return None
    return np.where(ok, (ub + ua) / 2, np.nan)


def to_training_arrays(
    df: pd.DataFrame,
    time_unit: str = "year",
    S0_ref: float | None = None,
    min_price: float = 0.0,
    r: float = 0.0,
    synthetic_call_from_puts: bool = False,
    s0_fallback: str = "mean_strike",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Convert dataframe to arrays for training.

    min_price:
        If > 0, drop rows with option mid < min_price (stabilizes relative-error losses).

    synthetic_call_from_puts:
        If True, rows with option_type P are converted with put-call parity
        ``C = P + S0 - K exp(-r τ)`` so targets match the pipeline's **call** pricer.

    s0_fallback (when no ``S0_ref`` and no per-row ``underlying_bid``/``underlying_ask``):
        ``mean_strike`` — scalar ``mean(K)`` for all rows (legacy).
        ``daily_median_strike`` — per calendar quote-day ``median(K)`` (rough ATM proxy).
        ``daily_vwap_strike`` — per day ``sum(K * w) / sum(w)`` with ``w = open_interest`` (from Volume).
        Option **premium** (bid+ask)/2 cannot be used as spot: wrong economic meaning and, in many
        exports (e.g. BTC coin margined), not even the same units as USD strikes.

    Returns:
        tau: time to maturity in years, (T - t0) — t0 and T are already in years
        k: log-moneyness log(K / S0)
        K: strike (for pricing and reporting)
        price: mid option price (target)
        S0: mean of per-row spot (metadata / legacy scalar)
        S0_row: per-row spot used for ``k`` and parity (same length as ``tau``); pass to the pricer
        so Carr–Madan uses the same ``S0`` as targets when ``S0_ref`` is None.
    """
    df = df.copy()
    t0_min = df["quote_datetime"].min()
    df["t0"] = (df["quote_datetime"] - t0_min).dt.total_seconds() / (365.25 * 24 * 3600)
    df["T"] = (df["expiration"] - t0_min).dt.total_seconds() / (365.25 * 24 * 3600)
    df = df[df["T"] > df["t0"]].copy()
    # t0, T are already fractional years; do NOT divide by 365.25 again
    df["tau"] = df["T"] - df["t0"]

    # S0: use provided ref, or per-row underlying when available, else global mean strike
    S0_row = get_underlying_mid(df)
    if S0_ref is not None:
        S0_used = np.full(len(df), float(S0_ref))
    elif S0_row is not None and not np.isnan(S0_row).all():
        # Where underlying is missing, fill with mean strike
        S0_used = np.where(np.isnan(S0_row), np.nanmean(df["strike"]), S0_row)
    else:
        day = df["quote_datetime"].dt.normalize()
        if s0_fallback == "daily_median_strike":
            S0_used = df.groupby(day, sort=False)["strike"].transform("median").astype(np.float64).values
        elif s0_fallback == "daily_vwap_strike":
            w = np.maximum(df["open_interest"].astype(np.float64).values, 1e-6)
            num = pd.Series(df["strike"].values * w, index=df.index).groupby(day).transform("sum")
            den = pd.Series(w, index=df.index).groupby(day).transform("sum")
            S0_used = (num / den).values.astype(np.float64)
        elif s0_fallback == "mean_strike":
            S0_used = np.full(len(df), float(df["strike"].mean()))
        else:
            raise ValueError(
                f"Unknown s0_fallback={s0_fallback!r}; "
                "use mean_strike | daily_median_strike | daily_vwap_strike"
            )

    df["S0"] = S0_used
    df["k"] = np.log(df["strike"].values / df["S0"].values)

    if synthetic_call_from_puts and "option_type" in df.columns:
        is_put = df["option_type"].astype(str).str.upper().str.startswith("P")
        if bool(is_put.any()):
            Pm = df.loc[is_put, "mid"].astype(np.float64)
            S0p = df.loc[is_put, "S0"].astype(np.float64)
            Kp = df.loc[is_put, "strike"].astype(np.float64)
            taup = df.loc[is_put, "tau"].astype(np.float64)
            df.loc[is_put, "mid"] = (Pm + S0p - Kp * np.exp(-float(r) * taup)).astype(np.float32)

    if min_price > 0:
        df = df[df["mid"] >= float(min_price)].copy()

    tau = df["tau"].values.astype(np.float32)
    k = df["k"].values.astype(np.float32)
    K = df["strike"].values.astype(np.float32)
    price = df["mid"].values.astype(np.float32)

    S0_row = df["S0"].values.astype(np.float32)
    S0 = float(np.mean(S0_row))

    return tau, k, K, price, S0, S0_row


def split_by_quote_date(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split on calendar quote date (normalized to midnight).
    train_frac + val_frac + test_frac should sum to 1.0.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1, got {total}")
    day = df["quote_datetime"].dt.normalize()
    uniq = np.sort(day.unique())
    n = len(uniq)
    if n < 3:
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()
    n_tr = max(1, int(round(n * train_frac)))
    n_va = max(1, int(round(n * val_frac)))
    n_te = n - n_tr - n_va
    if n_te < 1:
        # With few quote days, rounding can leave no test days; borrow from train.
        need = 1 - n_te
        n_tr = max(1, n_tr - need)
        n_te = n - n_tr - n_va
    if n_te < 1:
        n_va = max(1, n_va - 1)
        n_te = n - n_tr - n_va
    tr_days = set(uniq[:n_tr])
    va_days = set(uniq[n_tr : n_tr + n_va])
    te_days = set(uniq[n_tr + n_va :])
    train_df = df[day.isin(tr_days)].copy()
    val_df = df[day.isin(va_days)].copy()
    test_df = df[day.isin(te_days)].copy()
    return train_df, val_df, test_df


def strike_band_scalar_for_filter(df: pd.DataFrame, s0_ref: float | None) -> float:
    """Scalar S0 used only for strike band edges (parity uses per-row/ref inside ``to_training_arrays``)."""
    if s0_ref is not None:
        return float(s0_ref)
    return float(df["strike"].mean())


def apply_strike_band_by_spot_scalar(
    df: pd.DataFrame,
    *,
    spot_scalar: float,
    rel_low: float,
    rel_high: float,
) -> pd.DataFrame:
    """Keep rows with rel_low * spot_scalar < strike < rel_high * spot_scalar (strict inequalities)."""
    lo = float(rel_low) * float(spot_scalar)
    hi = float(rel_high) * float(spot_scalar)
    return df[(df["strike"] > lo) & (df["strike"] < hi)].copy()


def normalize_tau_for_net(tau: np.ndarray, tau_max: float | None = None) -> np.ndarray:
    """Normalize tau to [0, 1] for NN input; returns (tau / tau_max)."""
    if tau_max is None:
        tau_max = float(np.max(tau)) if len(tau) else 1.0
    tau_max = max(tau_max, 1e-9)
    return (tau / tau_max).astype(np.float32)


if __name__ == "__main__":
    df = load_options_data()
    print("Loaded options shape:", df.shape)
    tau, k, K, price, S0, _S0_row = to_training_arrays(df)
    print("tau min/max:", tau.min(), tau.max())
    print("k min/max:", k.min(), k.max())
    print("K min/max:", K.min(), K.max())
    print("price min/max:", price.min(), price.max())
    print("S0:", S0)
