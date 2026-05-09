"""
Plot full-model training convergence from ``--convergence-csv`` produced by ``train_full.py``.

Usage:
  python plot_full_model_convergence.py --csv output/full_model_convergence.csv --out output/convergence.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_convergence_csv(path: Path) -> dict[str, list]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"No rows in {path}")
    keys = rows[0].keys()
    out: dict[str, list] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            out[k].append(row.get(k, "").strip())
    return out


def _col_float(data: dict[str, list], name: str) -> tuple[list[int], list[float]]:
    ep = []
    vals = []
    for i, e in enumerate(data["epoch"]):
        s = data[name][i] if name in data else ""
        if s == "":
            continue
        try:
            vals.append(float(s))
            ep.append(int(float(e)))
        except ValueError:
            continue
    return ep, vals


def main() -> None:
    p = argparse.ArgumentParser(description="Plot full model convergence CSV")
    p.add_argument("--csv", type=Path, required=True, help="Path to convergence CSV from train_full.py")
    p.add_argument("--out", type=Path, default=Path("output/convergence_plot.png"))
    args = p.parse_args()

    data = _read_convergence_csv(args.csv)
    epochs = [int(float(x)) for x in data["epoch"]]
    train = [float(x) for x in data["train_loss"]]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax0 = axes[0]
    ax0.plot(epochs, train, label="train_loss", color="tab:blue", alpha=0.85)
    val_ep, val_y = _col_float(data, "val_loss")
    if val_y:
        ax0.plot(val_ep, val_y, label="val_loss", color="tab:orange", alpha=0.85)
    sel = [float(x) for x in data["selection_metric"]]
    ax0.plot(epochs, sel, label="selection_metric (this epoch)", color="tab:green", alpha=0.6, linewidth=1)
    best = [float(x) for x in data["best_metric_so_far"]]
    ax0.plot(epochs, best, label="best_metric_so_far", color="tab:red", linewidth=1.5)
    ax0.set_ylabel("loss / metric")
    ax0.set_title("Full model: loss convergence")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    for name, c in [
        ("V0", "tab:purple"),
        ("V_bar", "tab:brown"),
        ("k", "tab:pink"),
        ("a", "tab:gray"),
        ("b", "tab:olive"),
        ("c", "tab:cyan"),
    ]:
        ys = [float(x) for x in data[name]]
        ax1.plot(epochs, ys, label=name, color=c, linewidth=1.0)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("learned scalars")
    ax1.set_title("Tempered-stable / mean-reversion parameters over epochs")
    ax1.legend(loc="upper right", ncol=3, fontsize=7)
    ax1.grid(True, alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
