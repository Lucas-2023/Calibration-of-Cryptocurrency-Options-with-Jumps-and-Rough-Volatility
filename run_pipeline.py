"""
Run the full pipeline: preprocess -> train -> evaluate.

Usage:
  python run_pipeline.py [--data-dir ../data] [--epochs 500] [--loss relative_mae] [--min-price 0.05] [--time-split] [--no-eval] [--eval-timestamp-out-dir]
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
OUT_DIR = SCRIPT_DIR / "output"
CKPT_PATH = SCRIPT_DIR / "output" / "best_kernel.pt"


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Path to option CSVs")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--relative-loss", action="store_true", help="Use relative MSE")
    parser.add_argument(
        "--loss",
        type=str,
        choices=("mse", "relative_mse", "relative_mae"),
        default="relative_mae",
        help="Price objective for train.py (default: relative_mae)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.05,
        help="Drop option mids below this for train/eval (0 disables)",
    )
    parser.add_argument(
        "--time-split",
        action="store_true",
        help="Train/val/test by quote date",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Cap training samples")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--save", type=Path, default=CKPT_PATH, help="Checkpoint path")
    parser.add_argument(
        "--eval-timestamp-out-dir",
        action="store_true",
        help="Pass --timestamp-out-dir to evaluate.py (writes plots under output/eval_YYYYmmdd_HHMMSS/).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1–4: preprocess is done inside train
    # Step 5–8: train
    cmd_train = [
        sys.executable, str(SCRIPT_DIR / "train.py"),
        "--data-dir", str(args.data_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--save", str(args.save),
    ]
    if args.relative_loss:
        cmd_train.append("--relative-loss")
    cmd_train.extend(["--loss", args.loss])
    cmd_train.extend(["--min-price", str(args.min_price)])
    if args.time_split:
        cmd_train.append("--time-split")
    if args.max_samples is not None:
        cmd_train.extend(["--max-samples", str(args.max_samples)])

    print("Running: " + " ".join(cmd_train))
    ret = subprocess.run(cmd_train, cwd=SCRIPT_DIR, env=_subprocess_env())
    if ret.returncode != 0:
        print("Training failed.")
        return ret.returncode

    if args.no_eval or not args.save.exists():
        print("Done (evaluation skipped or no checkpoint).")
        return 0

    cmd_eval = [
        sys.executable, str(SCRIPT_DIR / "evaluate.py"),
        "--checkpoint", str(args.save),
        "--data-dir", str(args.data_dir),
        "--out-dir", str(OUT_DIR),
        "--min-price", str(args.min_price),
    ]
    if args.eval_timestamp_out_dir:
        cmd_eval.append("--timestamp-out-dir")
    print("Running: " + " ".join(cmd_eval))
    ret = subprocess.run(cmd_eval, cwd=SCRIPT_DIR, env=_subprocess_env())
    return ret.returncode


if __name__ == "__main__":
    sys.exit(main())
