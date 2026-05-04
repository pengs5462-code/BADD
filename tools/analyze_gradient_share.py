#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser("Analyze gradient-budget redistribution / gradient share")
    p.add_argument("--root", required=True)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir or root / "analysis_gradient_share")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in root.rglob("epoch_metrics.csv"):
        df = pd.read_csv(f)
        if df.empty:
            continue
        last = df.tail(max(1, min(20, len(df))))
        meta = {"run_dir": str(f.parent)}
        for col in ["grad_share_top_shift_a", "grad_share_top_shift_b", "grad_share_top_weighted_a", "grad_share_top_weighted_b", "grad_share_top_unweighted_a", "grad_share_top_unweighted_b"]:
            if col in df.columns:
                meta[f"last20_{col}"] = last[col].mean()
        rows.append(meta)
        for side in ["a", "b"]:
            wu = f"grad_share_top_unweighted_{side}"
            ww = f"grad_share_top_weighted_{side}"
            if wu in df.columns and ww in df.columns:
                plt.figure()
                plt.plot(df["epoch"], df[wu], label="unweighted")
                plt.plot(df["epoch"], df[ww], label="weighted")
                plt.xlabel("Epoch")
                plt.ylabel("Top-residual gradient share")
                plt.title(f"{f.parent.name} side {side}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / f"{f.parent.name}_side_{side}_gradient_share.png", dpi=200)
                plt.close()
    if not rows:
        raise SystemExit("No epoch_metrics.csv found")
    out_csv = out_dir / "gradient_share_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
