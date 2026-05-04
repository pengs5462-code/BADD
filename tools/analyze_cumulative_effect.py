#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser("Analyze cumulative KL/gradient redistribution caused by micro perturbations")
    p.add_argument("--root", required=True)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir or root / "analysis_cumulative_effect")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in root.rglob("epoch_metrics.csv"):
        df = pd.read_csv(f)
        if df.empty:
            continue
        final = df.iloc[-1]
        row = {"run_dir": str(f.parent)}
        for col in ["cum_abs_reallocated_kl_a", "cum_abs_reallocated_kl_b", "cum_abs_reallocated_grad_a", "cum_abs_reallocated_grad_b"]:
            if col in df.columns:
                row[col] = final[col]
        rows.append(row)
        plt.figure()
        for col in ["cum_abs_reallocated_kl_a", "cum_abs_reallocated_kl_b"]:
            if col in df.columns:
                plt.plot(df["epoch"], df[col], label=col)
        plt.xlabel("Epoch")
        plt.ylabel("Cumulative |w-1| weighted KL mass")
        plt.title(f.parent.name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{f.parent.name}_cumulative_kl.png", dpi=200)
        plt.close()
    if not rows:
        raise SystemExit("No epoch_metrics.csv found")
    out_csv = out_dir / "cumulative_effect_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
