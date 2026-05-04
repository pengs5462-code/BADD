#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    p = argparse.ArgumentParser("Aggregate multi-seed BADD experiment summaries")
    p.add_argument("--root", required=True, help="output root, e.g. paper_experiments_reviewer1")
    p.add_argument("--out", default=None, help="CSV output path")
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    for path in root.rglob("summary.json"):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["run_dir"] = str(path.parent)
        rows.append(d)
    if not rows:
        raise SystemExit(f"No summary.json found under {root}")
    df = pd.DataFrame(rows)
    out = args.out or str(root / "all_run_summaries.csv")
    df.to_csv(out, index=False)

    group_cols = ["dataset", "peer_a", "peer_b", "mode"]
    agg = df.groupby(group_cols).agg(
        n=("seed", "count"),
        best_acc_a_mean=("best_acc_a", "mean"),
        best_acc_a_std=("best_acc_a", "std"),
        best_acc_b_mean=("best_acc_b", "mean"),
        best_acc_b_std=("best_acc_b", "std"),
        final_acc_a_mean=("final_acc_a", "mean"),
        final_acc_a_std=("final_acc_a", "std"),
        final_acc_b_mean=("final_acc_b", "mean"),
        final_acc_b_std=("final_acc_b", "std"),
    ).reset_index()
    agg_path = str(root / "aggregate_mean_std.csv")
    agg.to_csv(agg_path, index=False)
    print(f"Saved raw summaries: {out}")
    print(f"Saved mean/std table: {agg_path}")
    print(agg.to_string(index=False))

    # Paired BADD-vs-baseline deltas where both seeds exist.
    if "baseline" in set(df["mode"]) and "badd" in set(df["mode"]):
        key_cols = ["dataset", "peer_a", "peer_b", "seed"]
        base = df[df.mode == "baseline"][key_cols + ["best_acc_a", "best_acc_b"]].rename(columns={"best_acc_a": "base_a", "best_acc_b": "base_b"})
        badd = df[df.mode == "badd"][key_cols + ["best_acc_a", "best_acc_b"]].rename(columns={"best_acc_a": "badd_a", "best_acc_b": "badd_b"})
        merged = pd.merge(base, badd, on=key_cols, how="inner")
        if not merged.empty:
            merged["delta_a"] = merged["badd_a"] - merged["base_a"]
            merged["delta_b"] = merged["badd_b"] - merged["base_b"]
            paired_path = str(root / "paired_badd_vs_baseline.csv")
            merged.to_csv(paired_path, index=False)
            print(f"Saved paired BADD-vs-baseline deltas: {paired_path}")
            print(merged.groupby(["dataset", "peer_a", "peer_b"])[["delta_a", "delta_b"]].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()
