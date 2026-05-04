#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def low_freq_ratio(x, frac=0.1):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.nan
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    if len(power) <= 1:
        return np.nan
    k = max(1, int(len(power) * frac))
    return float(power[1:k+1].sum() / (power[1:].sum() + 1e-12))


def analyze_file(path: Path, out_dir: Path):
    df = pd.read_csv(path)
    candidates = [
        "a_learns_from_b_msp_mean",
        "a_learns_from_b_residual_std",
        "a_learns_from_b_grad_proxy_mean",
        "a_learns_from_b_grad_proxy_weighted_mean",
        "b_learns_from_a_msp_mean",
        "b_learns_from_a_residual_std",
        "b_learns_from_a_grad_proxy_mean",
        "b_learns_from_a_grad_proxy_weighted_mean",
    ]
    rows = []
    for col in candidates:
        if col in df.columns:
            rows.append({"run_dir": str(path.parent), "signal": col, "low_freq_ratio_10pct": low_freq_ratio(df[col].values, 0.1)})
            # Plot PSD for each signal.
            x = df[col].dropna().values.astype(float)
            if len(x) >= 8:
                x = x - x.mean()
                power = np.abs(np.fft.rfft(x)) ** 2
                freq = np.fft.rfftfreq(len(x), d=1.0)
                plt.figure()
                plt.plot(freq[1:], power[1:])
                plt.xlabel("Normalized frequency")
                plt.ylabel("Power")
                plt.title(col)
                plt.tight_layout()
                safe = col.replace("/", "_")
                plt.savefig(out_dir / f"{path.parent.name}_{safe}_psd.png", dpi=200)
                plt.close()
    return rows


def main():
    p = argparse.ArgumentParser("Frequency-domain evidence for confidence drift removal")
    p.add_argument("--root", required=True)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir or root / "analysis_frequency")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for trace in root.rglob("filter_gradient_trace.csv"):
        all_rows.extend(analyze_file(trace, out_dir))
    if not all_rows:
        raise SystemExit("No filter_gradient_trace.csv found")
    out_csv = out_dir / "frequency_low_ratio_summary.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
