"""
Revised Fig. 1 (H1) for the R1 of CHEMOLAB-D-26-01046.

Reads one or more r1_h1_runs.csv files (e.g. results_r1_nir/ and
results_r1_raman/), drops the reference repetition "paper", and draws one
panel per branch: mean test R2 (bar), SD (error bar) and individual splits
(points) for PLS-10, PLS-cvmin, PLS-1se and PGSGv2Model. No plot titles;
panel letters only.

Usage:
    python scripts/r1_fig_h1.py results_r1_nir/r1_h1_runs.csv \
        results_r1_raman/r1_h1_runs.csv --out paper/h1_performance.pdf
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["PLS-10", "PLS-cvmin", "PLS-1se", "PGSGv2"]
LABELS = ["PLS\n(H = 10)", "PLS\n(cvmin)", "PLS\n(1-SE)", "PGSGv2Model"]
COLORS = ["#F2C4A0", "#E8A06A", "#DD8452", "#4C72B0"]
BRANCH_ORDER = ["NIR", "Raman"]
BRANCH_LABEL = {"NIR": "NIR (Mango DMC v3)", "Raman": "Raman (bioprocess_substrates)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="h1_performance.pdf")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    df = df[df["repetition"].astype(str) != "paper"]
    branches = [b for b in BRANCH_ORDER if b in set(df["branch"])]

    fig, axes = plt.subplots(1, len(branches), figsize=(4.2 * len(branches), 3.6), squeeze=False)
    rng = np.random.default_rng(0)
    for k, (ax, branch) in enumerate(zip(axes[0], branches)):
        d = df[df["branch"] == branch]
        x = np.arange(len(MODELS))
        for i, m in enumerate(MODELS):
            v = d.loc[d["model"] == m, "r2_test"].to_numpy()
            ax.bar(x[i], v.mean(), width=0.65, color=COLORS[i], edgecolor="none", zorder=2)
            ax.errorbar(x[i], v.mean(), yerr=v.std(ddof=1), color="black",
                        capsize=3, lw=1, zorder=3)
            ax.scatter(x[i] + rng.uniform(-0.15, 0.15, v.size), v, s=9,
                       color="black", alpha=0.55, lw=0, zorder=4)
        lo = d["r2_test"].min()
        ax.set_ylim(max(0.0, np.floor((lo - 0.05) * 20) / 20), 1.0)
        ax.set_xticks(x, LABELS, fontsize=8)
        ax.set_ylabel(r"$R^2$ (test set)" if k == 0 else "")
        ax.set_xlabel(BRANCH_LABEL[branch], fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.02, 0.98, f"({chr(97 + k)})", transform=ax.transAxes,
                va="top", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
