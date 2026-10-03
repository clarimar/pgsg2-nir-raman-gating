"""Regenerate h5_correlation.pdf (H5: local gate structure vs. linewidth).

Three scatter panels over the twenty glucose Raman bands: local gate FWHM,
normalized local entropy, and local smoothness, each against the
literature-reported linewidth, with an ordinary least-squares fit and the
Pearson correlation in the panel label.

All values are read from h5_bandwidth_correlation_result.npz, produced by
scripts/run_h5_bandwidth_correlation.py. The script computes the
correlations from the data rather than restating them, so the figure cannot
drift from the numbers in Section 6.4.

Fonts are embedded as TrueType (pdf.fonttype 42) rather than matplotlib's
default Type 3, which Elsevier production does not accept.

Usage:
    python scripts/make_h5_figure.py [--npz h5_bandwidth_correlation_result.npz]
                                     [--out h5_correlation.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

POINT, FIT = "#4C72B0", "#DD8452"


def scatter_panel(ax, x, y, ylabel, tag):
    r, p = pearsonr(x, y)
    ax.scatter(x, y, s=38, color=POINT, zorder=3)
    xs = np.linspace(x.min(), x.max(), 100)
    b, a = np.polyfit(x, y, 1)
    ax.plot(xs, a + b * xs, "--", color=FIT, lw=1.6, zorder=2)
    ax.set_xlabel(r"Reported literature linewidth (cm$^{-1}$)")
    ax.set_ylabel(ylabel)
    ax.text(0.03, 0.97, f"{tag} r={r:+.3f}, p={p:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    return r, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="h5_bandwidth_correlation_result.npz")
    ap.add_argument("--out", default="h5_correlation.pdf")
    args = ap.parse_args()

    p = Path(args.npz)
    if not p.exists():
        print(f"[error] {p} not found — run scripts/run_h5_bandwidth_correlation.py first")
        return 1
    d = np.load(p, allow_pickle=True)
    w = np.asarray(d["real_widths"], dtype=float)
    print(f"{len(w)} bands read from {p}")

    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    panels = [
        (np.asarray(d["fwhms"], dtype=float),
         r"Local gate FWHM (cm$^{-1}$)", "(a)"),
        (np.asarray(d["entropies_normalized"], dtype=float),
         r"Normalized local entropy ($H/\log n$)", "(b)"),
        (np.asarray(d["smoothnesses"], dtype=float),
         "Local smoothness (TV)", "(c)"),
    ]
    for ax, (y, ylabel, tag) in zip(axes, panels):
        r, pv = scatter_panel(ax, w, y, ylabel, tag)
        print(f"  {tag} r = {r:+.3f}  p = {pv:.3f}")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
