"""Regenerate h2h3_topology.pdf (gate topology across modalities, H2 and H3).

Panel (a) Hoyer sparsity under both prior encodings, (b) smoothness per raw
variable index — the misleading comparison — and (c) smoothness per cm^-1,
the comparison that controls for spectral sampling density.

The Gaussian-prior NIR sparsity is read from ablation_h3_result.npz when
present (produced by scripts/ablation_h3_smoothness.py). The remaining
values are those reported in Section 6.2; they come from
scripts/reanalyze_smoothness_physical.py, which prints rather than saves
them.

Fonts are embedded as TrueType (pdf.fonttype 42) rather than matplotlib's
default Type 3, which Elsevier production does not accept.

Usage:
    python scripts/make_h2h3_figure.py [--npz ablation_h3_result.npz]
                                       [--out h2h3_topology.pdf]
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

BLUE, BLUE_L, ORANGE = "#4C72B0", "#A1C9F4", "#DD8452"

# Reported in Section 6.2; see the module docstring for provenance.
SPARSITY_NIR_STEP = 0.3176
SPARSITY_NIR_GAUSS = 0.2345      # overridden by the npz when available
SPARSITY_RAMAN = 0.4150
TV_RAW_NIR, TV_RAW_RAMAN = 0.0167, 0.0076
TV_PHYS_NIR, TV_PHYS_RAMAN = 3.69e-4, 3.795e-3


def label_bars(ax, bars, fmt):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                format(b.get_height(), fmt), ha="center", va="bottom",
                fontsize=11)


def panel(ax, text):
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="ablation_h3_result.npz")
    ap.add_argument("--out", default="h2h3_topology.pdf")
    args = ap.parse_args()

    s_nir_gauss = SPARSITY_NIR_GAUSS
    p = Path(args.npz)
    if p.exists():
        s_nir_gauss = float(np.load(p, allow_pickle=True)["sparsity_nir_gauss"])
        print(f"NIR Gaussian-prior sparsity read from {p}: {s_nir_gauss:.4f}")
    else:
        print("npz not found; using the value reported in the manuscript")

    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    vals = [SPARSITY_NIR_STEP, s_nir_gauss, SPARSITY_RAMAN]
    bars = ax.bar(range(3), vals, color=[BLUE, BLUE_L, ORANGE], width=0.6)
    label_bars(ax, bars, ".3f")
    ax.set_ylabel("Hoyer sparsity index")
    ax.set_ylim(0, 0.52)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["NIR\n(step prior)", "NIR\n(Gaussian prior)",
                        "Raman\n(Gaussian prior)"], fontsize=11)
    panel(ax, "(a)")

    ax = axes[1]
    bars = ax.bar([0, 1], [TV_RAW_NIR, TV_RAW_RAMAN], color=[BLUE, ORANGE],
                  width=0.5)
    label_bars(ax, bars, ".4f")
    ax.set_ylabel("TV per raw band index")
    ax.set_ylim(0, 0.0185)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["NIR", "Raman"])
    panel(ax, "(b)")

    ax = axes[2]
    bars = ax.bar([0, 1], [TV_PHYS_NIR, TV_PHYS_RAMAN], color=[BLUE, ORANGE],
                  width=0.5)
    label_bars(ax, bars, ".6f")
    ax.set_ylabel(r"TV per cm$^{-1}$ (physical unit)")
    ax.set_ylim(0, 0.00425)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["NIR", "Raman"])
    panel(ax, "(c)")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
