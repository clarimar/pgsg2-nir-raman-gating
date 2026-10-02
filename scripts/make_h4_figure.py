"""Regenerate h4_prior_ablation.pdf with a third panel for the displacement control.

Panel (a) predictive performance, (b) stability of the final gate,
(c) stability of the learned displacement d = g_final - g0.

Values are read from revision_r1/r1_h4_displacement.json when present
(produced by scripts/r1_h4_displacement_control.py); otherwise the figures
reported in the manuscript are used.

Usage:
    python scripts/make_h4_figure.py [--json revision_r1/r1_h4_displacement.json]
                                     [--out h4_prior_ablation.pdf]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# seaborn-deep palette, as in the other figures of this manuscript
BLUE, BLUE_L = "#4C72B0", "#A1C9F4"
ORANGE, ORANGE_L = "#DD8452", "#FFB482"

FALLBACK = {
    "nir_literature":   {"r2_mean": 0.7933, "r2_std": 0.0080,
                         "gate": {"rho_mean": 0.9995, "jaccard_mean": 0.9418},
                         "displacement": {"rho_mean": 0.9462, "jaccard_mean": 0.3767}},
    "nir_uninformed":   {"r2_mean": 0.7983, "r2_std": 0.0077,
                         "gate": {"rho_mean": 0.8653, "jaccard_mean": 0.6002},
                         "displacement": {"rho_mean": 0.8653, "jaccard_mean": 0.6002}},
    "raman_literature": {"r2_mean": 0.9302, "r2_std": 0.0015,
                         "gate": {"rho_mean": 0.9976, "jaccard_mean": 0.9011},
                         "displacement": {"rho_mean": 0.9760, "jaccard_mean": 0.7943}},
    "raman_uninformed": {"r2_mean": 0.9310, "r2_std": 0.0014,
                         "gate": {"rho_mean": 0.8931, "jaccard_mean": 0.6545},
                         "displacement": {"rho_mean": 0.8931, "jaccard_mean": 0.6545}},
}


def panel_label(ax, text):
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="revision_r1/r1_h4_displacement.json")
    ap.add_argument("--out", default="h4_prior_ablation.pdf")
    args = ap.parse_args()

    d = FALLBACK
    p = Path(args.json)
    if p.exists():
        loaded = json.loads(p.read_text())
        if all(k in loaded for k in FALLBACK):
            d = loaded
            print(f"values read from {p}")
    else:
        print("JSON not found; using the values reported in the manuscript")

    mods = ["NIR", "Raman"]
    x = np.arange(len(mods))
    w = 0.35

    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 13,
        "xtick.labelsize": 13, "ytick.labelsize": 12,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # ---- (a) predictive performance -------------------------------------
    ax = axes[0]
    lit = [d["nir_literature"]["r2_mean"], d["raman_literature"]["r2_mean"]]
    rnd = [d["nir_uninformed"]["r2_mean"], d["raman_uninformed"]["r2_mean"]]
    lit_e = [d["nir_literature"]["r2_std"], d["raman_literature"]["r2_std"]]
    rnd_e = [d["nir_uninformed"]["r2_std"], d["raman_uninformed"]["r2_std"]]
    ax.bar(x - w/2, lit, w, yerr=lit_e, capsize=4, color=BLUE,
           label="Literature prior")
    ax.bar(x + w/2, rnd, w, yerr=rnd_e, capsize=4, color=ORANGE,
           label="Uninformed init.")
    ax.set_ylabel(r"$R^2$ (test set, mean $\pm$ SD)")
    ax.set_ylim(0.60, 1.10)
    ax.set_yticks(np.arange(0.60, 1.001, 0.05))
    ax.set_xticks(x); ax.set_xticklabels(mods)
    ax.legend(loc="upper center", fontsize=11.5, frameon=True)
    panel_label(ax, "(a)")

    # ---- (b) stability of the final gate --------------------------------
    ax = axes[1]
    wb = 0.2
    series = [
        ([d["nir_literature"]["gate"]["rho_mean"],
          d["raman_literature"]["gate"]["rho_mean"]], BLUE,
         r"$\rho$ (literature prior)", -1.5),
        ([d["nir_uninformed"]["gate"]["rho_mean"],
          d["raman_uninformed"]["gate"]["rho_mean"]], BLUE_L,
         r"$\rho$ (uninformed)", -0.5),
        ([d["nir_literature"]["gate"]["jaccard_mean"],
          d["raman_literature"]["gate"]["jaccard_mean"]], ORANGE,
         "Jaccard (literature prior)", 0.5),
        ([d["nir_uninformed"]["gate"]["jaccard_mean"],
          d["raman_uninformed"]["gate"]["jaccard_mean"]], ORANGE_L,
         "Jaccard (uninformed)", 1.5),
    ]
    for vals, color, label, off in series:
        ax.bar(x + off * wb, vals, wb, color=color, label=label)
    ax.set_ylabel("Gate stability across seeds")
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(x); ax.set_xticklabels(mods)
    handles, labels = ax.get_legend_handles_labels()
    panel_label(ax, "(b)")

    # ---- (c) stability of the learned displacement ----------------------
    ax = axes[2]
    series = [
        ([d["nir_literature"]["displacement"]["rho_mean"],
          d["raman_literature"]["displacement"]["rho_mean"]], BLUE,
         r"$\rho$ (literature prior)", -1.5),
        ([d["nir_uninformed"]["displacement"]["rho_mean"],
          d["raman_uninformed"]["displacement"]["rho_mean"]], BLUE_L,
         r"$\rho$ (uninformed)", -0.5),
        ([d["nir_literature"]["displacement"]["jaccard_mean"],
          d["raman_literature"]["displacement"]["jaccard_mean"]], ORANGE,
         "Jaccard (literature prior)", 0.5),
        ([d["nir_uninformed"]["displacement"]["jaccard_mean"],
          d["raman_uninformed"]["displacement"]["jaccard_mean"]], ORANGE_L,
         "Jaccard (uninformed)", 1.5),
    ]
    for vals, color, label, off in series:
        ax.bar(x + off * wb, vals, wb, color=color, label=label)
    ax.set_ylabel("Displacement stability across seeds")
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(x); ax.set_xticklabels(mods)
    panel_label(ax, "(c)")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=12,
               frameon=False, bbox_to_anchor=(0.62, -0.01))
    fig.savefig(args.out, bbox_inches="tight")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
