#!/usr/bin/env python3
"""
plot_figures.py — Phase 1h visualisations.

Generates:
  1. headline_covariation.{pdf,png} — dehum_density vs net migration over time, per outlet (dual axis)
  2. subcategory_stacked_area.{pdf,png} — dehumanisation subcategory mix over time, per outlet
  3. subcategory_convergence_bar.{pdf,png} — time-averaged subcategory composition per outlet

Usage:
    python plot_figures.py \
        --aggregated aggregated_migration.csv \
        --subcategory-aggregated subcategory_aggregated.csv \
        --subcategory-composition subcategory_composition.csv \
        --output figures/
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- house style ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 110,
})

OUTLET_ORDER = ["guardian", "independent", "express", "mirror"]
OUTLET_LABEL = {
    "guardian": "The Guardian", "independent": "The Independent",
    "express": "Daily Express", "mirror": "Daily Mirror",
}
# detrended Spearman rho (from composite_detrend_results.csv)
RHO_DET = {"express": 0.53, "guardian": 0.59, "independent": 0.53, "mirror": 0.55}

# Okabe-Ito colour-blind-safe palette
SUBCATS = [
    ("dehum_criminalisation_density", "Criminalisation", "#0072B2"),
    ("dehum_threat_density",          "Threat",          "#D55E00"),
    ("dehum_mass_metaphor_density",   "Mass metaphor",   "#009E73"),
    ("dehum_burden_density",          "Burden",          "#E69F00"),
]
C_DEHUM = "#0072B2"
C_MIG = "#444444"
METHOD_BREAK = pd.Timestamp("2020-03-01")  # IPS -> LTIM


def add_date(df):
    df = df.copy()
    df["date"] = pd.to_datetime(dict(year=df.quarter_end_year,
                                     month=df.quarter_end_month, day=1))
    return df.sort_values(["outlet", "date"])


def _method_line(ax):
    ax.axvline(METHOD_BREAK, color="0.55", ls=":", lw=1.0, zorder=1)


def fig_headline(mig, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    for ax, outlet in zip(axes.ravel(), OUTLET_ORDER):
        d = mig[mig.outlet == outlet]
        l1, = ax.plot(d.date, d.dehum_density, color=C_DEHUM, lw=1.8, zorder=3)
        ax.set_ylabel("Dehumanisation density", color=C_DEHUM)
        ax.tick_params(axis="y", labelcolor=C_DEHUM)
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        l2, = ax2.plot(d.date, d.net_migration / 1000, color=C_MIG, lw=1.6,
                       ls="--", zorder=2)
        ax2.set_ylabel("Net migration (000s)", color=C_MIG)
        ax2.tick_params(axis="y", labelcolor=C_MIG)
        ax2.grid(False)
        _method_line(ax)
        ax.set_title(f"{OUTLET_LABEL[outlet]}   (detrended \u03c1 = {RHO_DET[outlet]:.2f}***)")
    fig.legend([l1, l2], ["Dehumanisation density", "Net migration"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Dehumanisation density tracks net migration, 2012\u20132025",
                 fontsize=13, y=0.99)
    fig.text(0.5, 0.93, "Dotted line: ONS measurement change (IPS \u2192 LTIM, 2020)",
             ha="center", fontsize=8.5, color="0.4")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    for ext in ("pdf", "png"):
        fig.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_stacked(subagg, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    cols = [c for c, _, _ in SUBCATS]
    labels = [l for _, l, _ in SUBCATS]
    colors = [c for _, _, c in SUBCATS]
    for ax, outlet in zip(axes.ravel(), OUTLET_ORDER):
        d = subagg[subagg.outlet == outlet]
        ax.stackplot(d.date, [d[c] for c in cols], colors=colors, alpha=0.92,
                     edgecolor="white", linewidth=0.2)
        ax.set_ylabel("Density (stacked)")
        _method_line(ax)
        ax.set_title(OUTLET_LABEL[outlet])
        ax.margins(x=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Dehumanisation subcategory mix over time "
                 "(stacked density = total dehumanisation)", fontsize=13, y=0.99)
    fig.text(0.5, 0.93, "Criminalisation is the dominant register across all four outlets "
             "\u2014 the shared-lexicon pattern", ha="center", fontsize=8.5, color="0.4")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    for ext in ("pdf", "png"):
        fig.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_convergence(comp, outpath):
    # pivot to share_of_dehum per outlet x subcategory
    label_map = {c: l for c, l, _ in SUBCATS}
    color_map = {l: col for _, l, col in SUBCATS}
    comp = comp.copy()
    comp["sub"] = comp.subcategory.map(label_map)
    piv = comp.pivot(index="outlet", columns="sub", values="share_of_dehum")
    piv = piv.reindex(OUTLET_ORDER)[[l for _, l, _ in SUBCATS]]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    left = np.zeros(len(piv))
    ypos = np.arange(len(piv))
    for col in piv.columns:
        vals = piv[col].values
        ax.barh(ypos, vals, left=left, color=color_map[col], label=col,
                edgecolor="white", height=0.62)
        for y, v, l0 in zip(ypos, vals, left):
            if v > 0.06:
                ax.text(l0 + v / 2, y, f"{v*100:.0f}%", va="center", ha="center",
                        fontsize=8.5, color="white", fontweight="bold")
        left += vals
    ax.set_yticks(ypos)
    ax.set_yticklabels([OUTLET_LABEL[o] for o in piv.index])
    ax.set_xlabel("Share of total dehumanisation")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.32))
    ax.set_title("Convergence: criminalisation dominates dehumanisation in every outlet")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregated", default="aggregated_migration.csv")
    ap.add_argument("--subcategory-aggregated", default="subcategory_aggregated.csv")
    ap.add_argument("--subcategory-composition", default="subcategory_composition.csv")
    ap.add_argument("--output", default="figures/")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)

    mig = add_date(pd.read_csv(a.aggregated))
    subagg = add_date(pd.read_csv(a.subcategory_aggregated))
    comp = pd.read_csv(a.subcategory_composition)

    fig_headline(mig, os.path.join(a.output, "headline_covariation"))
    fig_stacked(subagg, os.path.join(a.output, "subcategory_stacked_area"))
    fig_convergence(comp, os.path.join(a.output, "subcategory_convergence_bar"))
    print("Wrote figures to", a.output)


if __name__ == "__main__":
    main()
