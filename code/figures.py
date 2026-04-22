"""
figures.py — Publication-quality figures for CBA786 report

Generates 4 figures:
  Figure 1: Spearman correlation heatmap (migration + hate crime)
  Figure 2: Time series — dehumanisation density by outlet (2010–2025)
  Figure 3: Scatter — dehumanisation density × net migration
  Figure 4: Lag comparison — hypothesised vs reverse direction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["STIXGeneral", "DejaVu Serif", "Liberation Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "lines.linewidth":    1.4,
    "patch.linewidth":    0.5,
    "text.usetex":        False,
    "mathtext.fontset":   "stix",
})

FEATURES = {
    "sentiment_compound":    "Sentiment\n(VADER compound)",
    "dehum_density":         "Dehumanisation\n(per 1,000 words)",
    "outrage_density":       "Moral outrage\n(per 1,000 words)",
    "modal_epistemic_ratio": "Epistemic\nratio",
    "pron_inclusion_index":  "Inclusion\nindex",
}
FEAT_COLS = list(FEATURES.keys())

OUTLET_ORDER  = ["Daily Express", "Daily Mirror", "The Independent", "guardian"]
OUTLET_LABELS = {
    "Daily Express":    "Daily Express",
    "Daily Mirror":     "Daily Mirror",
    "The Independent":  "The Independent",
    "guardian":         "The Guardian",
}

# Colourblind-safe palette (4 colours, distinguishable in greyscale)
OUTLET_COLOURS = {
    "Daily Express":    "#D55E00",   # vermillion
    "Daily Mirror":     "#CC79A7",   # reddish purple
    "The Independent":  "#009E73",   # bluish green
    "guardian":         "#0072B2",   # blue
}

OUTPUT = "/home/claude/figures"


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

def figure_1(spearman_df):
    fig = plt.figure(figsize=(7.5, 4.2))

    # Three columns: heatmap left, colourbar centre, heatmap right (mirrored)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.06, 1], wspace=0.15)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    vmin, vmax = -0.85, 0.85
    cmap = sns.color_palette("RdBu_r", as_cmap=True)

    panels = [
        (ax_left,  "migration", "A", "Net migration\n(ONS, quarterly rolling)", False),
        (ax_right, "hate",      "B", "Race hate crime\n(Home Office, annual)",   True),
    ]

    for ax, outcome_key, panel_label, title, mirror in panels:
        sub = spearman_df[spearman_df["outcome"].str.contains(outcome_key)]
        outlet_order = [o for o in OUTLET_ORDER if o in sub["outlet"].unique()]

        # Mirror panel B so outlets read outward from centre
        if mirror:
            outlet_order = list(reversed(outlet_order))

        rho_matrix = pd.DataFrame(
            index=FEAT_COLS, columns=outlet_order, dtype=float,
        )
        annot_matrix = rho_matrix.copy().astype(str)

        for _, r in sub.iterrows():
            if r["outlet"] in rho_matrix.columns and r["feature"] in rho_matrix.index:
                rho_matrix.loc[r["feature"], r["outlet"]] = r["rho"]
                sig = ""
                if r["p_value"] <= 0.001: sig = "***"
                elif r["p_value"] <= 0.01: sig = "**"
                elif r["p_value"] <= 0.05: sig = "*"
                annot_matrix.loc[r["feature"], r["outlet"]] = f"{r['rho']:+.2f}{sig}"

        rho_matrix = rho_matrix.astype(float)
        rho_matrix.index = [FEATURES[f] for f in rho_matrix.index]
        rho_matrix.columns = [OUTLET_LABELS.get(c, c) for c in rho_matrix.columns]
        annot_matrix.index = rho_matrix.index
        annot_matrix.columns = rho_matrix.columns

        sns.heatmap(
            rho_matrix, ax=ax, annot=annot_matrix, fmt="",
            cmap=cmap, center=0, vmin=vmin, vmax=vmax,
            linewidths=0.4, linecolor="white",
            cbar=False,
            annot_kws={"fontsize": 8},
            yticklabels=(not mirror),
        )

        ax.set_title(title, fontsize=10, pad=8)
        ax.tick_params(axis="x", rotation=30, length=0)
        ax.tick_params(axis="y", rotation=0, length=0)
        ax.set_xlabel("")
        ax.set_ylabel("")

        x_pos = -0.15 if not mirror else -0.05
        ax.text(x_pos, 1.05, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")

    # Shared colourbar in centre
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar, ticks=[-0.8, -0.4, 0, 0.4, 0.8])
    cb.set_label("Spearman $\\rho$", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.text(0.5, -0.06,
             "* p < .05    ** p < .01    *** p < .001",
             ha="center", fontsize=8, style="italic", color="#555555")

    path = f"{OUTPUT}/fig1_heatmap.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — TIME SERIES: DEHUMANISATION DENSITY
# ═══════════════════════════════════════════════════════════════════════════════

def figure_2(features_df):
    fig, ax = plt.subplots(figsize=(7.5, 3.5))

    for outlet in OUTLET_ORDER:
        sub = features_df[features_df["outlet"] == outlet]
        annual = sub.groupby("year")["dehum_density"].mean().reset_index()
        annual = annual.sort_values("year")

        ax.plot(annual["year"], annual["dehum_density"],
                color=OUTLET_COLOURS[outlet],
                label=OUTLET_LABELS[outlet],
                marker="o", markersize=3.5, markeredgecolor="white",
                markeredgewidth=0.4, zorder=3)

    # Key events — stagger vertically to avoid overlap
    events = [
        (2015, 0.97, "EU migration crisis"),
        (2016, 0.78, "Brexit referendum"),
        (2020, 0.97, "COVID-19 / LTIM switch"),
    ]
    for yr, y_frac, label in events:
        ax.axvline(yr, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=1)
        ax.text(yr + 0.15, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_frac,
                label, fontsize=7, color="#777777", va="top", ha="left")

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean dehumanisation density\n(hits per 1,000 words)")
    ax.set_xlim(2009.5, 2025.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9,
              edgecolor="#CCCCCC", fancybox=False)

    path = f"{OUTPUT}/fig2_timeseries_dehum.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — SCATTER: DEHUMANISATION × NET MIGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def figure_3(migration_agg, spearman_df):
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.8), sharey=True,
                              gridspec_kw={"wspace": 0.08})

    for i, outlet in enumerate(OUTLET_ORDER):
        ax = axes[i]
        sub = migration_agg[migration_agg["outlet"] == outlet].dropna(
            subset=["dehum_density", "net_migration"]
        )

        # Get Spearman result
        sp = spearman_df[
            (spearman_df["outlet"] == outlet) &
            (spearman_df["feature"] == "dehum_density") &
            (spearman_df["outcome"].str.contains("migration"))
        ].iloc[0]

        sig = ""
        if sp["p_value"] <= 0.001: sig = "***"
        elif sp["p_value"] <= 0.01: sig = "**"
        elif sp["p_value"] <= 0.05: sig = "*"

        ax.scatter(sub["net_migration"] / 1000, sub["dehum_density"],
                   color=OUTLET_COLOURS[outlet], s=18, alpha=0.55,
                   edgecolors="white", linewidth=0.3, zorder=3)

        # Trend line
        z = np.polyfit(sub["net_migration"] / 1000, sub["dehum_density"], 1)
        x_line = np.linspace(sub["net_migration"].min() / 1000,
                             sub["net_migration"].max() / 1000, 100)
        ax.plot(x_line, np.poly1d(z)(x_line),
                color=OUTLET_COLOURS[outlet], linewidth=1, linestyle="--",
                alpha=0.6, zorder=2)

        ax.set_title(OUTLET_LABELS[outlet], fontsize=9, pad=6)
        ax.text(0.05, 0.95,
                f"ρ = {sp['rho']:+.2f}{sig}\nn = {sp['n']}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))

        ax.grid(linewidth=0.3, alpha=0.4)

        if i == 0:
            ax.set_ylabel("Dehumanisation density\n(per 1,000 words)")

    # Single shared x-label
    fig.text(0.5, -0.01, "Net migration (thousands, ONS rolling 12-month)",
             ha="center", fontsize=9)

    path = f"{OUTPUT}/fig3_scatter_dehum_migration.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — LAG COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def figure_4(lag_df):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5),
                              gridspec_kw={"wspace": 0.35})

    feat_labels_short = {
        "sentiment_compound":    "Sentiment",
        "dehum_density":         "Dehumanisation",
        "outrage_density":       "Outrage",
        "modal_epistemic_ratio": "Epistemic ratio",
        "pron_inclusion_index":  "Inclusion index",
    }

    panels = [
        (axes[0], "Migration", "A",
         "Migration → Rhetoric\nvs Rhetoric → Migration",
         "Migration_T → Rhetoric_T+1",
         "Rhetoric_T → Migration_T+1"),
        (axes[1], "HateCrime", "B",
         "Rhetoric → Hate crime\nvs Hate crime → Rhetoric",
         "Rhetoric_T → HateCrime_T+1",
         "HateCrime_T → Rhetoric_T+1"),
    ]

    for ax, outcome_key, panel_label, title, hyp_type, rev_type in panels:
        hyp = lag_df[lag_df["lag_type"] == hyp_type]
        rev = lag_df[lag_df["lag_type"] == rev_type]

        hyp_means = hyp.groupby("feature")["rho"].mean()
        rev_means = rev.groupby("feature")["rho"].mean()

        # Count significant per feature
        hyp_sig = hyp.groupby("feature").apply(
            lambda g: (g["p_value"] <= 0.05).sum()
        )
        rev_sig = rev.groupby("feature").apply(
            lambda g: (g["p_value"] <= 0.05).sum()
        )

        x = np.arange(len(FEAT_COLS))
        w = 0.32

        hyp_vals = [hyp_means.get(f, 0) for f in FEAT_COLS]
        rev_vals = [rev_means.get(f, 0) for f in FEAT_COLS]

        bars_h = ax.bar(x - w/2, hyp_vals, w,
                        color="#0072B2", alpha=0.85,
                        label="Hypothesised direction", zorder=3)
        bars_r = ax.bar(x + w/2, rev_vals, w,
                        color="#D55E00", alpha=0.85,
                        label="Reverse direction", zorder=3)

        # Mark significant bars
        for j, f in enumerate(FEAT_COLS):
            if hyp_sig.get(f, 0) > 0:
                ax.text(x[j] - w/2, hyp_vals[j] + 0.02 * np.sign(hyp_vals[j]),
                        "*", ha="center", fontsize=10, fontweight="bold",
                        color="#0072B2")
            if rev_sig.get(f, 0) > 0:
                ax.text(x[j] + w/2, rev_vals[j] + 0.02 * np.sign(rev_vals[j]),
                        "*", ha="center", fontsize=10, fontweight="bold",
                        color="#D55E00")

        ax.set_xticks(x)
        ax.set_xticklabels([feat_labels_short[f] for f in FEAT_COLS],
                           fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Mean Spearman ρ\n(across outlets)", fontsize=9)
        ax.set_title(title, fontsize=9.5, pad=8)
        ax.axhline(0, color="black", linewidth=0.6, zorder=2)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
        ax.set_ylim(-0.45, 0.65)

        if panel_label == "A":
            ax.legend(loc="upper left", frameon=True, framealpha=0.9,
                      edgecolor="#CCCCCC", fancybox=False, fontsize=8)

        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")

    fig.text(0.5, -0.08,
             "* = at least one outlet significant at p < .05",
             ha="center", fontsize=8, style="italic", color="#555555")

    path = f"{OUTPUT}/fig4_lag_comparison.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT, exist_ok=True)

    print("Generating publication figures...\n")

    features = pd.read_csv("/home/claude/all_features_4outlet.csv", low_memory=False)
    spearman = pd.read_csv("/home/claude/results_4outlet/spearman_results.csv")
    mig_agg = pd.read_csv("/home/claude/results_4outlet/aggregated_migration.csv")
    lag = pd.read_csv("/home/claude/results_4outlet/lag_results.csv")

    figure_1(spearman)
    figure_2(features)
    figure_3(mig_agg, spearman)
    figure_4(lag)

    print("\nAll figures saved to /home/claude/figures/")
