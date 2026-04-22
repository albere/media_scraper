"""
lag_and_visuals.py
Immigration News Corpus — Lag Analysis & Visualisations

Runs:
  1. Lag-1 Spearman correlations
     - Migration_T → Rhetoric_{T+1}  (does migration drive rhetoric?)
     - Rhetoric_T → Hate_crime_{T+1} (does rhetoric drive hate crime?)
     - Reverse lags as robustness checks
  2. Visualisations
     - Correlation heatmaps (migration & hate crime)
     - Time series of annual feature means by outlet
     - Scatter plots for strongest correlations

Usage:
    python lag_and_visuals.py \
        --features all_features.csv \
        --ons ltimnov25.xlsx \
        --hatecrime hate-crime-england-and-wales-2024-to-2025-data-tables.ods \
        --spearman results/spearman_results.csv \
        --output results/
"""

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

FEATURES = {
    "sentiment_compound":    "Sentiment",
    "dehum_density":         "Dehumanisation",
    "outrage_density":       "Outrage",
    "modal_epistemic_ratio": "Epistemic ratio",
    "pron_inclusion_index":  "Inclusion index",
}

FEAT_COLS = list(FEATURES.keys())

HATECRIME_EXCLUDE_FY_START = 2019  # 2019/20 excluded

# Outlet display order and colours
OUTLET_ORDER = ["Daily Express", "Daily Star", "Daily Mirror",
                "The Independent", "guardian"]
OUTLET_COLOURS = {
    "Daily Express":    "#D62728",
    "Daily Star":       "#FF7F0E",
    "Daily Mirror":     "#9467BD",
    "The Independent":  "#2CA02C",
    "guardian":         "#1F77B4",
}
OUTLET_LABELS = {
    "Daily Express":    "Daily Express",
    "Daily Star":       "Daily Star",
    "Daily Mirror":     "Daily Mirror",
    "The Independent":  "The Independent",
    "guardian":         "The Guardian",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD EXTERNAL DATA (same logic as statistical_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_ons_annual(xlsx_path):
    """Load ONS net migration, one row per YE Jun → mapped to calendar year."""
    df = pd.read_excel(xlsx_path, sheet_name="1", header=None)
    start_row = None
    for i in range(len(df)):
        if str(df.iloc[i, 0]).strip() == "Net migration":
            start_row = i
            break

    records = []
    for i in range(start_row, len(df)):
        if str(df.iloc[i, 0]).strip() != "Net migration":
            break
        period = str(df.iloc[i, 1]).strip()
        m = re.match(r'YE\s+Jun\s+(\d{2})', period)
        if not m:
            continue
        year = 2000 + int(m.group(1))
        records.append({"year": year, "net_migration": int(df.iloc[i, 2])})

    return pd.DataFrame(records)


def load_hatecrime_annual(ods_path):
    """Load hate crime race data, one row per financial year."""
    df = pd.read_excel(ods_path, sheet_name="2", engine="odf", header=None)
    headers = df.iloc[6, 1:15].tolist()
    values = df.iloc[7, 1:15].tolist()

    records = []
    for h, v in zip(headers, values):
        label = str(h).strip()
        if str(v).strip() == "[x]":
            continue
        fy_match = re.match(r'(\d{4})/(\d{2})', label)
        if not fy_match:
            continue
        start_year = int(fy_match.group(1))
        if start_year == HATECRIME_EXCLUDE_FY_START:
            continue
        records.append({
            "fy_start_year": start_year,
            "fy_label": label,
            "race_hate_crimes": int(v),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# ANNUAL NLP AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def annual_nlp_means(features_df):
    """Compute annual means per outlet for all features."""
    records = []
    for outlet in features_df["outlet"].unique():
        sub = features_df[features_df["outlet"] == outlet]
        for year, group in sub.groupby("year"):
            row = {"outlet": outlet, "year": year, "n_articles": len(group)}
            for feat in FEAT_COLS:
                row[feat] = group[feat].mean()
            records.append(row)
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# LAG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def sig_stars(p):
    if p <= 0.001: return "***"
    if p <= 0.01:  return "**"
    if p <= 0.05:  return "*"
    return ""


def run_lag_analysis(annual_nlp, ons_annual, hc_annual, outlets):
    """
    Run four sets of lagged Spearman correlations:
      1. Migration_T → Rhetoric_{T+1}  (hypothesised causal direction)
      2. Rhetoric_T → Migration_{T+1}  (reverse check)
      3. Rhetoric_T → Hate_crime_{T+1} (hypothesised causal direction)
      4. Hate_crime_T → Rhetoric_{T+1} (reverse check)
    """
    results = []

    for outlet in outlets:
        nlp = annual_nlp[annual_nlp["outlet"] == outlet].copy()

        # ── Migration lags ──
        # Merge NLP and ONS on year
        merged_mig = nlp.merge(ons_annual, on="year", how="inner")

        for feat in FEAT_COLS:
            # 1. Migration_T → Rhetoric_{T+1}
            # Pair: ONS year T with NLP year T+1
            pairs = []
            for _, row in ons_annual.iterrows():
                t = row["year"]
                nlp_next = nlp[nlp["year"] == t + 1]
                if len(nlp_next) == 1 and not pd.isna(nlp_next[feat].values[0]):
                    pairs.append((row["net_migration"], nlp_next[feat].values[0]))

            if len(pairs) >= 5:
                x, y = zip(*pairs)
                rho, p = stats.spearmanr(x, y)
                results.append({
                    "outlet": outlet, "feature": feat,
                    "lag_type": "Migration_T → Rhetoric_T+1",
                    "direction": "hypothesised",
                    "n": len(pairs), "rho": round(rho, 4),
                    "p_value": round(p, 6), "sig": sig_stars(p),
                })

            # 2. Rhetoric_T → Migration_{T+1}  (reverse)
            pairs = []
            for _, row in nlp.iterrows():
                t = row["year"]
                ons_next = ons_annual[ons_annual["year"] == t + 1]
                if len(ons_next) == 1 and not pd.isna(row[feat]):
                    pairs.append((row[feat], ons_next["net_migration"].values[0]))

            if len(pairs) >= 5:
                x, y = zip(*pairs)
                rho, p = stats.spearmanr(x, y)
                results.append({
                    "outlet": outlet, "feature": feat,
                    "lag_type": "Rhetoric_T → Migration_T+1",
                    "direction": "reverse_check",
                    "n": len(pairs), "rho": round(rho, 4),
                    "p_value": round(p, 6), "sig": sig_stars(p),
                })

        # ── Hate crime lags ──
        for feat in FEAT_COLS:
            # 3. Rhetoric_T → Hate_crime_{T+1}
            # NLP year T → HC financial year starting T (i.e. T/T+1)
            # But T+1 means HC starting T+1, so NLP T → HC (T+1)/(T+2)
            # Wait — let's be precise:
            # HC "2012/13" starts Apr 2012, fy_start_year=2012
            # NLP_T → HC_{T+1} means NLP 2011 → HC 2012/13 (fy_start=2012)
            pairs = []
            for _, row in nlp.iterrows():
                t = row["year"]
                hc_next = hc_annual[hc_annual["fy_start_year"] == t + 1]
                if len(hc_next) == 1 and not pd.isna(row[feat]):
                    pairs.append((row[feat], hc_next["race_hate_crimes"].values[0]))

            if len(pairs) >= 5:
                x, y = zip(*pairs)
                rho, p = stats.spearmanr(x, y)
                results.append({
                    "outlet": outlet, "feature": feat,
                    "lag_type": "Rhetoric_T → HateCrime_T+1",
                    "direction": "hypothesised",
                    "n": len(pairs), "rho": round(rho, 4),
                    "p_value": round(p, 6), "sig": sig_stars(p),
                })

            # 4. Hate_crime_T → Rhetoric_{T+1}  (reverse)
            # HC fy_start T → NLP year T+1
            pairs = []
            for _, hc_row in hc_annual.iterrows():
                t = hc_row["fy_start_year"]
                nlp_next = nlp[nlp["year"] == t + 1]
                if len(nlp_next) == 1 and not pd.isna(nlp_next[feat].values[0]):
                    pairs.append((hc_row["race_hate_crimes"], nlp_next[feat].values[0]))

            if len(pairs) >= 5:
                x, y = zip(*pairs)
                rho, p = stats.spearmanr(x, y)
                results.append({
                    "outlet": outlet, "feature": feat,
                    "lag_type": "HateCrime_T → Rhetoric_T+1",
                    "direction": "reverse_check",
                    "n": len(pairs), "rho": round(rho, 4),
                    "p_value": round(p, 6), "sig": sig_stars(p),
                })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmaps(spearman_df, output_dir):
    """
    Two correlation heatmaps: features × outlets, one for migration,
    one for hate crime. Cells show rho with significance stars.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Spearman Correlations: NLP Features × Outcome Variables",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, outcome_substr, title in [
        (axes[0], "migration", "Net Migration (ONS quarterly)"),
        (axes[1], "hate",      "Race Hate Crime (Home Office annual)"),
    ]:
        subset = spearman_df[spearman_df["outcome"].str.contains(outcome_substr)]

        # Build matrix
        matrix = pd.DataFrame(index=FEAT_COLS,
                              columns=[o for o in OUTLET_ORDER if o in subset["outlet"].unique()])
        annot = matrix.copy()

        for _, r in subset.iterrows():
            if r["outlet"] in matrix.columns and r["feature"] in matrix.index:
                matrix.loc[r["feature"], r["outlet"]] = r["rho"]
                annot.loc[r["feature"], r["outlet"]] = f"{r['rho']:+.2f}{r['sig']}"

        matrix = matrix.astype(float)

        # Rename axes for display
        matrix.index = [FEATURES[f] for f in matrix.index]
        matrix.columns = [OUTLET_LABELS.get(c, c) for c in matrix.columns]
        annot.index = matrix.index
        annot.columns = matrix.columns

        sns.heatmap(matrix, ax=ax, annot=annot, fmt="",
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Spearman ρ"},
                    annot_kws={"fontsize": 9})
        ax.set_title(title, fontsize=11, pad=10)
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = Path(output_dir) / "heatmap_correlations.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_time_series(annual_means_df, output_dir):
    """
    Time series: one panel per feature, lines for each outlet.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Annual Mean NLP Features by Outlet (2010–2025)",
                 fontsize=14, fontweight="bold", y=1.01)

    axes_flat = axes.flatten()

    for i, feat in enumerate(FEAT_COLS):
        ax = axes_flat[i]

        for outlet in OUTLET_ORDER:
            sub = annual_means_df[annual_means_df["outlet"] == outlet].sort_values("year")
            if len(sub) == 0:
                continue
            ax.plot(sub["year"], sub[feat],
                    color=OUTLET_COLOURS[outlet],
                    label=OUTLET_LABELS[outlet],
                    linewidth=1.5, marker="o", markersize=3, alpha=0.85)

        ax.set_title(FEATURES[feat], fontsize=11, fontweight="bold")
        ax.set_xlabel("Year", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        if i == 0:
            ax.legend(fontsize=8, loc="best", framealpha=0.8)

    # Hide unused subplot
    if len(FEAT_COLS) < len(axes_flat):
        axes_flat[-1].set_visible(False)

    plt.tight_layout()
    path = Path(output_dir) / "timeseries_features.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_scatter_top(spearman_df, migration_agg, hatecrime_agg, output_dir, n_plots=6):
    """
    Scatter plots for the strongest correlations.
    """
    # Sort by |rho|, take top n
    top = spearman_df.reindex(
        spearman_df["rho"].abs().sort_values(ascending=False).index
    ).head(n_plots)

    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    fig.suptitle("Scatter Plots: Strongest Spearman Correlations",
                 fontsize=14, fontweight="bold", y=1.02)

    axes_flat = axes.flatten() if n_plots > 1 else [axes]

    for idx, (_, r) in enumerate(top.iterrows()):
        ax = axes_flat[idx]
        outlet = r["outlet"]
        feat = r["feature"]

        if "migration" in r["outcome"]:
            agg = migration_agg[migration_agg["outlet"] == outlet]
            outcome_col = "net_migration"
            outcome_label = "Net Migration"
        else:
            agg = hatecrime_agg[hatecrime_agg["outlet"] == outlet]
            outcome_col = "race_hate_crimes"
            outcome_label = "Race Hate Crimes"

        valid = agg.dropna(subset=[feat, outcome_col])

        ax.scatter(valid[outcome_col], valid[feat],
                   color=OUTLET_COLOURS.get(outlet, "#333"),
                   alpha=0.6, s=30, edgecolors="white", linewidth=0.5)

        # Trend line
        if len(valid) > 2:
            z = np.polyfit(valid[outcome_col], valid[feat], 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(valid[outcome_col].min(), valid[outcome_col].max(), 100)
            ax.plot(x_range, p_line(x_range), color=OUTLET_COLOURS.get(outlet, "#333"),
                    linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel(outcome_label, fontsize=9)
        ax.set_ylabel(FEATURES[feat], fontsize=9)
        ax.set_title(f"{OUTLET_LABELS.get(outlet, outlet)}\n"
                     f"ρ = {r['rho']:+.3f}{r['sig']}  (n={r['n']})",
                     fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)

    # Hide unused axes
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    path = Path(output_dir) / "scatter_top_correlations.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_lag_comparison(lag_df, output_dir):
    """
    Bar chart comparing hypothesised vs reverse lag correlations
    for each feature, grouped by outcome.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Lag-1 Analysis: Hypothesised vs Reverse Direction",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, outcome_str, title in [
        (axes[0], "Migration", "Migration → Rhetoric vs Rhetoric → Migration"),
        (axes[1], "HateCrime", "Rhetoric → Hate Crime vs Hate Crime → Rhetoric"),
    ]:
        hyp = lag_df[(lag_df["lag_type"].str.contains(outcome_str)) &
                     (lag_df["direction"] == "hypothesised")]
        rev = lag_df[(lag_df["lag_type"].str.contains(outcome_str)) &
                     (lag_df["direction"] == "reverse_check")]

        if len(hyp) == 0:
            continue

        # Average rho across outlets for each feature
        hyp_means = hyp.groupby("feature")["rho"].mean()
        rev_means = rev.groupby("feature")["rho"].mean()

        x = np.arange(len(FEAT_COLS))
        w = 0.35

        hyp_vals = [hyp_means.get(f, 0) for f in FEAT_COLS]
        rev_vals = [rev_means.get(f, 0) for f in FEAT_COLS]

        ax.bar(x - w/2, hyp_vals, w, label="Hypothesised direction",
               color="#2166AC", alpha=0.85)
        ax.bar(x + w/2, rev_vals, w, label="Reverse direction",
               color="#D6604D", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([FEATURES[f] for f in FEAT_COLS],
                           rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Mean Spearman ρ (across outlets)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = Path(output_dir) / "lag_comparison.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Lag Analysis & Visualisations — Immigration News Corpus"
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--ons", required=True)
    parser.add_argument("--hatecrime", required=True)
    parser.add_argument("--spearman", required=True,
                        help="Path to spearman_results.csv from statistical_analysis.py")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Lag Analysis & Visualisations")
    print("=" * 65)

    # ── Load data ──
    print("\n── Loading data ──")
    features = pd.read_csv(args.features, low_memory=False)
    ons_annual = load_ons_annual(args.ons)
    hc_annual = load_hatecrime_annual(args.hatecrime)
    spearman_df = pd.read_csv(args.spearman)

    print(f"  Features: {len(features)} articles")
    print(f"  ONS annual: {len(ons_annual)} years (YE Jun)")
    print(f"  Hate crime: {len(hc_annual)} years")

    outlets = [o for o in OUTLET_ORDER if o in features["outlet"].unique()]

    # ── Annual NLP means ──
    annual_nlp = annual_nlp_means(features)

    # ── Lag analysis ──
    print("\n── Lag-1 analysis ──")
    lag_df = run_lag_analysis(annual_nlp, ons_annual, hc_annual, outlets)
    lag_path = output_dir / "lag_results.csv"
    lag_df.to_csv(lag_path, index=False)
    print(f"  ✓ {lag_path} ({len(lag_df)} correlations)")

    # Print lag results
    for direction in ["hypothesised", "reverse_check"]:
        sub = lag_df[lag_df["direction"] == direction]
        label = "HYPOTHESISED" if direction == "hypothesised" else "REVERSE CHECK"
        print(f"\n  ── {label} ──")
        print(f"  {'outlet':20s}  {'feature':25s}  {'lag_type':35s}  "
              f"{'n':>3s}  {'rho':>7s}  {'p':>9s}")
        for _, r in sub.iterrows():
            print(f"  {r['outlet']:20s}  {r['feature']:25s}  {r['lag_type']:35s}  "
                  f"{r['n']:3d}  {r['rho']:+7.4f}  {r['p_value']:9.6f} {r['sig']}")

    # Compare hypothesised vs reverse
    hyp_sig = lag_df[(lag_df["direction"] == "hypothesised") & (lag_df["p_value"] <= 0.05)]
    rev_sig = lag_df[(lag_df["direction"] == "reverse_check") & (lag_df["p_value"] <= 0.05)]
    print(f"\n  Summary: {len(hyp_sig)} hypothesised significant, "
          f"{len(rev_sig)} reverse significant (out of "
          f"{len(lag_df[lag_df['direction']=='hypothesised'])} each)")

    # ── Visualisations ──
    print("\n── Visualisations ──")

    # Load aggregated data for scatter plots
    agg_mig_path = output_dir / "aggregated_migration.csv"
    agg_hc_path = output_dir / "aggregated_hatecrime.csv"

    if agg_mig_path.exists() and agg_hc_path.exists():
        migration_agg = pd.read_csv(agg_mig_path)
        hatecrime_agg = pd.read_csv(agg_hc_path)
    else:
        print("  WARNING: aggregated files not found — skipping scatter plots")
        print(f"  Run statistical_analysis.py first to generate them in {output_dir}")
        migration_agg = None
        hatecrime_agg = None

    plot_heatmaps(spearman_df, output_dir)
    plot_time_series(annual_nlp, output_dir)

    if migration_agg is not None:
        plot_scatter_top(spearman_df, migration_agg, hatecrime_agg, output_dir)

    plot_lag_comparison(lag_df, output_dir)

    print("\n" + "=" * 65)
    print("COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
