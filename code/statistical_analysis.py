"""
statistical_analysis.py
Immigration News Corpus — Aggregation & Statistical Analysis

Aggregates per-article NLP features to:
  1. Rolling 12-month means aligned to ONS quarterly net migration
  2. Financial-year means aligned to Home Office hate crime data

Then runs:
  - Spearman correlations (5 features × 2 outcomes × 5 outlets = 50 tests)
  - ANOVA on tertile groups
  - Descriptive statistics per outlet

Usage:
    python statistical_analysis.py \
        --features all_features.csv \
        --ons ltimnov25.xlsx \
        --hatecrime hate-crime-england-and-wales-2024-to-2025-data-tables.ods \
        --output results/
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# The 5 NLP features used in correlations (primary metrics only)
FEATURES = {
    "sentiment_compound":    "Sentiment (VADER compound)",
    "dehum_density":         "Dehumanisation density (per 1k)",
    "outrage_density":       "Moral outrage density (per 1k)",
    "modal_epistemic_ratio": "Epistemic ratio",
    "pron_inclusion_index":  "Pronoun inclusion index",
}

# ONS methodology change: IPS (pre-2020) → LTIM admin-based (2020+)
# Quarter where LTIM starts: YE Jun 2020 onwards
ONS_METHOD_CHANGE_YEAR = 2020

# Hate crime: exclude 2019/20 (COVID disruption)
HATECRIME_EXCLUDE_YEAR = "2019/20"

# Significance thresholds for display
SIG_LEVELS = {0.001: "***", 0.01: "**", 0.05: "*", 1.0: ""}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD EXTERNAL DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_ons_quarterly(xlsx_path):
    """
    Load ONS LTIM quarterly net migration from the official spreadsheet.
    Returns DataFrame with columns: quarter_end_month, quarter_end_year,
    net_migration, ons_period, methodology
    """
    df = pd.read_excel(xlsx_path, sheet_name="1", header=None)

    # Net migration starts at the row where column 0 == "Net migration"
    start_row = None
    for i in range(len(df)):
        if str(df.iloc[i, 0]).strip() == "Net migration":
            start_row = i
            break

    if start_row is None:
        raise ValueError("Could not find 'Net migration' section in ONS spreadsheet")

    records = []
    month_map = {"Mar": 3, "Jun": 6, "Sep": 9, "Dec": 12}

    for i in range(start_row, len(df)):
        flow = str(df.iloc[i, 0]).strip()
        if flow != "Net migration":
            break

        period = str(df.iloc[i, 1]).strip()
        value = df.iloc[i, 2]

        m = re.match(r'YE\s+(Mar|Jun|Sep|Dec)\s+(\d{2})', period)
        if not m:
            continue

        month_str, yy = m.groups()
        year = 2000 + int(yy)
        month = month_map[month_str]

        methodology = "LTIM" if year >= ONS_METHOD_CHANGE_YEAR else "IPS"

        records.append({
            "ons_period": period,
            "quarter_end_year": year,
            "quarter_end_month": month,
            "net_migration": int(value),
            "methodology": methodology,
        })

    ons = pd.DataFrame(records)
    print(f"  ONS: {len(ons)} quarterly points loaded "
          f"(YE {ons.iloc[0]['ons_period']} to YE {ons.iloc[-1]['ons_period']})")
    return ons


def load_hatecrime(ods_path):
    """
    Load Home Office racial hate crime annual data from ODS file.
    Returns DataFrame with columns: fy_label, fy_start_year, race_hate_crimes
    """
    df = pd.read_excel(ods_path, sheet_name="2", engine="odf", header=None)

    # Row 6 has year headers, row 7 has Race values
    headers = df.iloc[6, 1:15].tolist()
    values = df.iloc[7, 1:15].tolist()

    records = []
    for h, v in zip(headers, values):
        label = str(h).strip()

        # Skip excluded year
        if HATECRIME_EXCLUDE_YEAR in label:
            continue

        # Skip [x] values
        if str(v).strip() == "[x]":
            continue

        # Parse "2011/12" → start year 2011
        fy_match = re.match(r'(\d{4})/(\d{2})', label)
        if not fy_match:
            continue

        start_year = int(fy_match.group(1))

        records.append({
            "fy_label": label,
            "fy_start_year": start_year,
            "race_hate_crimes": int(v),
        })

    hc = pd.DataFrame(records)
    print(f"  Hate crime: {len(hc)} annual points loaded "
          f"({hc['fy_label'].iloc[0]} to {hc['fy_label'].iloc[-1]})")
    return hc


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AGGREGATE NLP FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_rolling_12m(features_df, ons_df):
    """
    For each ONS quarter and each outlet, compute the mean of each NLP
    feature over the matching 12-month window.

    E.g. ONS "YE Sep 2016" → NLP articles from Oct 2015 to Sep 2016.

    Returns DataFrame with one row per (outlet, ons_quarter) pair.
    """
    # Parse dates
    features_df = features_df.copy()
    features_df["date_parsed"] = pd.to_datetime(
        features_df["date"].str[:10], errors="coerce"
    )
    features_df = features_df.dropna(subset=["date_parsed"])

    feat_cols = list(FEATURES.keys())
    records = []

    outlets = sorted(features_df["outlet"].unique())

    for _, ons_row in ons_df.iterrows():
        end_year = ons_row["quarter_end_year"]
        end_month = ons_row["quarter_end_month"]

        # 12-month window ending at this quarter
        window_end = pd.Timestamp(year=end_year, month=end_month, day=28)
        window_start = window_end - pd.DateOffset(months=12) + pd.DateOffset(days=1)

        for outlet in outlets:
            mask = (
                (features_df["outlet"] == outlet) &
                (features_df["date_parsed"] >= window_start) &
                (features_df["date_parsed"] <= window_end)
            )
            subset = features_df.loc[mask]

            if len(subset) < 10:  # minimum articles for a reliable mean
                continue

            row = {
                "outlet": outlet,
                "ons_period": ons_row["ons_period"],
                "quarter_end_year": end_year,
                "quarter_end_month": end_month,
                "net_migration": ons_row["net_migration"],
                "methodology": ons_row["methodology"],
                "n_articles": len(subset),
            }

            for feat in feat_cols:
                row[feat] = subset[feat].mean()

            records.append(row)

    result = pd.DataFrame(records)
    print(f"  Rolling 12-month aggregation: {len(result)} rows "
          f"({len(result) // len(outlets)} quarters × {len(outlets)} outlets)")
    return result


def aggregate_financial_year(features_df, hc_df):
    """
    For each hate crime financial year and each outlet, compute the mean
    of each NLP feature over Apr–Mar.

    E.g. hate crime "2021/22" → NLP articles from Apr 2021 to Mar 2022.

    Returns DataFrame with one row per (outlet, financial_year) pair.
    """
    features_df = features_df.copy()
    features_df["date_parsed"] = pd.to_datetime(
        features_df["date"].str[:10], errors="coerce"
    )
    features_df = features_df.dropna(subset=["date_parsed"])

    feat_cols = list(FEATURES.keys())
    records = []

    outlets = sorted(features_df["outlet"].unique())

    for _, hc_row in hc_df.iterrows():
        start_year = hc_row["fy_start_year"]

        # Financial year: Apr of start_year to Mar of start_year+1
        window_start = pd.Timestamp(year=start_year, month=4, day=1)
        window_end = pd.Timestamp(year=start_year + 1, month=3, day=31)

        for outlet in outlets:
            mask = (
                (features_df["outlet"] == outlet) &
                (features_df["date_parsed"] >= window_start) &
                (features_df["date_parsed"] <= window_end)
            )
            subset = features_df.loc[mask]

            if len(subset) < 10:
                continue

            row = {
                "outlet": outlet,
                "fy_label": hc_row["fy_label"],
                "fy_start_year": start_year,
                "race_hate_crimes": hc_row["race_hate_crimes"],
                "n_articles": len(subset),
            }

            for feat in feat_cols:
                row[feat] = subset[feat].mean()

            records.append(row)

    result = pd.DataFrame(records)
    print(f"  Financial-year aggregation: {len(result)} rows "
          f"({len(result) // len(outlets)} years × {len(outlets)} outlets)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SPEARMAN CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def sig_stars(p):
    for threshold, stars in SIG_LEVELS.items():
        if p <= threshold:
            return stars
    return ""


def run_spearman(agg_df, outcome_col, outcome_label, outlets):
    """
    Run Spearman correlations: each feature × each outlet against outcome.
    Returns list of result dicts.
    """
    feat_cols = list(FEATURES.keys())
    results = []

    for outlet in outlets:
        sub = agg_df[agg_df["outlet"] == outlet].dropna(subset=[outcome_col])

        for feat in feat_cols:
            valid = sub.dropna(subset=[feat])
            n = len(valid)

            if n < 5:
                continue

            rho, p = stats.spearmanr(valid[feat], valid[outcome_col])

            results.append({
                "outlet": outlet,
                "feature": feat,
                "feature_label": FEATURES[feat],
                "outcome": outcome_label,
                "n": n,
                "rho": round(rho, 4),
                "p_value": round(p, 6),
                "sig": sig_stars(p),
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANOVA — TERTILE GROUP COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_anova(agg_df, outcome_col, outcome_label, outlets):
    """
    Split outcome into tertile groups (low/mid/high), compare NLP feature
    means across groups using one-way ANOVA.
    """
    feat_cols = list(FEATURES.keys())
    results = []

    for outlet in outlets:
        sub = agg_df[agg_df["outlet"] == outlet].dropna(subset=[outcome_col]).copy()

        if len(sub) < 9:  # need at least 3 per tertile
            continue

        # Create tertile groups
        sub["tertile"] = pd.qcut(sub[outcome_col], q=3, labels=["low", "mid", "high"])

        for feat in feat_cols:
            valid = sub.dropna(subset=[feat])

            groups = [valid[valid["tertile"] == t][feat].values for t in ["low", "mid", "high"]]
            groups = [g for g in groups if len(g) >= 2]

            if len(groups) < 3:
                continue

            f_stat, p = stats.f_oneway(*groups)

            group_means = {
                t: valid[valid["tertile"] == t][feat].mean()
                for t in ["low", "mid", "high"]
            }

            results.append({
                "outlet": outlet,
                "feature": feat,
                "feature_label": FEATURES[feat],
                "outcome": outcome_label,
                "n": len(valid),
                "F": round(f_stat, 4),
                "p_value": round(p, 6),
                "sig": sig_stars(p),
                "mean_low": round(group_means.get("low", float("nan")), 4),
                "mean_mid": round(group_means.get("mid", float("nan")), 4),
                "mean_high": round(group_means.get("high", float("nan")), 4),
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def descriptive_stats(features_df, output_dir):
    """
    Per-outlet descriptive statistics for all features.
    """
    feat_cols = list(FEATURES.keys())
    outlets = sorted(features_df["outlet"].unique())

    records = []
    for outlet in outlets:
        sub = features_df[features_df["outlet"] == outlet]
        for feat in feat_cols:
            vals = sub[feat].dropna()
            records.append({
                "outlet": outlet,
                "feature": feat,
                "feature_label": FEATURES[feat],
                "n": len(vals),
                "mean": round(vals.mean(), 4),
                "std": round(vals.std(), 4),
                "median": round(vals.median(), 4),
                "min": round(vals.min(), 4),
                "max": round(vals.max(), 4),
                "q25": round(vals.quantile(0.25), 4),
                "q75": round(vals.quantile(0.75), 4),
            })

    desc_df = pd.DataFrame(records)
    path = Path(output_dir) / "descriptive_stats.csv"
    desc_df.to_csv(path, index=False)
    print(f"  ✓ {path}")
    return desc_df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANNUAL DESCRIPTIVE TABLE (for reporting)
# ═══════════════════════════════════════════════════════════════════════════════

def annual_means_table(features_df, output_dir):
    """
    Annual means per outlet for all features — useful for time series
    plots and supplementary tables.
    """
    feat_cols = list(FEATURES.keys())
    outlets = sorted(features_df["outlet"].unique())

    records = []
    for outlet in outlets:
        sub = features_df[features_df["outlet"] == outlet]
        annual = sub.groupby("year")
        for year, group in annual:
            row = {"outlet": outlet, "year": year, "n_articles": len(group)}
            for feat in feat_cols:
                row[feat] = round(group[feat].mean(), 4)
            records.append(row)

    annual_df = pd.DataFrame(records)
    path = Path(output_dir) / "annual_means.csv"
    annual_df.to_csv(path, index=False)
    print(f"  ✓ {path}")
    return annual_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Statistical Analysis — Immigration News Corpus"
    )
    parser.add_argument("--features", required=True,
                        help="Path to all_features.csv")
    parser.add_argument("--ons", required=True,
                        help="Path to ONS LTIM xlsx file")
    parser.add_argument("--hatecrime", required=True,
                        help="Path to Home Office hate crime ODS file")
    parser.add_argument("--output", required=True,
                        help="Output directory for results")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Statistical Analysis Pipeline")
    print("=" * 65)

    # ── Load data ──
    print("\n── Loading data ──")
    features = pd.read_csv(args.features, low_memory=False)
    print(f"  Features: {len(features)} articles, "
          f"{len(features['outlet'].unique())} outlets")

    ons = load_ons_quarterly(args.ons)
    hc = load_hatecrime(args.hatecrime)

    outlets = sorted(features["outlet"].unique())
    print(f"  Outlets: {outlets}")

    # ── Aggregate ──
    print("\n── Aggregating NLP features ──")
    migration_agg = aggregate_rolling_12m(features, ons)
    hatecrime_agg = aggregate_financial_year(features, hc)

    # Save aggregated data
    migration_agg.to_csv(output_dir / "aggregated_migration.csv", index=False)
    hatecrime_agg.to_csv(output_dir / "aggregated_hatecrime.csv", index=False)
    print(f"  ✓ {output_dir / 'aggregated_migration.csv'}")
    print(f"  ✓ {output_dir / 'aggregated_hatecrime.csv'}")

    # ── Spearman correlations ──
    print("\n── Spearman correlations ──")
    spearman_results = []

    migration_spearman = run_spearman(
        migration_agg, "net_migration", "Net migration (ONS quarterly)", outlets
    )
    spearman_results.extend(migration_spearman)

    hatecrime_spearman = run_spearman(
        hatecrime_agg, "race_hate_crimes", "Race hate crimes (Home Office annual)", outlets
    )
    spearman_results.extend(hatecrime_spearman)

    spearman_df = pd.DataFrame(spearman_results)
    spearman_path = output_dir / "spearman_results.csv"
    spearman_df.to_csv(spearman_path, index=False)
    print(f"  ✓ {spearman_path} ({len(spearman_df)} correlations)")

    # Print summary
    print("\n  ── MIGRATION CORRELATIONS ──")
    print(f"  {'outlet':20s}  {'feature':30s}  {'n':>3s}  {'rho':>7s}  {'p':>9s}")
    print(f"  {'-'*20}  {'-'*30}  {'-'*3}  {'-'*7}  {'-'*9}")
    for _, r in spearman_df[spearman_df["outcome"].str.contains("migration")].iterrows():
        print(f"  {r['outlet']:20s}  {r['feature']:30s}  {r['n']:3d}  "
              f"{r['rho']:+7.4f}  {r['p_value']:9.6f} {r['sig']}")

    print("\n  ── HATE CRIME CORRELATIONS ──")
    print(f"  {'outlet':20s}  {'feature':30s}  {'n':>3s}  {'rho':>7s}  {'p':>9s}")
    print(f"  {'-'*20}  {'-'*30}  {'-'*3}  {'-'*7}  {'-'*9}")
    for _, r in spearman_df[spearman_df["outcome"].str.contains("hate")].iterrows():
        print(f"  {r['outlet']:20s}  {r['feature']:30s}  {r['n']:3d}  "
              f"{r['rho']:+7.4f}  {r['p_value']:9.6f} {r['sig']}")

    # ── ANOVA ──
    print("\n── ANOVA (tertile group comparisons) ──")
    anova_results = []

    migration_anova = run_anova(
        migration_agg, "net_migration", "Net migration", outlets
    )
    anova_results.extend(migration_anova)

    hatecrime_anova = run_anova(
        hatecrime_agg, "race_hate_crimes", "Race hate crimes", outlets
    )
    anova_results.extend(hatecrime_anova)

    anova_df = pd.DataFrame(anova_results)
    anova_path = output_dir / "anova_results.csv"
    anova_df.to_csv(anova_path, index=False)
    print(f"  ✓ {anova_path} ({len(anova_df)} tests)")

    # Print significant ANOVA results
    sig_anova = anova_df[anova_df["p_value"] <= 0.05]
    if len(sig_anova) > 0:
        print(f"\n  Significant ANOVA results (p <= 0.05): {len(sig_anova)}")
        print(f"  {'outlet':20s}  {'feature':25s}  {'outcome':15s}  "
              f"{'F':>7s}  {'p':>9s}  {'low':>7s}  {'mid':>7s}  {'high':>7s}")
        for _, r in sig_anova.iterrows():
            print(f"  {r['outlet']:20s}  {r['feature']:25s}  {r['outcome']:15s}  "
                  f"{r['F']:7.2f}  {r['p_value']:9.6f} {r['sig']}  "
                  f"{r['mean_low']:7.4f}  {r['mean_mid']:7.4f}  {r['mean_high']:7.4f}")
    else:
        print("  No significant ANOVA results at p <= 0.05")

    # ── Descriptive stats ──
    print("\n── Descriptive statistics ──")
    descriptive_stats(features, output_dir)
    annual_means_table(features, output_dir)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"\nOutput files in {output_dir}/:")
    print(f"  aggregated_migration.csv   — rolling 12-month NLP means × ONS quarters")
    print(f"  aggregated_hatecrime.csv   — financial-year NLP means × hate crime")
    print(f"  spearman_results.csv       — {len(spearman_df)} Spearman correlations")
    print(f"  anova_results.csv          — {len(anova_df)} ANOVA tests")
    print(f"  descriptive_stats.csv      — per-outlet feature summaries")
    print(f"  annual_means.csv           — annual means for time series plots")

    # Count significant results
    sig_spearman = spearman_df[spearman_df["p_value"] <= 0.05]
    print(f"\n  Significant Spearman (p <= 0.05): {len(sig_spearman)}/{len(spearman_df)}")
    print(f"  Significant ANOVA (p <= 0.05):    {len(sig_anova)}/{len(anova_df)}")

    # Strongest correlations
    if len(spearman_df) > 0:
        print("\n  ── TOP 10 STRONGEST CORRELATIONS (by |rho|) ──")
        top = spearman_df.reindex(
            spearman_df["rho"].abs().sort_values(ascending=False).index
        ).head(10)
        for _, r in top.iterrows():
            print(f"    {r['outlet']:20s}  {r['feature']:25s}  × {r['outcome']:20s}  "
                  f"rho={r['rho']:+.4f}  p={r['p_value']:.6f} {r['sig']}")


if __name__ == "__main__":
    main()
