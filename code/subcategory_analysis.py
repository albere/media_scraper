"""
subcategory_analysis.py
Immigration News Corpus — Phase 1c: Dehumanisation Subcategory Analysis

Breaks the composite dehumanisation signal into its four subcategories and
tests which ones drive the migration relationship per outlet.

Subcategories (per-article density columns from nlp_pipeline.py):
    dehum_mass_metaphor_density   — flood/swarm/wave/invasion
    dehum_criminalisation_density — illegal/bogus/smuggling/gangs
    dehum_burden_density          — scrounger/handout/health tourism
    dehum_threat_density          — crisis/threat/uncontrolled

Reuses the rolling-12-month aggregation aligned to ONS quarters (identical
windows to statistical_analysis.py), then for each subcategory × outlet:
  - Spearman correlation vs net migration (raw)
  - Linear-detrended Spearman (residuals after removing linear time trend)
  - Benjamini-Hochberg FDR correction across the 16 raw tests

Outputs:
    subcategory_aggregated.csv   — quarterly subcategory means per outlet
    subcategory_spearman.csv     — raw + detrended rho/p, with FDR flags
    subcategory_composition.csv  — mean subcategory share per outlet (for plots)

Usage:
    python subcategory_analysis.py \
        --features all_features.csv \
        --ons ltimnov25.xlsx \
        --output results/
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


# ─── Subcategories under analysis ──────────────────────────────────────────
SUBCATEGORIES = {
    "dehum_mass_metaphor_density":   "Mass metaphor (flood/swarm/invasion)",
    "dehum_criminalisation_density": "Criminalisation (illegal/smuggling)",
    "dehum_burden_density":          "Burden (scrounger/handout)",
    "dehum_threat_density":          "Threat (crisis/uncontrolled)",
}

ONS_METHOD_CHANGE_YEAR = 2020
SIG_LEVELS = {0.001: "***", 0.01: "**", 0.05: "*", 1.0: ""}
MIN_ARTICLES = 10   # minimum articles in a window to compute a mean


# ─── ONS loader (same logic as statistical_analysis.py) ────────────────────
def load_ons_quarterly(xlsx_path):
    df = pd.read_excel(xlsx_path, sheet_name="1", header=None)

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
        records.append({
            "ons_period": period,
            "quarter_end_year": year,
            "quarter_end_month": month_map[month_str],
            "net_migration": int(value),
            "methodology": "LTIM" if year >= ONS_METHOD_CHANGE_YEAR else "IPS",
        })

    ons = pd.DataFrame(records)
    print(f"  ONS: {len(ons)} quarterly points loaded")
    return ons


# ─── Aggregation (rolling 12-month, matches main script) ───────────────────
def aggregate_subcategories(features_df, ons_df):
    features_df = features_df.copy()
    features_df["date_parsed"] = pd.to_datetime(
        features_df["date"].str[:10], errors="coerce"
    )
    features_df = features_df.dropna(subset=["date_parsed"])

    sub_cols = list(SUBCATEGORIES.keys())
    outlets = sorted(features_df["outlet"].unique())
    records = []

    for _, ons_row in ons_df.iterrows():
        end_year = ons_row["quarter_end_year"]
        end_month = ons_row["quarter_end_month"]
        window_end = pd.Timestamp(year=end_year, month=end_month, day=28)
        window_start = window_end - pd.DateOffset(months=12) + pd.DateOffset(days=1)

        for outlet in outlets:
            mask = (
                (features_df["outlet"] == outlet) &
                (features_df["date_parsed"] >= window_start) &
                (features_df["date_parsed"] <= window_end)
            )
            subset = features_df.loc[mask]
            if len(subset) < MIN_ARTICLES:
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
            for col in sub_cols:
                row[col] = subset[col].mean()
            records.append(row)

    result = pd.DataFrame(records)
    print(f"  Subcategory aggregation: {len(result)} rows "
          f"({len(result) // max(len(outlets), 1)} quarters × {len(outlets)} outlets)")
    return result


# ─── Detrending helper ─────────────────────────────────────────────────────
def linear_detrend(series, time_index):
    """Return residuals after removing the best-fit linear trend over time."""
    valid = series.notna()
    if valid.sum() < 3:
        return pd.Series(np.nan, index=series.index)
    x = np.asarray(time_index)[valid.values]
    y = series[valid].values
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    out = pd.Series(np.nan, index=series.index)
    out[valid] = resid
    return out


def sig_stars(p):
    for threshold, stars in SIG_LEVELS.items():
        if p <= threshold:
            return stars
    return ""


def benjamini_hochberg(pvals):
    """Return FDR-adjusted q-values for a list of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    cummin = 1.0
    # iterate from largest p to smallest
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        q = p[idx] * n / (rank + 1)
        cummin = min(cummin, q)
        ranked[idx] = cummin
    return ranked


# ─── Spearman: raw + detrended ─────────────────────────────────────────────
def run_subcategory_spearman(agg_df, outlets):
    sub_cols = list(SUBCATEGORIES.keys())
    rows = []

    for outlet in outlets:
        sub = agg_df[agg_df["outlet"] == outlet].copy()
        sub = sub.sort_values(["quarter_end_year", "quarter_end_month"])
        sub = sub.dropna(subset=["net_migration"])
        if len(sub) < 5:
            continue

        # time index for detrending (sequential quarter number)
        t = np.arange(len(sub))
        mig = sub["net_migration"].values
        mig_detr = linear_detrend(pd.Series(mig, index=sub.index),
                                  t).values

        for col in sub_cols:
            vals = sub[col]
            if vals.notna().sum() < 5:
                continue

            rho_raw, p_raw = stats.spearmanr(vals, sub["net_migration"])

            feat_detr = linear_detrend(vals.reset_index(drop=True),
                                       t).values
            mask = ~np.isnan(feat_detr) & ~np.isnan(mig_detr)
            if mask.sum() >= 5:
                rho_dt, p_dt = stats.spearmanr(feat_detr[mask], mig_detr[mask])
            else:
                rho_dt, p_dt = np.nan, np.nan

            rows.append({
                "outlet": outlet,
                "subcategory": col,
                "subcategory_label": SUBCATEGORIES[col],
                "n": len(sub),
                "rho_raw": round(rho_raw, 4),
                "p_raw": round(p_raw, 6),
                "sig_raw": sig_stars(p_raw),
                "rho_detrended": round(rho_dt, 4) if rho_dt == rho_dt else np.nan,
                "p_detrended": round(p_dt, 6) if p_dt == p_dt else np.nan,
                "sig_detrended": sig_stars(p_dt) if p_dt == p_dt else "",
            })

    df = pd.DataFrame(rows)

    # FDR across the raw tests (the 16-test family)
    if len(df) > 0:
        df["q_raw_fdr"] = benjamini_hochberg(df["p_raw"].values).round(6)
        df["survives_fdr"] = df["q_raw_fdr"] <= 0.05

    return df


# ─── Composition table (for stacked-area plots) ────────────────────────────
def composition_table(agg_df, outlets):
    sub_cols = list(SUBCATEGORIES.keys())
    rows = []
    for outlet in outlets:
        sub = agg_df[agg_df["outlet"] == outlet]
        means = {c: sub[c].mean() for c in sub_cols}
        total = sum(means.values())
        for c in sub_cols:
            rows.append({
                "outlet": outlet,
                "subcategory": c,
                "subcategory_label": SUBCATEGORIES[c],
                "mean_density": round(means[c], 4),
                "share_of_dehum": round(means[c] / total, 4) if total > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Dehumanisation subcategory analysis")
    parser.add_argument("--features", required=True)
    parser.add_argument("--ons", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Dehumanisation Subcategory Analysis (Phase 1c)")
    print("=" * 65)

    print("\n── Loading ──")
    features = pd.read_csv(args.features, low_memory=False)
    print(f"  Features: {len(features):,} articles, "
          f"{features['outlet'].nunique()} outlets")
    missing = set(SUBCATEGORIES) - set(features.columns)
    if missing:
        raise SystemExit(f"ERROR: missing subcategory columns: {missing}")
    ons = load_ons_quarterly(args.ons)
    outlets = sorted(features["outlet"].unique())

    print("\n── Aggregating ──")
    agg = aggregate_subcategories(features, ons)
    agg.to_csv(out / "subcategory_aggregated.csv", index=False)
    print(f"  ✓ {out / 'subcategory_aggregated.csv'}")

    print("\n── Spearman (raw + detrended, FDR) ──")
    sp = run_subcategory_spearman(agg, outlets)
    sp.to_csv(out / "subcategory_spearman.csv", index=False)
    print(f"  ✓ {out / 'subcategory_spearman.csv'} ({len(sp)} tests)")

    print("\n  ── RESULTS: subcategory × migration ──")
    print(f"  {'outlet':12s} {'subcategory':32s} {'raw':>8s} {'detr':>8s} {'FDR':>5s}")
    print(f"  {'-'*12} {'-'*32} {'-'*8} {'-'*8} {'-'*5}")
    for _, r in sp.iterrows():
        fdr = "yes" if r.get("survives_fdr", False) else "no"
        detr = f"{r['rho_detrended']:+.3f}" if pd.notna(r['rho_detrended']) else "  n/a"
        print(f"  {r['outlet']:12s} {r['subcategory_label']:32s} "
              f"{r['rho_raw']:+.3f}{r['sig_raw']:<3s} {detr}{r['sig_detrended']:<3s} {fdr:>5s}")

    print("\n── Composition (mean share per outlet) ──")
    comp = composition_table(agg, outlets)
    comp.to_csv(out / "subcategory_composition.csv", index=False)
    print(f"  ✓ {out / 'subcategory_composition.csv'}")
    for outlet in outlets:
        sub = comp[comp["outlet"] == outlet].sort_values("share_of_dehum", ascending=False)
        top = sub.iloc[0]
        print(f"  {outlet:12s} dominant: {top['subcategory_label']} "
              f"({top['share_of_dehum']:.0%})")

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
