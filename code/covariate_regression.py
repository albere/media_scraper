"""
covariate_regression.py
Immigration News Corpus — Covariate Regression

Tests whether each linguistic feature's relationship with net migration
survives controlling for the ONS methodology discontinuity (the IPS → LTIM
change around 2020). For each outlet and each feature, fits:

    feature ~ net_migration + methodology_change

where methodology_change is a binary dummy (0 = IPS / pre-2020,
1 = LTIM / 2020+). The coefficient of interest is net_migration: if it
remains significant after including the dummy, the feature–migration
association is not an artefact of the measurement change.

Reports per model (JCADS standard): R², adjusted R², F, df, model p,
and the full coefficient table (beta, SE, t, p) for the intercept,
net_migration, and methodology_change.

Input:  aggregated_migration.csv from statistical_analysis.py
        (columns: outlet, net_migration, methodology, <feature columns>)
Output: regression_results.csv       — one row per outlet × feature model
        regression_coefficients.csv  — one row per coefficient (long form)

Usage:
    python covariate_regression.py \
        --aggregated results_allforces/aggregated_migration.csv \
        --output results_allforces/
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm


# Feature columns to model (must match those in aggregated_migration.csv)
FEATURES = {
    "sentiment_compound":    "Sentiment (VADER compound)",
    "dehum_density":         "Dehumanisation density (per 1k)",
    "outrage_density":       "Moral outrage density (per 1k)",
    "modal_epistemic_ratio": "Epistemic ratio",
    "pron_inclusion_index":  "Pronoun inclusion index",
}

# Methodology era split: LTIM (new method) applies from this year onward.
METHODOLOGY_CHANGE_YEAR = 2020

SIG_LEVELS = {0.001: "***", 0.01: "**", 0.05: "*", 1.0: ""}


def sig_stars(p):
    if p != p:
        return ""
    for threshold, stars in SIG_LEVELS.items():
        if p <= threshold:
            return stars
    return ""


def build_methodology_dummy(df):
    """
    Return a 0/1 series: 0 for IPS/pre-change, 1 for LTIM/post-change.
    Prefers an explicit 'methodology' label column; falls back to the
    quarter_end_year if the label is absent.
    """
    if "methodology" in df.columns:
        # 'LTIM' => 1, anything else (e.g. 'IPS') => 0
        return (df["methodology"].astype(str).str.upper() == "LTIM").astype(int)
    if "quarter_end_year" in df.columns:
        return (df["quarter_end_year"] >= METHODOLOGY_CHANGE_YEAR).astype(int)
    raise ValueError("Cannot build methodology dummy: need 'methodology' "
                     "or 'quarter_end_year' column.")


def fit_model(sub, feature):
    """
    Fit feature ~ net_migration + methodology_change for one outlet.
    Returns (summary_row, coef_rows) or (None, None) if not fittable.
    """
    data = sub.dropna(subset=[feature, "net_migration", "methodology_change"]).copy()
    n = len(data)
    if n < 6:
        return None, None

    y = data[feature].values
    X = data[["net_migration", "methodology_change"]].astype(float)
    X = sm.add_constant(X)  # intercept

    model = sm.OLS(y, X).fit()

    # Migration coefficient is the one of interest
    mig_beta = model.params["net_migration"]
    mig_p = model.pvalues["net_migration"]

    summary_row = {
        "outlet": sub["outlet"].iloc[0],
        "feature": feature,
        "feature_label": FEATURES.get(feature, feature),
        "n": n,
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "F": round(model.fvalue, 4),
        "df_model": int(model.df_model),
        "df_resid": int(model.df_resid),
        "model_p": round(model.f_pvalue, 6),
        "model_sig": sig_stars(model.f_pvalue),
        # the headline test: does migration survive the methodology control?
        "migration_beta": round(mig_beta, 6),
        "migration_p": round(mig_p, 6),
        "migration_sig": sig_stars(mig_p),
        "migration_survives": mig_p <= 0.05,
    }

    coef_rows = []
    for term in model.params.index:
        coef_rows.append({
            "outlet": sub["outlet"].iloc[0],
            "feature": feature,
            "term": term,
            "beta": round(model.params[term], 6),
            "std_err": round(model.bse[term], 6),
            "t": round(model.tvalues[term], 4),
            "p": round(model.pvalues[term], 6),
            "sig": sig_stars(model.pvalues[term]),
            "ci_low": round(model.conf_int().loc[term, 0], 6),
            "ci_high": round(model.conf_int().loc[term, 1], 6),
        })

    return summary_row, coef_rows


def main():
    parser = argparse.ArgumentParser(description="Covariate regression (methodology control)")
    parser.add_argument("--aggregated", required=True,
                        help="aggregated_migration.csv from statistical_analysis.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Covariate Regression — feature ~ net_migration + methodology_change")
    print("=" * 70)

    agg = pd.read_csv(args.aggregated)
    if "net_migration" not in agg.columns or "outlet" not in agg.columns:
        raise SystemExit("ERROR: aggregated file missing 'net_migration' or 'outlet'")

    agg["methodology_change"] = build_methodology_dummy(agg)

    # Rescale migration to per-100k persons so coefficients are legible
    # (raw net migration is in hundreds of thousands; per-unit betas would
    # otherwise be ~1e-6). This changes only the scale of the migration beta,
    # not its significance, t, or the model fit.
    agg["net_migration"] = agg["net_migration"] / 100_000.0
    print("\n  (net_migration rescaled to units of 100,000 persons for "
          "coefficient readability)")

    # report the split actually used
    split = agg.groupby("methodology_change").size()
    print(f"\n  Methodology dummy: 0 (IPS/pre-{METHODOLOGY_CHANGE_YEAR}) = "
          f"{split.get(0, 0)} rows, 1 (LTIM/post) = {split.get(1, 0)} rows")

    outlets = sorted(agg["outlet"].unique())
    available = [f for f in FEATURES if f in agg.columns]
    missing = set(FEATURES) - set(available)
    if missing:
        print(f"  (note: features not in file, skipped: {sorted(missing)})")

    summary_rows = []
    coef_rows = []
    for outlet in outlets:
        sub = agg[agg["outlet"] == outlet]
        for feat in available:
            s, c = fit_model(sub, feat)
            if s is not None:
                summary_rows.append(s)
                coef_rows.extend(c)

    summary = pd.DataFrame(summary_rows)
    coefs = pd.DataFrame(coef_rows)
    summary.to_csv(out / "regression_results.csv", index=False)
    coefs.to_csv(out / "regression_coefficients.csv", index=False)

    # ── Report: focus on dehum_density (the headline) and migration survival ──
    print("\n  ── Migration coefficient after methodology control ──")
    print(f"  {'outlet':12s} {'feature':24s} {'R2':>6s} {'F':>8s} "
          f"{'mig_beta':>11s} {'mig_p':>9s} {'survives':>9s}")
    print(f"  {'-'*12} {'-'*24} {'-'*6} {'-'*8} {'-'*11} {'-'*9} {'-'*9}")
    for _, r in summary.iterrows():
        surv = "yes" if r["migration_survives"] else "no"
        print(f"  {r['outlet']:12s} {r['feature_label']:24s} "
              f"{r['r_squared']:6.3f} {r['F']:8.2f} "
              f"{r['migration_beta']:+11.5f} {r['migration_p']:9.5f}{r['migration_sig']:<3s} {surv:>6s}")

    # Headline summary for dehum_density
    print("\n  ── Headline: dehum_density migration survival by outlet ──")
    dehum = summary[summary["feature"] == "dehum_density"]
    n_surv = dehum["migration_survives"].sum()
    print(f"  Migration coefficient survives methodology control in "
          f"{n_surv}/{len(dehum)} outlets:")
    for _, r in dehum.iterrows():
        eq = (f"dehum = {coefs[(coefs.outlet==r['outlet']) & (coefs.feature=='dehum_density') & (coefs.term=='const')]['beta'].iloc[0]:.3f} "
              f"+ {r['migration_beta']:.2e}·migration "
              f"+ {coefs[(coefs.outlet==r['outlet']) & (coefs.feature=='dehum_density') & (coefs.term=='methodology_change')]['beta'].iloc[0]:.3f}·methodology")
        print(f"    {r['outlet']:12s} R²={r['r_squared']:.3f}, "
              f"F({r['df_model']},{r['df_resid']})={r['F']:.2f}, p={r['model_p']:.5f}; "
              f"migration {'SURVIVES' if r['migration_survives'] else 'absorbed'} "
              f"(β={r['migration_beta']:.2e}, p={r['migration_p']:.4f})")

    print(f"\n  ✓ {out / 'regression_results.csv'}")
    print(f"  ✓ {out / 'regression_coefficients.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
