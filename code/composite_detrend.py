"""
composite_detrend.py
Immigration News Corpus — Phase 1b: Composite Detrending

Confirms whether the headline finding — dehumanisation density tracks net
migration — survives removal of the shared upward time trend. Both series
rise over 2010–2025, so a raw correlation could reflect nothing more than two
quantities trending together. This script tests three things per outlet:

  1. Raw Spearman              dehum_density vs net_migration
  2. Linear-detrended Spearman residuals after removing each series' linear
                               trend (does the relationship hold within-period?)
  3. First-differenced Spearman Δdehum vs Δmigration (does quarter-to-quarter
                               *change* co-move? — a stricter test)

Also runs Augmented Dickey-Fuller tests for stationarity (context for why
detrending is needed) when statsmodels is available.

Input:  an aggregated_migration.csv produced by statistical_analysis.py
        (columns: outlet, ons_period, quarter_end_year, quarter_end_month,
         net_migration, dehum_density, ...)
Output: composite_detrend_results.csv

Usage:
    python composite_detrend.py \
        --aggregated results_allforces/aggregated_migration.csv \
        --output results_allforces/
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

try:
    from statsmodels.tsa.stattools import adfuller
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

FEATURE = "dehum_density"
OUTCOME = "net_migration"
SIG_LEVELS = {0.001: "***", 0.01: "**", 0.05: "*", 1.0: ""}


def sig_stars(p):
    if p != p:  # NaN
        return ""
    for threshold, stars in SIG_LEVELS.items():
        if p <= threshold:
            return stars
    return ""


def linear_detrend(y, x):
    """Residuals after removing best-fit linear trend."""
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)


def benjamini_hochberg(pvals):
    """FDR-adjusted q-values (Benjamini-Hochberg). NaNs pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[mask]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = np.empty(n, dtype=float)
    cummin = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = pv[idx] * n / (rank + 1)
        cummin = min(cummin, val)
        ranked[idx] = cummin
    q[mask] = ranked
    return q


def adf_pvalue(series):
    """ADF test p-value; low p => stationary. None if unavailable/too short."""
    if not HAVE_SM or len(series) < 8:
        return None
    try:
        return adfuller(series, autolag="AIC")[1]
    except Exception:
        return None


def analyse_outlet(sub):
    """Run raw / detrended / first-differenced correlations for one outlet."""
    sub = sub.sort_values(["quarter_end_year", "quarter_end_month"]).copy()
    sub = sub.dropna(subset=[FEATURE, OUTCOME])
    n = len(sub)
    if n < 5:
        return None

    feat = sub[FEATURE].values.astype(float)
    mig = sub[OUTCOME].values.astype(float)
    t = np.arange(n)

    # 1. Raw
    rho_raw, p_raw = stats.spearmanr(feat, mig)

    # 2. Linear-detrended
    feat_dt = linear_detrend(feat, t)
    mig_dt = linear_detrend(mig, t)
    rho_dt, p_dt = stats.spearmanr(feat_dt, mig_dt)

    # 3. First-differenced
    if n >= 6:
        feat_d = np.diff(feat)
        mig_d = np.diff(mig)
        rho_fd, p_fd = stats.spearmanr(feat_d, mig_d)
    else:
        rho_fd, p_fd = np.nan, np.nan

    return {
        "n": n,
        "rho_raw": round(rho_raw, 4),
        "p_raw": round(p_raw, 6),
        "sig_raw": sig_stars(p_raw),
        "rho_detrended": round(rho_dt, 4),
        "p_detrended": round(p_dt, 6),
        "sig_detrended": sig_stars(p_dt),
        "rho_first_diff": round(rho_fd, 4) if rho_fd == rho_fd else np.nan,
        "p_first_diff": round(p_fd, 6) if p_fd == p_fd else np.nan,
        "sig_first_diff": sig_stars(p_fd),
        "adf_p_dehum": adf_pvalue(feat),
        "adf_p_migration": adf_pvalue(mig),
    }


def main():
    parser = argparse.ArgumentParser(description="Composite detrending (Phase 1b)")
    parser.add_argument("--aggregated", required=True,
                        help="aggregated_migration.csv from statistical_analysis.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Composite Detrending — dehum_density × net_migration (Phase 1b)")
    print("=" * 70)

    agg = pd.read_csv(args.aggregated)
    for col in (FEATURE, OUTCOME, "outlet"):
        if col not in agg.columns:
            raise SystemExit(f"ERROR: column '{col}' not in {args.aggregated}")

    if not HAVE_SM:
        print("\n  (statsmodels not installed — ADF stationarity tests skipped)")

    outlets = sorted(agg["outlet"].unique())
    rows = []
    for outlet in outlets:
        res = analyse_outlet(agg[agg["outlet"] == outlet])
        if res is None:
            print(f"  {outlet}: too few points, skipped")
            continue
        res = {"outlet": outlet, **res}
        rows.append(res)

    df = pd.DataFrame(rows)

    # FDR correction across the 4-outlet composite test families
    if len(df) > 0:
        df["q_raw_fdr"] = benjamini_hochberg(df["p_raw"].values).round(6)
        df["raw_survives_fdr"] = df["q_raw_fdr"] <= 0.05
        df["q_detrended_fdr"] = benjamini_hochberg(df["p_detrended"].values).round(6)
        df["detrended_survives_fdr"] = df["q_detrended_fdr"] <= 0.05

    df.to_csv(out / "composite_detrend_results.csv", index=False)

    # ── Report ──
    print(f"\n  {'outlet':12s} {'n':>3s} {'raw rho':>9s} {'raw p':>9s} "
          f"{'raw q(FDR)':>10s} {'detr rho':>9s} {'detr p':>9s} {'detr q(FDR)':>11s} {'1st-diff':>9s}")
    print(f"  {'-'*12} {'-'*3} {'-'*9} {'-'*9} {'-'*10} {'-'*9} {'-'*9} {'-'*11} {'-'*9}")
    for _, r in df.iterrows():
        fd = (f"{r['rho_first_diff']:+.3f}{r['sig_first_diff']}"
              if pd.notna(r['rho_first_diff']) else "n/a")
        print(f"  {r['outlet']:12s} {r['n']:3d} "
              f"{r['rho_raw']:+8.3f} {r['p_raw']:9.5f} {r['q_raw_fdr']:10.5f} "
              f"{r['rho_detrended']:+8.3f} {r['p_detrended']:9.5f} {r['q_detrended_fdr']:11.5f} "
              f"{fd:>9s}")

    if HAVE_SM:
        print(f"\n  ── ADF stationarity (p < 0.05 = stationary; high p = has unit root) ──")
        for _, r in df.iterrows():
            dp = r['adf_p_dehum']; mp = r['adf_p_migration']
            ds = f"{dp:.3f}" if dp is not None else "n/a"
            ms = f"{mp:.3f}" if mp is not None else "n/a"
            print(f"  {r['outlet']:12s} dehum ADF p={ds:>6s}   migration ADF p={ms:>6s}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  - Detrended survives  => relationship is not just shared trend (GOOD)")
    print("  - First-diff lost     => association is structural, not reactive")
    print("                           quarter-to-quarter (a limitation to report)")
    print("=" * 70)
    print(f"\n✓ {out / 'composite_detrend_results.csv'}")


if __name__ == "__main__":
    main()
