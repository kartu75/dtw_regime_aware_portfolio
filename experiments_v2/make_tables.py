# -*- coding: utf-8 -*-
"""Summary tables for the objectives runs.
Usage: python make_tables.py [prefix]      (default prefix "full": reads results/objectives/full_*.pkl)
Writes results/objectives/<prefix>_summary.txt and <prefix>_summary.tsv"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import dtwlib as L
import objectives as O

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "objectives")
ROOT = os.path.dirname(HERE)
prefix = sys.argv[1] if len(sys.argv) > 1 else "full"
DORDER = ["2p_N50_A", "2p_N50_B", "2p_BANK_A", "2p_BANK_B", "single_N50", "exp_N50", "exp_BANK"]
NAMES = {"2p_N50_A": "Two-period A, Nifty 50", "2p_N50_B": "Two-period B (strict), Nifty 50",
         "2p_BANK_A": "Two-period A, Nifty Bank", "2p_BANK_B": "Two-period B (strict), Nifty Bank",
         "single_N50": "Single split, Nifty 50", "exp_N50": "Expanding window, Nifty 50",
         "exp_BANK": "Expanding window, Nifty Bank"}

res = {os.path.basename(p)[len(prefix) + 1:-4]: L.load_pickle(p)
       for p in glob.glob(os.path.join(OUT, f"{prefix}_*.pkl"))}
designs = [d for d in DORDER if d in res]
met = pd.concat([res[d]["metrics"] for d in designs], ignore_index=True)
reg = pd.concat([res[d]["regression"] for d in designs], ignore_index=True)
ws = pd.concat([res[d]["weight_stats"].assign(design=d) for d in designs], ignore_index=True)
bet = pd.concat([res[d]["betas"].assign(design=d) for d in designs], ignore_index=True)
methods = sorted(set(met.method))

# ---- primary family corrections (12 tests: 4 two-period 1-day designs x sharpe/sortino/treynor, matched)
fam = met[met.primary_family].copy()
fam["p_bonf"] = np.minimum(1.0, fam["p_two"] * len(fam))
fam["p_bh"] = O.bh_adjust(fam["p_two"].values)
met = met.merge(fam[["design", "horizon", "objective", "method", "metric", "p_bonf", "p_bh"]],
                on=["design", "horizon", "objective", "method", "metric"], how="left")

PRIM = ["sharpe", "sortino", "treynor"]
f3 = lambda x: "" if pd.isna(x) else f"{x:.3f}"
blocks = []


def add(title, df, note=None):
    blocks.append((title, df, note))


add("NOTE", pd.DataFrame({"note": [
    "Primary family (12 tests) = the four two-period 1-day tests (Nifty 50 and Bank, Periods A and B) x "
    "{Sharpe, Sortino, Treynor}, matched metric, DTW vs Standard. This family was chosen AFTER seeing the "
    "earlier Sharpe results. Bonferroni and Benjamini-Hochberg are applied to the two-sided p-values of these 12 only; "
    "everything else is descriptive.",
    "Treynor is a side result: the Treynor objective uses the covariance only through the betas, so DTW vs Standard "
    "Treynor compares beta estimates. DTW betas are negatively correlated with sample betas (see beta diagnostic).",
    "Primary DTW = FastDTW (radius 1, |x-y| cost) on raw daily log returns. FastDTW is not continuous in its input "
    "(1e-12 relative perturbations change ~7% of pairwise distances by up to 1%), so re-downloaded data moves DTW "
    "Sharpe by up to ~0.001; Standard results are unaffected.",
    "rf = MAR = 0; annualisation sqrt(252/h); horizons = non-overlapping blocks of daily log returns; bootstrap: paired "
    "moving-block, 5000 resamples, block 20 observations, seed 20261021, days resampled jointly for both portfolios "
    "and all metrics; p_one = P(boot diff <= 0); p_two = 2 min(p_one, 1 - p_one).",
    "Treynor metric = annualised mean / average training-period sample beta (vs ^NSEI) of the portfolio held each day.",
    "Nifty Midcap 100 (expanding window) not rerun: its regime index is no longer available from Yahoo. Original Sharpe "
    "result kept, labelled NOT REPRODUCED."]}))

# ---- per design: matched and cross-evaluated, primary objectives
for d in designs:
    for meth in methods:
        t = met[(met.design == d) & (met.method == meth) & met.objective.isin(PRIM)].copy()
        if t.empty:
            continue
        t = t[["horizon", "objective", "metric", "matched", "std", "alt", "gap", "p_one", "p_two", "p_bonf", "p_bh"]]
        t = t.rename(columns={"alt": meth})
        add(f"{NAMES[d]} - Standard vs {meth.upper()} (objective = optimised ratio, metric = evaluation ratio)", t)

# ---- primary family
f = met[met.primary_family][["design", "objective", "std", "alt", "gap", "p_one", "p_two", "p_bonf", "p_bh"]]
add("PRIMARY FAMILY (12 tests, 1-day, matched)", f.rename(columns={"alt": "dtw"}),
    "Bonferroni threshold 0.05/12 = 0.0042")

# ---- win counts
rows = []
for meth in methods:
    if meth == "std":
        continue
    for obj in PRIM:
        m = met[(met.method == meth) & (met.objective == obj) & met.matched]
        x = met[(met.method == meth) & (met.objective == obj) & ~met.matched]
        p = m[m.primary_family]
        rows.append(dict(method=meth, objective=obj, matched_wins=f"{(m.gap > 0).sum()} of {len(m)}",
                         primary_family_wins=f"{(p.gap > 0).sum()} of {len(p)}",
                         cross_eval_wins=f"{(x.gap > 0).sum()} of {len(x)}",
                         matched_sig_p_two_lt_05=int((m.p_two < 0.05).sum())))
add("DTW WIN COUNTS (gap > 0)", pd.DataFrame(rows))

# ---- realised volatility and training beta (1-day)
v = met[(met.horizon == 1) & (met.metric == "sharpe")][["design", "objective", "method", "std_ann_vol", "alt_ann_vol",
                                                         "std_avg_beta", "alt_avg_beta"]]
add("REALISED VOLATILITY (annualised) AND AVERAGE TRAINING BETA (vs ^NSEI unless own-index), 1-day", v)

# ---- alpha regression
add("DAILY (ALT - STANDARD) RETURN ON MARKET RETURN: alpha (annualised), beta, HAC(5) alpha p-value",
    reg[["design", "objective", "method", "market", "mean_diff_ann", "alpha_ann", "beta", "alpha_p_hac", "beta_p_hac", "r2",
         "n_days"]])

# ---- robustness
r = met[met.objective.isin(["sortino_estrada_clip", "treynor_cap10", "treynor_ownidx"]) & met.matched]
add("ROBUSTNESS (matched metric): Estrada (clipped) Standard vs DTW Sortino; Treynor 10% cap; Treynor own index",
    r[["design", "horizon", "objective", "method", "std", "alt", "gap", "p_one", "p_two"]])

# ---- weights
add("WEIGHT CONCENTRATION (mean over designs, folds, regimes)",
    ws.groupby(["objective", "method"])[["n_stocks", "n_nonzero", "n_at_cap", "max_weight", "half_l1_diff_vs_sharpe"]]
      .mean().reset_index())
deg = ws[ws.objective.str.startswith("treynor") & ((ws.n_at_cap >= 2))]
add("DEGENERATE TREYNOR OPTIMA (>= 2 names at the cap)",
    deg.groupby(["objective", "method"]).size().rename("cells").reset_index()
      .merge(ws[ws.objective.str.startswith("treynor")].groupby(["objective", "method"]).size().rename("of_cells").reset_index()))

# ---- beta diagnostic
add("BETA DIAGNOSTIC (method beta vs sample beta, per fold and regime; summary over cells)",
    bet.groupby(["design", "method", "market"]).agg(cells=("corr", "size"), corr_mean=("corr", "mean"),
                                                    corr_min=("corr", "min"), corr_max=("corr", "max"),
                                                    ratio_median=("median_ratio", "median"),
                                                    meth_neg=("meth_n_neg", "sum"), std_neg=("std_n_neg", "sum"))
      .reset_index())

# ---- midcap original
mid = L.load_pickle(os.path.join(ROOT, "expanding_window_midcap_results.pkl"))
add("NIFTY MIDCAP 100, expanding window - ORIGINAL RESULT, NOT REPRODUCED (Sharpe only)",
    pd.DataFrame([dict(std=mid['pooled_std_metrics']['sharpe'], dtw=mid['pooled_dtw_metrics']['sharpe'],
                       gap=mid['pooled_dtw_metrics']['sharpe'] - mid['pooled_std_metrics']['sharpe'],
                       p_one=mid['bootstrap']['p_value'])]))

# ---- write
with open(os.path.join(OUT, f"{prefix}_summary.txt"), "w", encoding="utf-8") as ft, \
        open(os.path.join(OUT, f"{prefix}_summary.tsv"), "w", encoding="utf-8") as fv:
    for title, df, note in blocks:
        ft.write(f"\n=== {title} ===\n")
        fv.write(f"\n# {title}\n")
        if title == "NOTE":
            for n in df["note"]:
                ft.write(f"- {n}\n")
                fv.write(f"# {n}\n")
            continue
        ft.write(df.to_string(index=False, float_format=f3) + "\n")
        if note:
            ft.write(f"({note})\n")
        fv.write(df.to_csv(sep="\t", index=False, float_format="%.4f"))
print(open(os.path.join(OUT, f"{prefix}_summary.txt"), encoding="utf-8").read())
