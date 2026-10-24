# -*- coding: utf-8 -*-
"""Summary for the volatility-level control (d) and the DTW variants, next to primary raw DTW and the
post hoc normalised FastDTW extension. Reads results/objectives/{full,ctrl,ctrl2,var}_*.pkl and weights csv.
Writes results/objectives/variants_summary.txt and variants_summary.tsv"""
import glob
import os

import numpy as np
import pandas as pd

import dtwlib as L
import objectives as O

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "objectives")
DORDER = ["2p_N50_A", "2p_N50_B", "2p_BANK_A", "2p_BANK_B", "single_N50", "exp_N50", "exp_BANK"]
PRIM = ["sharpe", "sortino", "treynor"]
LABEL = {"dtw": "raw FastDTW (PRIMARY, pre-specified)", "dtwz": "normalised FastDTW (main extension, post hoc)",
         "vollev": "control (d): volatility-level kernel", "exactL1_raw": "exact L1 DTW, raw",
         "exactL1_z": "exact L1 DTW, normalised", "dtw2_raw": "dtw2 (squared cost), raw",
         "dtw2_z": "dtw2 (squared cost), normalised", "sdtw_raw": "soft-DTW divergence, raw",
         "sdtw_z": "soft-DTW divergence, normalised"}
MORDER = list(LABEL)
f3 = lambda x: "" if pd.isna(x) else f"{x:.3f}"

frames = {"metrics": [], "regression": []}
for prefix in ["full", "ctrl", "ctrl2", "var"]:
    for p in glob.glob(os.path.join(OUT, f"{prefix}_*.pkl")):
        d = os.path.basename(p)[len(prefix) + 1:-4]
        if d in DORDER:
            x = L.load_pickle(p)
            frames["metrics"].append(x["metrics"])
            frames["regression"].append(x["regression"])
met = pd.concat(frames["metrics"], ignore_index=True)
met = met[met.method.isin(MORDER)].drop_duplicates(subset=["design", "horizon", "objective", "method", "metric"])
reg = pd.concat(frames["regression"], ignore_index=True)
reg = reg[reg.method.isin(MORDER)].drop_duplicates(subset=["design", "objective", "method"])
m = met[met.objective.isin(PRIM) & met.matched].copy()
methods = [x for x in MORDER if x in set(m.method)]

vlog = pd.read_csv(os.path.join(HERE, "variants_log.csv"), dtype=str)
new = vlog[vlog.stage != "step0_regression"]
count_line = (f"RUNNING VARIANT COUNT (variants_log.csv, excluding step0 reruns): {len(new)} logged tests "
              f"({new[['objective', 'metric', 'covariances', 'design', 'horizon']].drop_duplicates().shape[0]} distinct); "
              f"{(new.counted_in_family == 'yes').sum()} in the primary family; covariance methods tried vs Standard: "
              f"{sorted(set(new.covariances.str.replace('std,', '', regex=False)))}; "
              f"matched-metric tests only: {(new.note.str.contains('matched') & ~new.note.str.contains('cross')).sum()}")
blocks = [("RUNNING VARIANT COUNT", pd.DataFrame({"": [count_line]}))]

# 1) summary per method
rows = []
gp = m.pivot_table(index=["design", "horizon", "objective"], columns="method", values="gap")
for meth in methods:
    for obj in PRIM:
        x = m[(m.method == meth) & (m.objective == obj)]
        r_ = reg[(reg.method == meth) & (reg.objective == obj)]
        b1 = met[(met.method == meth) & (met.objective == obj) & (met.metric == "sharpe") & (met.horizon == 1)]
        corr = gp[meth].corr(gp["dtw"]) if meth != "dtw" and meth in gp else np.nan
        rows.append(dict(method=meth, objective=obj, wins=f"{(x.gap > 0).sum()} of {len(x)}",
                         wins_2p_1d=f"{(x[x.design.str.startswith('2p') & (x.horizon == 1)].gap > 0).sum()} of 4",
                         mean_gap=x.gap.mean(), n_p2_lt_05=int((x.p_two < 0.05).sum()),
                         sig_favours=",".join(np.where(x[x.p_two < 0.05].gap > 0, "method", "Std")) or "-",
                         gap_beta_mean=r_.beta.mean(), alpha_ann_mean=r_.alpha_ann.mean(),
                         n_alpha_p_lt_05=int((r_.alpha_p_hac < 0.05).sum()),
                         train_beta_minus_std=(b1.alt_avg_beta - b1.std_avg_beta).mean(),
                         corr_gaps_with_raw_dtw=corr))
blocks.append(("SUMMARY BY METHOD vs STANDARD (matched metric; descriptive except the primary family of 12)",
               pd.DataFrame(rows)))

# 2) full matched table: gap, p_one, p_two per test
t = m[["design", "horizon", "objective", "method", "std", "alt", "gap", "p_one", "p_two"]].copy()
t["_o"] = t.design.map({d: i for i, d in enumerate(DORDER)})
t["_m"] = t.method.map({x: i for i, x in enumerate(MORDER)})
t = t.sort_values(["_o", "objective", "horizon", "_m"]).drop(columns=["_o", "_m"])
blocks.append(("ALL MATCHED TESTS: Standard, method, gap, one-sided and two-sided p", t))

# 3) beta of the gap, alpha, HAC p; training beta (Sharpe objective, 1 day)
r = reg[reg.objective.isin(PRIM)][["design", "objective", "method", "mean_diff_ann", "alpha_ann", "alpha_p_hac", "beta",
                                    "beta_p_hac"]].copy()
tb = met[(met.metric == "sharpe") & (met.horizon == 1) & met.objective.isin(PRIM)][
    ["design", "objective", "method", "std_avg_beta", "alt_avg_beta"]]
r = r.merge(tb, on=["design", "objective", "method"], how="left")
r["_o"] = r.design.map({d: i for i, d in enumerate(DORDER)})
r["_m"] = r.method.map({x: i for i, x in enumerate(MORDER)})
blocks.append(("DAILY (METHOD - STANDARD) ON ^NSEI: mean gap, alpha (annualised), HAC alpha p, beta; training betas",
               r.sort_values(["_o", "objective", "_m"]).drop(columns=["_o", "_m"])))

# 4) Sharpe-portfolio overlap with raw DTW, and what the similarity matrices track
ov = []
for d in DORDER:
    wfiles = {"dtw": f"full_{d}_weights.csv", "dtwz": f"ctrl_{d}_weights.csv", "vollev": f"ctrl2_{d}_weights.csv",
              "var": f"var_{d}_weights.csv"}
    W = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in wfiles.values() if os.path.exists(os.path.join(OUT, f))])
    W = W[W.objective == "sharpe"].drop_duplicates(subset=["fold", "regime", "method", "stock"])
    base = W[W.method == "dtw"]
    for meth in methods:
        if meth == "dtw":
            continue
        x = W[W.method == meth]
        for (fold, reg_), g in x.groupby(["fold", "regime"]):
            b = base[(base.fold == fold) & (base.regime == reg_)].set_index("stock").weight
            a = g.set_index("stock").weight
            u = a.index.union(b.index)
            ov.append(dict(design=d, method=meth, half_l1_vs_raw_dtw=float((a.reindex(u, fill_value=0) - b.reindex(u, fill_value=0)).abs().sum() / 2)))
blocks.append(("SHARPE-PORTFOLIO OVERLAP WITH RAW DTW (half-L1: 0 = identical, 1 = disjoint), mean over folds/regimes",
               pd.DataFrame(ov).pivot_table(index="design", columns="method", values="half_l1_vs_raw_dtw")
               .reindex(columns=[x for x in methods if x != "dtw"]).reset_index()))

rows = []
for d in DORDER:
    for fp in sorted(glob.glob(os.path.join(HERE, "cache", "step0", d, "fold*.pkl"))):
        c = L.load_pickle(fp)
        k = c["fold"]["fold_idx"]
        for reg_, di in c["dtw_inputs"].items():
            st = di["stocks"]
            if len(st) < 4 or reg_ not in c["std_inputs"]:
                continue
            iu = np.triu_indices(len(st), 1)
            vol = di["vol"].loc[st].values
            pvol = np.sqrt(np.outer(vol, vol))[iu]
            C = c["std_inputs"][reg_]["corr"].reindex(index=st, columns=st).values[iu]
            ok = ~np.isnan(C)
            mats = {"dtw": di["S_tilde"].values[iu], "vollev": O.vol_level_kernel(di["vol"].loc[st])[0].values[iu]}
            for meth in methods:
                path = None
                if meth == "dtwz":
                    path = os.path.join(HERE, "cache", "dtwz", d, f"fold{k}.pkl")
                elif meth not in ("dtw", "vollev"):
                    path = os.path.join(HERE, "cache", "dtwvar", meth, d, f"fold{k}.pkl")
                if path and os.path.exists(path):
                    x = L.load_pickle(path)["dtw_inputs"].get(reg_)
                    if x is not None:
                        mats[meth] = x["S_tilde"].loc[st, st].values[iu]
            for meth, S in mats.items():
                rows.append(dict(design=d, method=meth, with_sample_corr=np.corrcoef(S[ok], C[ok])[0, 1],
                                 with_pair_vol=np.corrcoef(S, pvol)[0, 1], with_raw_dtw=np.corrcoef(S, mats["dtw"])[0, 1]))
mech = pd.DataFrame(rows).groupby("method")[["with_sample_corr", "with_pair_vol", "with_raw_dtw"]].mean()
blocks.append(("WHAT EACH SIMILARITY MATRIX TRACKS (off-diagonal correlations, mean over all designs/folds/regimes)",
               mech.reindex([x for x in MORDER if x in mech.index]).reset_index()))
blocks.append(("METHOD LABELS", pd.DataFrame([dict(method=k, description=v) for k, v in LABEL.items()])))

with open(os.path.join(OUT, "variants_summary.txt"), "w", encoding="utf-8") as ft, \
        open(os.path.join(OUT, "variants_summary.tsv"), "w", encoding="utf-8") as fv:
    for title, df in blocks:
        ft.write(f"\n=== {title} ===\n" + (df.iloc[0, 0] if title.startswith("RUNNING") else df.to_string(index=False, float_format=f3)) + "\n")
        fv.write(f"\n# {title}\n" + (f"# {df.iloc[0, 0]}\n" if title.startswith("RUNNING") else df.to_csv(sep="\t", index=False, float_format="%.4f")))
print(open(os.path.join(OUT, "variants_summary.txt"), encoding="utf-8").read())
