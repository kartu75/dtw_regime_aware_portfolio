# -*- coding: utf-8 -*-
"""Controls summary: DTW (primary) vs (a) return-normalised DTW, (b) volatility-only kernel,
(c) inverse volatility - each against Standard, same designs, objectives, horizons and bootstrap.
Reads results/objectives/full_*.pkl and ctrl_*.pkl plus the caches.
Writes results/objectives/controls_summary.txt and controls_summary.tsv"""
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
MORDER = ["dtw", "dtwz", "volk", "invvol"]
f3 = lambda x: "" if pd.isna(x) else f"{x:.3f}"


def load(prefix):
    r = {}
    for p in glob.glob(os.path.join(OUT, f"{prefix}_*.pkl")):
        d = os.path.basename(p)[len(prefix) + 1:-4]
        if d in DORDER:
            r[d] = L.load_pickle(p)
    return r


full, ctrl = load("full"), load("ctrl")
met = pd.concat([x["metrics"] for x in list(full.values()) + list(ctrl.values())], ignore_index=True)
reg = pd.concat([x["regression"] for x in list(full.values()) + list(ctrl.values())], ignore_index=True)
ws = pd.concat([x["weight_stats"].assign(design=d) for d, x in list(full.items()) + list(ctrl.items())], ignore_index=True)
ws = ws.drop_duplicates(subset=["design", "fold", "regime", "objective", "method"])
m = met[met.objective.isin(PRIM) & met.matched].copy()
blocks = []

# 1) matched gaps vs Standard, side by side
g = m.pivot_table(index=["design", "horizon", "objective"], columns="method", values="gap").reindex(columns=MORDER)
p = m.pivot_table(index=["design", "horizon", "objective"], columns="method", values="p_two").reindex(columns=MORDER)
side = g.round(3).astype(str) + " (p=" + p.round(3).astype(str) + ")"
side = side.reset_index()
side["_o"] = side.design.map({d: i for i, d in enumerate(DORDER)})
side = side.sort_values(["_o", "objective", "horizon"]).drop(columns="_o")
blocks.append(("MATCHED-METRIC GAP vs STANDARD (method - Standard), two-sided p in brackets", side))

# 2) win counts per method
rows = []
for meth in MORDER:
    for obj in PRIM:
        x = m[(m.method == meth) & (m.objective == obj)]
        prim = x[x.horizon.eq(1) & x.design.str.startswith("2p")]
        rows.append(dict(method=meth, objective=obj, wins=f"{(x.gap > 0).sum()} of {len(x)}",
                         wins_2p_1day=f"{(prim.gap > 0).sum()} of {len(prim)}",
                         mean_gap=x.gap.mean(), p_two_lt_05=int((x.p_two < 0.05).sum()),
                         sig_direction=",".join(np.where(x[x.p_two < 0.05].gap > 0, "method", "Standard"))))
blocks.append(("WIN COUNTS vs STANDARD (matched metric; descriptive - controls are not in the primary family)",
               pd.DataFrame(rows)))

# 3) agreement of each control with DTW: correlation of matched gaps across all tests
gg = g.dropna()
blocks.append(("CORRELATION OF MATCHED GAPS ACROSS TESTS (how closely each control tracks DTW)",
               gg.corr().round(3).reset_index().rename(columns={"method": ""})))

# 4) beta exposure: regression of (method - Standard) daily returns on ^NSEI, Sharpe and Sortino objectives
r = reg[reg.objective.isin(["sharpe", "sortino"]) & reg.method.isin(MORDER)]
blocks.append(("DAILY (METHOD - STANDARD) ON ^NSEI: annualised mean gap, alpha, beta, HAC alpha p",
               r.pivot_table(index=["design", "objective"], columns="method",
                             values=["mean_diff_ann", "alpha_ann", "beta", "alpha_p_hac"]).round(3)
                .reindex(columns=MORDER, level=1).reset_index()))

# 5) realised vol and training beta, Sharpe objective, 1 day
v = met[(met.horizon == 1) & (met.metric == "sharpe") & (met.objective == "sharpe")]
vv = v.pivot_table(index="design", columns="method", values=["alt_ann_vol", "alt_avg_beta"]).reindex(columns=MORDER, level=1)
std_v = v.groupby("design")[["std_ann_vol", "std_avg_beta"]].first()
blocks.append(("REALISED VOL AND AVERAGE TRAINING BETA, Sharpe objective, 1 day (Standard columns first)",
               pd.concat([std_v, vv], axis=1).round(3).reset_index()))

# 6) weight concentration
blocks.append(("WEIGHT CONCENTRATION (mean over folds and regimes)",
               ws[ws.objective.isin(PRIM)].groupby(["objective", "method"])[["n_nonzero", "n_at_cap", "max_weight"]]
               .mean().round(2).reset_index()))

# 7) mechanism: what the similarity matrices track, and DTW vs vol-kernel portfolio overlap
rows, ov = [], []
for d in DORDER:
    for fp in sorted(glob.glob(os.path.join(HERE, "cache", "step0", d, "fold*.pkl"))):
        c = L.load_pickle(fp)
        k = c["fold"]["fold_idx"]
        zc = L.load_pickle(os.path.join(HERE, "cache", "dtwz", d, f"fold{k}.pkl"))["dtw_inputs"]
        for reg_name, di in c["dtw_inputs"].items():
            st = di["stocks"]
            if len(st) < 4 or reg_name not in c["std_inputs"]:
                continue
            iu = np.triu_indices(len(st), 1)
            vol = di["vol"].loc[st].values
            pvol = np.sqrt(np.outer(vol, vol))[iu]
            C = c["std_inputs"][reg_name]["corr"].reindex(index=st, columns=st).values[iu]
            ok = ~np.isnan(C)
            mats = {"dtw": di["S_tilde"].values[iu], "volk": O.vol_kernel(di["vol"].loc[st])[0].values[iu]}
            if reg_name in zc:
                mats["dtwz"] = zc[reg_name]["S_tilde"].loc[st, st].values[iu]
            for meth, S in mats.items():
                rows.append(dict(design=d, method=meth, corr_with_sample_corr=np.corrcoef(S[ok], C[ok])[0, 1],
                                 corr_with_pair_vol=np.corrcoef(S, pvol)[0, 1],
                                 corr_with_dtw=np.corrcoef(S, mats["dtw"])[0, 1]))
            wd = c["weights_sharpe"]["dtw"].get(reg_name)
            if wd is not None:
                vk = pd.DataFrame(O.vol_kernel(di["vol"].loc[st])[0].values * np.outer(vol, vol), index=st, columns=st)
                wv = O.optimize_ratio_cov(di["mu"], vk)
                ws_ = c["weights_sharpe"]["std"].get(reg_name)
                u = wd.index.union(wv.index)
                ov.append(dict(design=d, half_l1_dtw_vs_volk=float((wd.reindex(u, fill_value=0) - wv.reindex(u, fill_value=0)).abs().sum() / 2),
                               half_l1_dtw_vs_std=float((wd.reindex(wd.index.union(ws_.index), fill_value=0)
                                                         - ws_.reindex(wd.index.union(ws_.index), fill_value=0)).abs().sum() / 2)
                               if ws_ is not None else np.nan))
mech = pd.DataFrame(rows).groupby(["design", "method"]).mean().round(3).reset_index()
blocks.append(("WHAT THE SIMILARITY MATRIX TRACKS (off-diagonal correlations, mean over folds and regimes)", mech))
blocks.append(("SHARPE PORTFOLIO OVERLAP: half-L1 distance DTW vs vol-kernel, and DTW vs Standard (0 = identical, 1 = disjoint)",
               pd.DataFrame(ov).groupby("design").mean().round(3).reset_index()))

with open(os.path.join(OUT, "controls_summary.txt"), "w", encoding="utf-8") as ft, \
        open(os.path.join(OUT, "controls_summary.tsv"), "w", encoding="utf-8") as fv:
    for title, df in blocks:
        ft.write(f"\n=== {title} ===\n" + df.to_string(index=False, float_format=f3) + "\n")
        fv.write(f"\n# {title}\n" + df.to_csv(sep="\t", index=False, float_format="%.4f"))
print(open(os.path.join(OUT, "controls_summary.txt"), encoding="utf-8").read())
