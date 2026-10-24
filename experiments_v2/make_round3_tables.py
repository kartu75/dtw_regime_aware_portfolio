# -*- coding: utf-8 -*-
"""Round 3 summary: baselines (Ledoit-Wolf, 1/N, GMV) vs Standard and vs raw DTW; Sakoe-Chiba windowed
DTW (w in {1,2,3,5}, normalised primary / raw secondary); lead-lag check; deflated Sharpe; running count.
Writes results/round3_summary.txt and .tsv"""
import glob
import os

import numpy as np
import pandas as pd

import dtwlib as L
import objectives as O

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OBJ = os.path.join(RES, "objectives")
DORDER = ["2p_N50_A", "2p_N50_B", "2p_BANK_A", "2p_BANK_B", "single_N50", "exp_N50", "exp_BANK"]
PRIM = ["sharpe", "sortino", "treynor"]
BANDS = [f"band{w}_{t}" for t in ("z", "raw") for w in (1, 2, 3, 5)]
f3 = lambda x: "" if pd.isna(x) else f"{x:.3f}"


def load(prefixes):
    met, reg = [], []
    for pre in prefixes:
        for p in glob.glob(os.path.join(OBJ, f"{pre}_*.pkl")):
            if os.path.basename(p)[len(pre) + 1:-4] in DORDER:
                x = L.load_pickle(p)
                met.append(x["metrics"])
                reg.append(x["regression"])
    met, reg = pd.concat(met, ignore_index=True), pd.concat(reg, ignore_index=True)
    for df in (met, reg):
        if "ref" not in df:
            df["ref"] = "std"
        df["ref"] = df["ref"].fillna("std")
    return (met.drop_duplicates(subset=["design", "horizon", "objective", "method", "ref", "metric"]),
            reg.drop_duplicates(subset=["design", "objective", "method", "ref"]))


met, reg = load(["full", "ctrl", "base", "band"])
m = met[met.objective.isin(PRIM) & met.matched]
gp_std = m[m.ref == "std"].pivot_table(index=["design", "horizon", "objective"], columns="method", values="gap")


def summary(methods, ref):
    rows = []
    for meth in methods:
        for obj in PRIM:
            x = m[(m.method == meth) & (m.objective == obj) & (m.ref == ref)]
            if x.empty:
                continue
            r_ = reg[(reg.method == meth) & (reg.objective == obj) & (reg.ref == ref)]
            b1 = met[(met.method == meth) & (met.objective == obj) & (met.ref == ref) & (met.metric == "sharpe") & (met.horizon == 1)]
            row = dict(method=meth, vs=ref, objective=obj, wins=f"{(x.gap > 0).sum()} of {len(x)}",
                       wins_2p_1d=f"{(x[x.design.str.startswith('2p') & (x.horizon == 1)].gap > 0).sum()} of 4",
                       mean_gap=x.gap.mean(), n_p2_lt_05=int((x.p_two < 0.05).sum()),
                       sig_favours=",".join(np.where(x[x.p_two < 0.05].gap > 0, "method", ref)) or "-",
                       gap_beta=r_.beta.mean(), alpha_ann=r_.alpha_ann.mean(), n_alpha_p_lt_05=int((r_.alpha_p_hac < 0.05).sum()),
                       alpha_sig_sign=",".join(np.where(r_[r_.alpha_p_hac < 0.05].alpha_ann > 0, "+", "-")) or "-",
                       train_beta_minus_ref=(b1.alt_avg_beta - b1.std_avg_beta).mean())
            if ref == "std" and meth in gp_std:
                row["corr_gaps_raw_dtw"] = gp_std[meth].corr(gp_std["dtw"]) if meth != "dtw" else 1.0
                row["corr_gaps_norm_fastdtw"] = gp_std[meth].corr(gp_std["dtwz"]) if "dtwz" in gp_std else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def primary_table(methods, ref):
    f = m[(m.horizon == 1) & m.design.str.startswith("2p") & (m.objective == "sharpe") & m.method.isin(methods) & (m.ref == ref)].copy()
    f["cell"] = f.gap.map("{:+.3f}".format) + " (" + f.p_two.map("{:.2f}".format) + ")"
    return f.pivot(index="method", columns="design", values="cell").reindex(methods)[["2p_N50_A", "2p_N50_B", "2p_BANK_A", "2p_BANK_B"]].reset_index()


def reg_table(methods, ref):
    r = reg[reg.method.isin(methods) & (reg.objective == "sharpe") & (reg.ref == ref)].copy()
    r["cell"] = r.beta.map("{:.2f}".format) + " / " + (r.alpha_ann * 100).map("{:+.1f}".format) + "% (" + r.alpha_p_hac.map("{:.2f}".format) + ")"
    return r.pivot(index="method", columns="design", values="cell").reindex(methods)[DORDER].reset_index()


vlog = pd.read_csv(os.path.join(HERE, "variants_log.csv"), dtype=str)
new = vlog[vlog.stage != "step0_regression"]
count = (f"RUNNING VARIANT COUNT: {len(new)} logged tests (excl. step0 reruns), "
         f"{new[['objective', 'metric', 'covariances', 'design', 'horizon']].drop_duplicates().shape[0]} distinct; "
         f"{(new.counted_in_family == 'yes').sum()} in the primary family; by stage: {new.stage.value_counts().to_dict()}; "
         f"method pairs: {sorted(set(new.covariances))}")
blocks = [("RUNNING VARIANT COUNT", count)]

BASE = ["lw", "ew", "gmv"]
blocks += [("BASELINES vs STANDARD (matched metric)", summary(["dtw"] + BASE, "std")),
           ("BASELINES vs RAW DTW (matched metric; gap = baseline - raw DTW)", summary(BASE, "dtw")),
           ("BASELINES: Sharpe gap, primary two-period designs, 1-day, vs STANDARD (two-sided p)", primary_table(["dtw"] + BASE, "std")),
           ("BASELINES: Sharpe gap, primary two-period designs, 1-day, vs RAW DTW (two-sided p)", primary_table(BASE, "dtw")),
           ("BASELINES: gap regression on ^NSEI, Sharpe objective, vs STANDARD: beta / alpha (HAC p)", reg_table(["dtw"] + BASE, "std")),
           ("BASELINES: gap regression on ^NSEI, Sharpe objective, vs RAW DTW: beta / alpha (HAC p)", reg_table(BASE, "dtw")),
           ("WINDOWED DTW (Sakoe-Chiba w in {1,2,3,5}, fixed before results) vs STANDARD", summary(BANDS, "std")),
           ("WINDOWED DTW: Sharpe gap, primary designs, 1-day, vs STANDARD (two-sided p)",
            primary_table(["dtw", "dtwz"] + BANDS, "std")),
           ("WINDOWED DTW: gap regression, Sharpe objective, vs STANDARD: beta / alpha (HAC p)", reg_table(["dtw", "dtwz"] + BANDS, "std"))]

# what the windowed similarity matrices track
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
            base = di["S_tilde"].values[iu]
            for meth in BANDS:
                pth = os.path.join(HERE, "cache", "dtwvar", meth, d, f"fold{k}.pkl")
                if os.path.exists(pth):
                    x = L.load_pickle(pth)["dtw_inputs"].get(reg_)
                    if x is not None:
                        S = x["S_tilde"].loc[st, st].values[iu]
                        rows.append(dict(method=meth, with_sample_corr=np.corrcoef(S[ok], C[ok])[0, 1],
                                         with_pair_vol=np.corrcoef(S, pvol)[0, 1], with_raw_dtw=np.corrcoef(S, base)[0, 1]))
blocks.append(("WHAT THE WINDOWED SIMILARITY MATRICES TRACK (mean off-diagonal correlations)",
               pd.DataFrame(rows).groupby("method").mean().reindex(BANDS).reset_index()))

for title, path in [("LEAD-LAG CHECK (normalised returns)", os.path.join(RES, "leadlag", "summary.txt")),
                    ("DEFLATED SHARPE RATIO", os.path.join(RES, "dsr", "dsr.txt"))]:
    blocks.append((title, open(path, encoding="utf-8").read() if os.path.exists(path) else "(not run yet)"))

with open(os.path.join(RES, "round3_summary.txt"), "w", encoding="utf-8") as ft, \
        open(os.path.join(RES, "round3_summary.tsv"), "w", encoding="utf-8") as fv:
    for title, b in blocks:
        if isinstance(b, str):
            ft.write(f"\n=== {title} ===\n{b}\n")
            fv.write(f"\n# {title}\n" + "\n".join("# " + l for l in b.splitlines()) + "\n")
        else:
            ft.write(f"\n=== {title} ===\n" + b.to_string(index=False, float_format=f3) + "\n")
            fv.write(f"\n# {title}\n" + b.to_csv(sep="\t", index=False, float_format="%.4f"))
print(open(os.path.join(RES, "round3_summary.txt"), encoding="utf-8").read())
