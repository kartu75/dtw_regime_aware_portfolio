# -*- coding: utf-8 -*-
"""
Lead-lag mechanism check on NORMALISED returns (z-scored with the fold's training-window mean/std).
Scope (fixed 2026-10-21 before results): the last fold of each two-period design (2p_N50_A, 2p_N50_B,
2p_BANK_A, 2p_BANK_B), its training episodes with at least 20 cleaned days, same stock inclusion and
row cleaning as the primary DTW. For every stock pair and episode:
  dtw_lag  = median of (q - p) along the unconstrained exact L1 DTW warping path
  cc_lag   = lag k in [-5, 5] maximising corr(a_t, b_{t+k})   (positive = b follows a, same convention)
Placebo: b circularly shifted by a random s in [10, T-10] days (seed 20261021), same two statistics.
Writes results/leadlag/pairs.csv and summary.txt/.tsv
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import dtw_variants as V
import dtwlib as L
from run_step0 import DESIGNS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "leadlag")
MAX_LAG, MIN_T, SEED, CHUNK = 5, 20, 20261021, 40
DESIGNS_LL = ["2p_N50_A", "2p_N50_B", "2p_BANK_A", "2p_BANK_B"]


def episode_task(payload):
    design, idx, regime, X, names, seed = payload
    n, T = X.shape
    i, j = np.triu_indices(n, 1)
    rng = np.random.RandomState(seed)
    shifts = rng.randint(10, T - 9, size=len(i))
    Bp = np.stack([np.roll(X[jj], s) for jj, s in zip(j, shifts)])
    dl, dlp = np.empty(len(i)), np.empty(len(i))
    for s in range(0, len(i), CHUNK):
        sl = slice(s, s + CHUNK)
        dl[sl] = V.path_lags(X[i[sl]], X[j[sl]])
        dlp[sl] = V.path_lags(X[i[sl]], Bp[sl])
    rows = []
    for k, (a, b) in enumerate(zip(i, j)):
        cl, cr = V.xcorr_lag(X[a], X[b], MAX_LAG)
        clp, crp = V.xcorr_lag(X[a], Bp[k], MAX_LAG)
        rows.append(dict(design=design, episode=idx, regime=regime, T=T, a=names[a], b=names[b],
                         dtw_lag=dl[k], cc_lag=cl, cc_peak=cr, dtw_lag_placebo=dlp[k], cc_lag_placebo=clp,
                         cc_peak_placebo=crp, shift=int(shifts[k])))
    return rows


def build_tasks():
    tasks = []
    for design in DESIGNS_LL:
        spec = DESIGNS[design]
        raw = np.log(L.load_frame(spec["stocks"]) / L.load_frame(spec["stocks"]).shift(1))
        src = os.path.join(HERE, "cache", "step0", design)
        last = max(os.listdir(src), key=lambda f: int(f[4:-4]))
        c = L.load_pickle(os.path.join(src, last))
        a, b = c["fold"]["train_start"], c["fold"]["train_end"]
        tr = raw.loc[a:b]
        z = (raw - tr.mean()) / tr.std()
        for idx, ep in c["episodes"].iterrows():
            er = raw.loc[ep['start']:ep['end']]
            vc = er.notna().sum()
            inc = vc[vc >= 0.9 * len(er)].index.tolist()
            rows = er[inc].dropna().index
            if len(inc) < 2 or len(rows) < MIN_T:
                continue
            X = z.loc[rows, inc].values.T
            tasks.append((design, int(idx), ep['regime'], np.ascontiguousarray(X), inc, SEED + int(idx)))
    return tasks


def summarise(df):
    def stats(g):
        out = {}
        for tag, dl, cl in [("real", "dtw_lag", "cc_lag"), ("placebo", "dtw_lag_placebo", "cc_lag_placebo")]:
            d, c = g[dl], g[cl]
            nz = (d.abs() >= 0.5) & (c != 0)
            out.update({
                f"{tag}_share_dtw_lag0": (d.abs() < 0.5).mean(),
                f"{tag}_mean_abs_dtw_lag": d.abs().mean(),
                f"{tag}_share_cc_lag0": (c == 0).mean(),
                f"{tag}_exact_agree": (np.round(d) == c).mean(),
                f"{tag}_sign_agree_nonzero": (np.sign(d[nz]) == np.sign(c[nz])).mean() if nz.any() else np.nan,
                f"{tag}_spearman": d.rank().corr(c.rank()),
            })
        strong = g[(g.cc_lag != 0) & (g.cc_peak > 2 / np.sqrt(g["T"]))]
        out["n_pair_episodes"] = len(g)
        out["n_strong_cc_nonzero"] = len(strong)
        out["strong_sign_agree"] = (np.sign(strong.dtw_lag) == np.sign(strong.cc_lag)).mean() if len(strong) else np.nan
        out["strong_spearman"] = strong.dtw_lag.rank().corr(strong.cc_lag.rank()) if len(strong) > 2 else np.nan
        return pd.Series(out)
    df = df.assign(universe=df.design.str.split("_").str[1], era=df.design.str.split("_").str[2])
    by_u = df.groupby("universe").apply(stats)
    by_d = df.groupby("design").apply(stats)
    return by_u, by_d


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tasks = build_tasks()
    print(f"{len(tasks)} episodes, {sum(t[3].shape[0] * (t[3].shape[0] - 1) // 2 for t in tasks)} pair-episodes")
    sys.stdout.flush()
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("MAX_WORKERS", "10"))) as ex:
        for f in as_completed([ex.submit(episode_task, t) for t in tasks]):
            rows += f.result()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    by_u, by_d = summarise(df)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as ft, \
            open(os.path.join(OUT, "summary.tsv"), "w", encoding="utf-8") as fv:
        for title, t in [("BY UNIVERSE", by_u), ("BY DESIGN (universe x era)", by_d)]:
            ft.write(f"\n=== {title} ===\n" + t.T.to_string(float_format=lambda x: f"{x:.3f}") + "\n")
            fv.write(f"\n# {title}\n" + t.T.to_csv(sep="\t", float_format="%.4f"))
    print(open(os.path.join(OUT, "summary.txt"), encoding="utf-8").read())
    print(f"done in {time.time() - t0:.0f}s")
