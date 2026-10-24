# -*- coding: utf-8 -*-
"""
DTW variants on the Step 0 folds: exact L1 DTW, dtw2 (squared cost) and soft-DTW divergence, each on
raw returns and on training-window z-scored returns (normalised exactly as run_dtwz_cache.py).
Everything except the distance is the primary specification: Step 0 episodes, 90% inclusion rule,
row cleaning, simple average over common stocks, kernel h = 0.5 x median positive distance,
eigenvalue clipping at 1e-8, D_sigma scaling with raw regime volatilities; stock-to-market distances
for Treynor on the same rows (market z-scored too in the normalised variants).
Writes cache/dtwvar/<variant>/<design>/fold<k>.pkl (resumable). Usage: python run_dtw_variants.py [design ...]
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
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))
KINDS = ["exactL1", "dtw2", "sdtw"]
TRANSFORMS = ["raw", "z"]
VARIANTS = [f"{k}_{t}" for k in KINDS for t in TRANSFORMS]
ORDER = ["2p_BANK_A", "2p_BANK_B", "exp_BANK", "single_N50", "exp_N50", "2p_N50_B", "2p_N50_A"]
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
_logf = open(os.path.join(HERE, "logs", "dtw_variants.log"), "a", encoding="utf-8")


def log(*a):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " + " ".join(map(str, a))
    print(line)
    sys.stdout.flush()
    _logf.write(line + "\n")
    _logf.flush()


def task(payload):
    """One episode x one distance kind x one transform."""
    idx, regime, kind, transform, clean, mkts = payload
    t0 = time.time()
    X = clean.values.T  # n x T
    gamma = V.episode_gamma(X) if kind == "sdtw" else None
    D = V.pairwise(X, kind, gamma)
    mk = {}
    for name, m in mkts.items():
        ok = m.notna().values
        d = V.to_series(np.ascontiguousarray(X[:, ok]), m.values[ok], kind, gamma)
        mk[name] = dict(dist=pd.Series(d, index=clean.columns), n_days=int(ok.sum()))
    return idx, regime, kind, transform, pd.DataFrame(D, index=clean.columns, columns=clean.columns), mk, gamma, time.time() - t0


def run_design(design):
    spec = DESIGNS[design]
    prices = L.load_frame(spec["stocks"])
    raw = np.log(prices / prices.shift(1))
    mk_raw = {n: np.log(L.load_index(k) / L.load_index(k).shift(1)) for n, k in spec["markets"].items()}
    src = os.path.join(HERE, "cache", "step0", design)
    for fname in sorted(os.listdir(src), key=lambda f: int(f[4:-4])):
        outs = {v: os.path.join(HERE, "cache", "dtwvar", v, design, fname) for v in VARIANTS}
        if all(os.path.exists(p) for p in outs.values()):
            log(f"{design} {fname}: exists, skipping")
            continue
        t0 = time.time()
        c = L.load_pickle(os.path.join(src, fname))
        a, b = c["fold"]["train_start"], c["fold"]["train_end"]
        tr = raw.loc[a:b]
        z = (raw - tr.mean()) / tr.std()
        mk_z = {n: (s - s.loc[a:b].mean()) / s.loc[a:b].std() for n, s in mk_raw.items()}
        tasks = []
        for idx, ep in c["episodes"].iterrows():
            er = raw.loc[ep['start']:ep['end']]
            vc = er.notna().sum()
            inc = vc[vc >= 0.9 * len(er)].index.tolist()      # same inclusion rule as the primary
            if len(inc) < 2:
                continue
            rows = er[inc].dropna().index                     # same cleaned rows as the primary
            for transform, src_df, mks in [("raw", raw, mk_raw), ("z", z, mk_z)]:
                clean = src_df.loc[rows, inc]
                mkts = {n: s.reindex(rows) for n, s in mks.items()}
                for kind in KINDS:
                    tasks.append((idx, ep['regime'], kind, transform, clean, mkts))
        res = {v: {} for v in VARIANTS}
        gammas = []
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(tasks)))) as ex:
            futs = [ex.submit(task, t) for t in tasks]
            for f in as_completed(futs):
                idx, regime, kind, transform, D, mk, gamma, _ = f.result()
                res[f"{kind}_{transform}"][idx] = {'regime': regime, 'dtw_matrix': D, 'stocks': list(D.index), 'market': mk}
                if gamma is not None:
                    gammas.append((transform, idx, gamma))
        for v in VARIANTS:
            eps = dict(sorted(res[v].items()))
            inputs = {}
            for r, di0 in c["dtw_inputs"].items():
                avg, stocks = L.aggregate_regime_dtw(eps, r)
                if avg is None:
                    continue
                if list(stocks) != list(di0["stocks"]):
                    log(f"  WARNING {v} {design} {fname} {r}: stocks differ from the primary DTW")
                s_psd, h, base = L.kernel_similarity_psd(avg.values, 0.5)
                reg_eps = [e for e in eps.values() if e['regime'] == r]
                mkt_avg = {n: pd.concat([e['market'][n]['dist'].loc[stocks] for e in reg_eps], axis=1).mean(axis=1)
                           for n in mk_raw}
                rets = c["train_regime_returns"][r][stocks]
                inputs[r] = dict(stocks=stocks, mu=rets.mean(), vol=rets.std(), avg_dtw=avg,
                                 S_tilde=pd.DataFrame(s_psd, index=stocks, columns=stocks),
                                 kernel_sigma=h, base_median=base, market_dtw=mkt_avg)
            L.save_pickle(dict(design=design, fold=c["fold"]["fold_idx"], variant=v, dtw_inputs=inputs,
                               episode_gammas=[g for g in gammas if v.startswith("sdtw") and g[0] == v.split("_")[1]],
                               gamma_rule=f"{V.GAMMA_FRAC} x median squared same-day difference (per episode)"),
                          outs[v])
        log(f"{design} {fname}: {len(tasks)} tasks done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    for d in (sys.argv[1:] or ORDER):
        run_design(d)
    log("DTW VARIANTS CACHE DONE")
