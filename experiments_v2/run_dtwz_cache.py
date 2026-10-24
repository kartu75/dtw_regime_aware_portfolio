# -*- coding: utf-8 -*-
"""
Control (a): return-normalised DTW. For each Step 0 fold, z-score every stock's daily log returns
(and the market's) with that series' mean and standard deviation over the fold's TRAINING window
only, then run the same FastDTW per episode, the same episode averaging over common stocks, the same
kernel rule (h = 0.5 x median positive distance), eigenvalue clipping and D_sigma scaling with the raw
regime volatilities. Episodes, stock inclusion, means and volatilities are the Step 0 ones.
Writes cache/dtwz/<design>/fold<k>.pkl (resumable).
Usage: python run_dtwz_cache.py [design ...]
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import dtwlib as L
from run_step0 import DESIGNS

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))
ORDER = ["2p_BANK_A", "2p_BANK_B", "exp_BANK", "single_N50", "exp_N50", "2p_N50_B", "2p_N50_A"]
_logf = open(os.path.join(HERE, "logs", "dtwz.log"), "a", encoding="utf-8")


def log(*a):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " + " ".join(map(str, a))
    print(line)
    sys.stdout.flush()
    _logf.write(line + "\n")
    _logf.flush()


def zscore_train(df, tr_start, tr_end):
    tr = df.loc[tr_start:tr_end]
    return (df - tr.mean()) / tr.std()


def run_design(design):
    spec = DESIGNS[design]
    prices = L.load_frame(spec["stocks"])
    log_returns = np.log(prices / prices.shift(1))
    markets = {name: np.log(L.load_index(k) / L.load_index(k).shift(1)) for name, k in spec["markets"].items()}
    src = os.path.join(HERE, "cache", "step0", design)
    for fname in sorted(os.listdir(src), key=lambda f: int(f[4:-4])):
        out = os.path.join(HERE, "cache", "dtwz", design, fname)
        if os.path.exists(out):
            log(f"{design} {fname}: exists, skipping")
            continue
        t0 = time.time()
        c = L.load_pickle(os.path.join(src, fname))
        tr_start, tr_end = c["fold"]["train_start"], c["fold"]["train_end"]
        z = zscore_train(log_returns, tr_start, tr_end)
        zm = {k: (s - s.loc[tr_start:tr_end].mean()) / s.loc[tr_start:tr_end].std() for k, s in markets.items()}
        eps = c["episodes"]
        tasks = [(idx, z.loc[ep['start']:ep['end']], ep['regime'],
                  {k: s.loc[ep['start']:ep['end']] for k, s in zm.items()}, "fastdtw_l1")
                 for idx, ep in eps.iterrows()]
        res = {}
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(tasks)))) as ex:
            futs = {ex.submit(L.dtw_episode_worker, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                ridx, regime, mat, stocks, mkt, _ = fut.result()
                if mat is not None:
                    res[ridx] = {'regime': regime, 'dtw_matrix': mat, 'stocks': stocks, 'market': mkt}
        res = dict(sorted(res.items()))
        inputs = {}
        for r, di0 in c["dtw_inputs"].items():
            avg, stocks = L.aggregate_regime_dtw(res, r)
            if avg is None:
                continue
            if list(stocks) != list(di0["stocks"]):
                log(f"  WARNING {design} {fname} {r}: stock set differs from Step 0 DTW ({len(stocks)} vs {len(di0['stocks'])})")
            s_psd, h, base_med = L.kernel_similarity_psd(avg.values, 0.5)
            reg_eps = [v for v in res.values() if v['regime'] == r]
            mkt_avg = {k: pd.concat([e['market'][k]['dist'].loc[stocks] for e in reg_eps], axis=1).mean(axis=1)
                       for k in zm}
            rets = c["train_regime_returns"][r][stocks]
            inputs[r] = dict(stocks=stocks, mu=rets.mean(), vol=rets.std(), avg_dtw=avg,
                             S_tilde=pd.DataFrame(s_psd, index=stocks, columns=stocks),
                             kernel_sigma=h, base_median=base_med, market_dtw=mkt_avg)
        L.save_pickle(dict(design=design, fold=c["fold"]["fold_idx"], dtw_inputs=inputs,
                           zscore="per stock, mean/std over the fold's training window", distance="fastdtw_l1"), out)
        log(f"{design} {fname}: done in {time.time() - t0:.0f}s, regimes {list(inputs)}")


if __name__ == "__main__":
    for d in (sys.argv[1:] or ORDER):
        run_design(d)
    log("DTWZ CACHE DONE")
