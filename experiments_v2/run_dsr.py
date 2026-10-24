# -*- coding: utf-8 -*-
"""
Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) for the main methods' Sharpe-objective portfolios,
1-day, pooled out-of-sample daily log returns, per design.

  SR   = daily mean / daily sd (ddof 1); skew g3, kurtosis g4 (non-excess) of the daily returns; T = days
  SR0  = sqrt(V) * ((1 - e_g) * Phi^-1(1 - 1/N) + e_g * Phi^-1(1 - 1/(N e)))   (e_g = Euler-Mascheroni)
  DSR  = Phi((SR - SR0) * sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2));  PSR(0) uses SR0 = 0
  V    = cross-sectional variance of daily SR over EVERY portfolio tried in that design (all methods x all
         objectives present in the result files), N = number of trials:
         N_log      = tests logged in variants_log.csv (excluding step0 reruns)  [as requested]
         N_strategy = distinct (method, objective, design) strategies in the log (sensitivity)
The DSR asks whether a portfolio's Sharpe exceeds what the best of N unskilled trials would show; it is
not a DTW-vs-Standard test. Writes results/dsr/dsr.csv and dsr.txt
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

import dtwlib as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "dsr")
DORDER = ["2p_N50_A", "2p_N50_B", "2p_BANK_A", "2p_BANK_B", "single_N50", "exp_N50", "exp_BANK"]
MAIN = ["std", "dtw", "dtwz", "lw", "ew", "gmv", "invvol"]
EG = 0.5772156649015329


def sr0(V, N):
    return np.sqrt(V) * ((1 - EG) * norm.ppf(1 - 1 / N) + EG * norm.ppf(1 - 1 / (N * np.e)))


def psr(sr, sr_star, T, g3, g4):
    return float(norm.cdf((sr - sr_star) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


vlog = pd.read_csv(os.path.join(HERE, "variants_log.csv"), dtype=str)
vlog = vlog[vlog.stage != "step0_regression"]
N_log = len(vlog)
meth_names = vlog.covariances.str.split(",").str[1]
N_strategy = pd.DataFrame({"m": meth_names, "o": vlog.objective, "d": vlog.design}).drop_duplicates().shape[0]

rows, cross = [], []
for d in DORDER:
    pooled = {}
    for p in glob.glob(os.path.join(HERE, "results", "objectives", f"*_{d}.pkl")):
        if os.path.basename(p).startswith(("pilot", "_")):
            continue
        for k, (r, _) in L.load_pickle(p)["pooled"].items():
            pooled.setdefault(k, r)
    srs = {k: r.mean() / r.std() for k, r in pooled.items() if len(r) > 30}
    V = float(np.var(list(srs.values()), ddof=1))
    cross.append(dict(design=d, n_portfolios=len(srs), V_daily=V, sd_annual_sr=np.sqrt(V * 252)))
    for m in MAIN:
        r = pooled.get(("sharpe", m))
        if r is None:
            continue
        sr, T, g3, g4 = r.mean() / r.std(), len(r), skew(r), kurtosis(r, fisher=False)
        row = dict(design=d, method=m, T=T, sr_ann=sr * np.sqrt(252), skew=g3, kurt=g4, psr0=psr(sr, 0, T, g3, g4))
        for tag, N in [("Nlog", N_log), ("Nstrat", N_strategy)]:
            s0 = sr0(V, N)
            row[f"sr0_ann_{tag}"] = s0 * np.sqrt(252)
            row[f"dsr_{tag}"] = psr(sr, s0, T, g3, g4)
        rows.append(row)

os.makedirs(OUT, exist_ok=True)
res, cr = pd.DataFrame(rows), pd.DataFrame(cross)
res.to_csv(os.path.join(OUT, "dsr.csv"), index=False)
txt = (f"N_log (tests in variants_log.csv excl. step0 reruns) = {N_log}; N_strategy (distinct method x objective x design) = {N_strategy}\n\n"
       "Cross-sectional dispersion of daily Sharpe across all portfolios tried, per design:\n"
       + cr.to_string(index=False, float_format=lambda x: f"{x:.5f}") + "\n\n"
       + res.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")
open(os.path.join(OUT, "dsr.txt"), "w", encoding="utf-8").write(txt)
print(txt)
