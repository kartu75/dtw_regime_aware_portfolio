# -*- coding: utf-8 -*-
"""Checkpoint 1 report: regression of the 16 original Sharpe tests against the snapshot rerun.
Tolerances (agreed 2026-10-21): |Standard Sharpe diff| <= 0.001, |DTW Sharpe diff| <= 0.002.
Writes results/checkpoint1_regression.csv and results/checkpoint1_report.txt."""
import os

import numpy as np
import pandas as pd

import dtwlib as L
from run_step0 import DESIGNS, load_original, BOOT_SEED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOL_STD, TOL_DTW = 0.001, 0.002
ORDER = [("single_N50", 1), ("exp_N50", 1), ("exp_BANK", 1), ("exp_MID", 1)] + \
        [(d, h) for d in ["2p_N50_A", "2p_N50_B", "2p_BANK_A", "2p_BANK_B"] for h in (1, 5, 22)]

rows, timing = [], []
for test_no, (design, h) in enumerate(ORDER, start=1):
    if design == "exp_MID":
        d = L.load_pickle(os.path.join(ROOT, "expanding_window_midcap_results.pkl"))
        rows.append(dict(test=test_no, design=design, horizon=h, std_orig=d['pooled_std_metrics']['sharpe'],
                         std_new=np.nan, std_absdiff=np.nan, dtw_orig=d['pooled_dtw_metrics']['sharpe'],
                         dtw_new=np.nan, dtw_absdiff=np.nan, p_one_orig=d['bootstrap']['p_value'], p_one_new=np.nan,
                         fold_std_maxabs=np.nan, fold_dtw_maxabs=np.nan,
                         status="NOT REPRODUCED (Nifty Midcap 100 index no longer on Yahoo); original value kept"))
        continue
    res = L.load_pickle(os.path.join(HERE, "results", f"step0_{design}.pkl"))
    orig_folds, orig_hz = load_original(DESIGNS[design])
    r, o = res["horizon_results"][h], orig_hz[h]
    fs = max(float((rec['std_series'] - os_).abs().max()) for rec, (os_, od) in zip(res["records"], orig_folds))
    fd = max(float((rec['dtw_series'] - od).abs().max()) for rec, (os_, od) in zip(res["records"], orig_folds))
    ds, dd = abs(r['std_sharpe'] - o['std']), abs(r['dtw_sharpe'] - o['dtw'])
    ok = ds <= TOL_STD and dd <= TOL_DTW
    rows.append(dict(test=test_no, design=design, horizon=h, std_orig=o['std'], std_new=r['std_sharpe'], std_absdiff=ds,
                     dtw_orig=o['dtw'], dtw_new=r['dtw_sharpe'], dtw_absdiff=dd, p_one_orig=o['p'], p_one_new=r['p_one'],
                     fold_std_maxabs=fs, fold_dtw_maxabs=fd, status="PASS" if ok else "FAIL"))
    if h == 1:
        for rec in res["records"]:
            timing.append(dict(design=design, fold=rec['fold_idx'], test_year=str(rec['test_start'])[:4],
                               minutes=rec['elapsed_sec'] / 60, dtw_seconds=rec['dtw_wall_time'],
                               train_days=rec['train_days'], n_episodes=rec['n_train_episodes']))

tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(HERE, "results", "checkpoint1_regression.csv"), index=False)
tim = pd.DataFrame(timing)
tim.to_csv(os.path.join(HERE, "results", "checkpoint1_fold_timing.csv"), index=False)

with open(os.path.join(HERE, "results", "checkpoint1_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Checkpoint 1 regression (bootstrap seed {BOOT_SEED}; tolerances Std {TOL_STD}, DTW {TOL_DTW})\n\n")
    f.write(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n\n")
    f.write(tim.groupby("design")[["minutes", "dtw_seconds"]].agg(["mean", "max", "sum"]).round(2).to_string() + "\n")
print(tab[["test", "design", "horizon", "std_orig", "std_new", "std_absdiff", "dtw_orig", "dtw_new", "dtw_absdiff",
           "p_one_orig", "p_one_new", "status"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()
print(tim.groupby("design")[["minutes", "dtw_seconds"]].agg(["mean", "max", "sum"]).round(2).to_string())
