# -*- coding: utf-8 -*-
"""
Sharpe / Sortino / Treynor objectives x Standard / DTW covariance, from the Step 0 caches.

Usage:
  python run_objectives.py pilot            # Nifty 50 two-period B, 1-day (checkpoint 2)
  python run_objectives.py <design> [h ...] # any Step 0 design, given horizons (default per design)
Outputs (new files only, in experiments_v2/results/objectives/):
  <tag>_weights.csv       every weight vector (fold, regime, objective, covariance, stock, weight)
  <tag>_weight_stats.csv  non-zero count, cap hits, difference vs the Sharpe portfolio
  <tag>_betas.csv         DTW vs Standard beta comparison per fold and regime
  <tag>_metrics.csv       Sharpe/Sortino/Treynor of every portfolio, gaps, one-sided and two-sided p
  <tag>_warnings.txt      degenerate optima, binding beta constraint, PSD and look-ahead checks
  <tag>_report.txt        readable summary (also printed)
"""
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import dtwlib as L
import objectives as O

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "objectives")
BOOT_SEED = 20261021
MARKET = "NSEI"  # primary market index for every universe
PRIMARY_FAMILY = {("2p_N50_A", 1), ("2p_N50_B", 1), ("2p_BANK_A", 1), ("2p_BANK_B", 1)}
OBJECTIVES = ["sharpe", "sortino", "treynor"]


def log_variant(**row):
    path = os.path.join(HERE, "variants_log.csv")
    if os.environ.get("NO_VARIANT_LOG") == "1":  # identical rerun of an already-logged variant
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "stage", "objective", "metric", "covariances", "design",
                                          "universe", "horizon", "counted_in_family", "note"])
        if new:
            w.writeheader()
        w.writerow(dict(timestamp=datetime.now().isoformat(timespec="seconds"), **row))


def fold_paths(design):
    d = os.path.join(HERE, "cache", "step0", design)
    return sorted((os.path.join(d, f) for f in os.listdir(d) if f.startswith("fold") and f.endswith(".pkl")),
                  key=lambda p: int(os.path.basename(p)[4:-4]))


def build_fold(cache, warns, tag_fold):
    """All weight vectors for one fold, from training-only cached inputs."""
    W = {}          # (objective, cov) -> {regime: weights}
    W_rob = {}      # robustness objectives
    eval_betas = {}  # regime -> sample beta of every stock (NSEI), for the Treynor metric
    beta_rows, info_rows = [], []
    tr_end = pd.Timestamp(cache["fold"]["train_end"])
    te_start = pd.Timestamp(cache["fold"]["test_start"])
    for r, rets in cache["train_regime_returns"].items():
        # look-ahead guard: every training input ends on or before the training end date
        if rets.index.max() > tr_end or cache["train_regime_market"][r][MARKET].index.max() > tr_end:
            warns.append(f"{tag_fold} {r}: LOOK-AHEAD training data after {tr_end.date()}")
        if cache["test_returns"].index.min() < te_start:
            warns.append(f"{tag_fold}: test data starts before test_start")
        mkt = cache["train_regime_market"][r][MARKET]
        si, di = cache["std_inputs"].get(r), cache["dtw_inputs"].get(r)
        d = cache["downside_dev"][r]
        union = sorted(set(si["stocks"] if si else []) | set(di["stocks"] if di else []))
        eval_betas[r] = O.sample_betas(rets, mkt, union) if union else pd.Series(dtype=float)

        if si is not None:
            W.setdefault(("sharpe", "std"), {})[r] = cache["weights_sharpe"]["std"][r]
            re = O.optimize_ratio_cov(si["mu"], si["cov"])
            dev = float((re - cache["weights_sharpe"]["std"][r]).abs().max())
            if dev > 1e-6:
                warns.append(f"{tag_fold} {r}: Sharpe Standard weights do not reproduce the cached ones (max diff {dev:.2e})")
            Sd = O.downside_cov(d, si["corr"])
            ev = float(np.linalg.eigvalsh(Sd.values).min())
            info_rows.append(dict(fold=tag_fold, regime=r, matrix="Sigma_down_std", min_eig=ev))
            if ev < -1e-12:
                warns.append(f"{tag_fold} {r}: Sigma_down_std not PSD (min eig {ev:.2e})")
            W.setdefault(("sortino", "std"), {})[r] = O.optimize_ratio_cov(si["mu"], Sd)
            E = O.estrada_semicov(rets, si["stocks"])
            if E.isna().any().any():
                warns.append(f"{tag_fold} {r}: Estrada semicovariance has NaN entries - skipped (robustness only)")
            else:
                info_rows.append(dict(fold=tag_fold, regime=r, matrix="Estrada_std",
                                      min_eig=float(np.linalg.eigvalsh(E.values).min())))
                W_rob.setdefault(("sortino_estrada", "std"), {})[r] = O.optimize_ratio_cov(si["mu"], E)
            b_std = O.sample_betas(rets, mkt, si["stocks"])
            wt, inf = O.optimize_treynor(si["mu"], b_std)
            info_rows.append(dict(fold=tag_fold, regime=r, matrix="treynor_std", **{k: v for k, v in inf.items()}))
            if wt is not None:
                W.setdefault(("treynor", "std"), {})[r] = wt
            else:
                warns.append(f"{tag_fold} {r}: Treynor Standard infeasible (w'beta >= {O.MIN_BETA} unattainable)")
        if di is not None:
            W.setdefault(("sharpe", "dtw"), {})[r] = cache["weights_sharpe"]["dtw"][r]
            Sd = O.downside_cov(d, di["S_tilde"])
            ev = float(np.linalg.eigvalsh(Sd.values).min())
            info_rows.append(dict(fold=tag_fold, regime=r, matrix="Sigma_down_dtw", min_eig=ev))
            if ev < -1e-12:
                warns.append(f"{tag_fold} {r}: Sigma_down_dtw not PSD (min eig {ev:.2e})")
            W.setdefault(("sortino", "dtw"), {})[r] = O.optimize_ratio_cov(di["mu"], Sd)
            b_dtw, bi = O.dtw_betas(di, MARKET, mkt)
            info_rows.append(dict(fold=tag_fold, regime=r, matrix="S_ext_dtw", min_eig=bi["min_eig_ext"], S_mm=bi["S_mm"]))
            wt, inf = O.optimize_treynor(di["mu"], b_dtw)
            info_rows.append(dict(fold=tag_fold, regime=r, matrix="treynor_dtw", **{k: v for k, v in inf.items()}))
            if wt is not None:
                W.setdefault(("treynor", "dtw"), {})[r] = wt
            else:
                warns.append(f"{tag_fold} {r}: Treynor DTW infeasible (w'beta >= {O.MIN_BETA} unattainable)")
            if si is not None:
                common = [s for s in di["stocks"] if s in b_std.index]
                bs, bd = b_std.loc[common], b_dtw.loc[common]
                beta_rows.append(dict(fold=tag_fold, regime=r, n=len(common),
                                      corr=float(np.corrcoef(bs, bd)[0, 1]) if len(common) > 2 else np.nan,
                                      mean_ratio_dtw_over_std=float((bd / bs).mean()),
                                      median_ratio=float((bd / bs).median()),
                                      std_min=float(bs.min()), std_max=float(bs.max()), std_mean=float(bs.mean()),
                                      dtw_min=float(bd.min()), dtw_max=float(bd.max()), dtw_mean=float(bd.mean()),
                                      std_n_neg=int((bs < 0).sum()), dtw_n_neg=int((bd < 0).sum()),
                                      S_im_mean=float(bi["S_im"].loc[common].mean()),
                                      corr_with_mkt_mean=float((bs * mkt.std() / rets[common].std()).mean())))
    return W, W_rob, eval_betas, beta_rows, info_rows


def weight_stats(W, tag_fold):
    rows, wrows = [], []
    for (obj, cov), byreg in W.items():
        for r, w in byreg.items():
            ref = W.get(("sharpe", cov), {}).get(r)
            u = w.index.union(ref.index) if ref is not None else w.index
            dw = (w.reindex(u, fill_value=0) - ref.reindex(u, fill_value=0)).abs() if ref is not None else None
            rows.append(dict(fold=tag_fold, regime=r, objective=obj, cov=cov, n_stocks=len(w),
                             n_nonzero=int((w > O.NZ).sum()), n_at_cap=int((w >= O.MAX_W - 1e-4).sum()),
                             max_weight=float(w.max()),
                             mean_abs_diff_vs_sharpe=float(dw.mean()) if dw is not None else np.nan,
                             half_l1_diff_vs_sharpe=float(dw.sum() / 2) if dw is not None else np.nan))
            for s, v in w.items():
                if v > O.NZ:
                    wrows.append(dict(fold=tag_fold, regime=r, objective=obj, cov=cov, stock=s, weight=float(v)))
    return rows, wrows


def run(design, horizons, tag, stage):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    warns, beta_rows, info_rows, ws_rows, w_rows = [], [], [], [], []
    series = {}  # (objective, cov) -> list of (returns, betas)
    for p in fold_paths(design):
        cache = L.load_pickle(p)
        tf = f"fold{cache['fold']['fold_idx']}({str(cache['fold']['test_start'])[:4]})"
        W, W_rob, eval_betas, br, ir = build_fold(cache, warns, tf)
        beta_rows += br
        info_rows += ir
        a, b = weight_stats({**W, **W_rob}, tf)
        ws_rows += a
        w_rows += b
        for key, byreg in {**W, **W_rob}.items():
            s, bseries = O.portfolio_returns(byreg, cache, eval_betas)
            series.setdefault(key, []).append((s, bseries))
            if key == ("sharpe", "std"):
                dev = float((s - cache["std_series"]).abs().max())
                if dev > 1e-12:
                    warns.append(f"{tf}: Sharpe Standard test returns differ from Step 0 (max {dev:.2e})")
            if key == ("sharpe", "dtw"):
                dev = float((s - cache["dtw_series"]).abs().max())
                if dev > 1e-12:
                    warns.append(f"{tf}: Sharpe DTW test returns differ from Step 0 (max {dev:.2e})")

    # degenerate Treynor / binding beta constraint
    ws = pd.DataFrame(ws_rows)
    for _, r in ws[ws.objective == "treynor"].iterrows():
        if r.n_nonzero <= 3 or r.n_at_cap >= 2:
            warns.append(f"DEGENERATE Treynor optimum: {r['fold']} {r['regime']} {r['cov']}: {r['n_nonzero']} non-zero, "
                         f"{r['n_at_cap']} at the 40% cap")
    inf = pd.DataFrame(info_rows)
    if "beta_constraint_binds" in inf:
        for _, r in inf[inf.beta_constraint_binds == True].iterrows():  # noqa: E712
            warns.append(f"BETA CONSTRAINT BINDS (w'beta = {r.w_beta:.4f}): {r.fold} {r.regime} {r.matrix}")

    # pooled evaluation, all three metrics for every portfolio
    met_rows = []
    pooled = {k: (pd.concat([x[0] for x in v]).sort_index(), pd.concat([x[1] for x in v]).sort_index())
              for k, v in series.items()}
    for obj in OBJECTIVES + ["sortino_estrada"]:
        for h in horizons:
            if obj == "sortino_estrada":
                # robustness: Estrada Standard vs DTW Sortino (DTW uses D_d S~ D_d)
                s_key, d_key = ("sortino_estrada", "std"), ("sortino", "dtw")
            else:
                s_key, d_key = (obj, "std"), (obj, "dtw")
            if s_key not in pooled or d_key not in pooled:
                continue
            c = O.compare(*pooled[s_key], *pooled[d_key], h=h, seed=BOOT_SEED)
            for metric in ["sharpe", "sortino", "treynor"]:
                bt = c["boot"][metric] if c["boot"] else dict(p_one=np.nan, p_two=np.nan, ci=(np.nan, np.nan))
                met_rows.append(dict(design=design, horizon=h, objective=obj, metric=metric,
                                     matched=(metric == obj), std=c["std"][metric], dtw=c["dtw"][metric],
                                     gap=c["dtw"][metric] - c["std"][metric], p_one=bt["p_one"], p_two=bt["p_two"],
                                     ci_lo=bt["ci"][0], ci_hi=bt["ci"][1], n_obs=c.get("n_obs"), n_blocks=c.get("n_blocks"),
                                     std_avg_beta=c["std"]["avg_beta"], dtw_avg_beta=c["dtw"]["avg_beta"],
                                     primary_family=((design, h) in PRIMARY_FAMILY and obj in OBJECTIVES
                                                     and metric == obj)))
                log_variant(stage=stage, objective=obj, metric=metric, covariances="std,dtw", design=design,
                            universe=design.split("_")[1] if design.startswith("2p") else design.split("_")[-1],
                            horizon=h, counted_in_family="yes" if ((design, h) in PRIMARY_FAMILY and
                                                                   obj in OBJECTIVES and metric == obj) else "no",
                            note=("robustness: Estrada semicovariance for Standard" if obj == "sortino_estrada"
                                  else "matched" if metric == obj else "cross-evaluated"))
    met = pd.DataFrame(met_rows)

    pd.DataFrame(w_rows).to_csv(os.path.join(OUT, f"{tag}_weights.csv"), index=False)
    ws.to_csv(os.path.join(OUT, f"{tag}_weight_stats.csv"), index=False)
    pd.DataFrame(beta_rows).to_csv(os.path.join(OUT, f"{tag}_betas.csv"), index=False)
    inf.to_csv(os.path.join(OUT, f"{tag}_solver_psd_info.csv"), index=False)
    met.to_csv(os.path.join(OUT, f"{tag}_metrics.csv"), index=False)
    with open(os.path.join(OUT, f"{tag}_warnings.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(warns) if warns else "no warnings")
    L.save_pickle(dict(design=design, horizons=horizons, metrics=met, weight_stats=ws, betas=pd.DataFrame(beta_rows),
                       solver_info=inf, warnings=warns, pooled=pooled, boot_seed=BOOT_SEED, market=MARKET),
                  os.path.join(OUT, f"{tag}.pkl"))
    return met, ws, pd.DataFrame(beta_rows), inf, warns, time.time() - t0


def report(tag, met, ws, betas, inf, warns, secs):
    f2 = lambda x: f"{x:.3f}"
    lines = [f"=== {tag}: {secs:.0f}s ===", "",
             "1) BETAS, DTW vs Standard (common stocks; market ^NSEI; training data of each fold/regime)",
             betas.to_string(index=False, float_format=f2), "",
             "2) WEIGHTS (non-zero > 1e-4; difference vs the Sharpe portfolio with the same covariance)",
             ws.to_string(index=False, float_format=f2), "",
             "   average over folds and regimes:",
             ws.groupby(["objective", "cov"])[["n_stocks", "n_nonzero", "n_at_cap", "max_weight",
                                                 "mean_abs_diff_vs_sharpe", "half_l1_diff_vs_sharpe"]]
               .mean().to_string(float_format=f2), "",
             "3) WARNINGS", *(warns or ["none"]), "",
             "   Treynor solver (method used; SLSQP vs exact LP objective; w'beta):",
             inf[inf.matrix.str.startswith("treynor")][["fold", "regime", "matrix", "method", "slsqp_obj", "lp_obj",
                                                         "w_beta", "beta_constraint_binds"]]
               .to_string(index=False, float_format=lambda x: f"{x:.5f}"), "",
             "   PSD check (min eigenvalue):",
             inf[~inf.matrix.str.startswith("treynor")][["fold", "regime", "matrix", "min_eig"]]
               .to_string(index=False, float_format=lambda x: f"{x:.2e}"), "",
             "4) EVALUATION (pooled over folds; gap = DTW - Standard; p_one = P(boot diff <= 0); seed "
             f"{BOOT_SEED}; days resampled jointly)",
             met[["horizon", "objective", "metric", "matched", "std", "dtw", "gap", "p_one", "p_two",
                  "std_avg_beta", "dtw_avg_beta"]].to_string(index=False, float_format=f2)]
    text = "\n".join(lines)
    with open(os.path.join(OUT, f"{tag}_report.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    if sys.argv[1] == "pilot":
        design, horizons, tag, stage = "2p_N50_B", [1], "pilot_2p_N50_B_1d", "pilot"
    else:
        design = sys.argv[1]
        horizons = [int(x) for x in sys.argv[2:]] or ([1, 5, 22] if design.startswith("2p") else [1])
        tag, stage = f"full_{design}", "full"
    report(tag, *run(design, horizons, tag, stage))
