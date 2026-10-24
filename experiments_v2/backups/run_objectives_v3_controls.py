# -*- coding: utf-8 -*-
"""
Objectives x covariance methods on the Step 0 caches (v2; pilot version kept in backups/).

Usage:
  python run_objectives.py full [design ...]   # primary full run, methods std + dtw
  python run_objectives.py pilot               # Nifty 50 two-period B, 1-day (checkpoint 2)
Each design writes results/objectives/full_<design>.pkl (+ csv/txt); designs already done are skipped.

Objectives per method (training data of fold k, regime R; rf = MAR = 0):
  sharpe   D_s C D_s                          (Standard and DTW weights reused from the Step 0 cache)
  sortino  D_d C D_d
  treynor  w'mu / w'beta, beta from the method's extended (N+1) matrix, market ^NSEI   [side result]
Robustness (labelled):
  sortino_estrada_clip  Standard uses the Estrada semicovariance, eigenvalue-clipped like S_tilde
  treynor_cap10         Treynor with a 10% single-name cap
  treynor_ownidx        Treynor with the universe's own index as market (Bank: ^NSEBANK)
Evaluation: Sharpe, Sortino, Treynor (training-period sample beta, same market as the objective),
realised volatility, average training beta; paired block bootstrap (5000, block 20, seed 20261021);
daily DTW-minus-Standard regression on the market return (alpha, beta, HAC alpha p-value).
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
PRIMARY_FAMILY = {("2p_N50_A", 1), ("2p_N50_B", 1), ("2p_BANK_A", 1), ("2p_BANK_B", 1)}
DESIGN_ORDER = ["2p_N50_A", "2p_N50_B", "2p_BANK_A", "2p_BANK_B", "single_N50", "exp_N50", "exp_BANK"]
OWN_INDEX = {"2p_BANK_A": "NSEBANK", "2p_BANK_B": "NSEBANK", "exp_BANK": "NSEBANK"}
ROLE = {"sharpe": "primary", "sortino": "primary", "treynor": "side (compares beta estimates only)",
        "sortino_estrada_clip": "robustness", "treynor_cap10": "robustness (side)",
        "treynor_ownidx": "robustness (side)"}
FAMILY_OBJECTIVES = ("sharpe", "sortino", "treynor")
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
_logf = open(os.path.join(HERE, "logs", "objectives.log"), "a", encoding="utf-8")


def log(*a):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " + " ".join(map(str, a))
    print(line)
    sys.stdout.flush()
    _logf.write(line + "\n")
    _logf.flush()


# ------------------------------------------------------------------ variant log (deduplicated)

VLOG = os.path.join(HERE, "variants_log.csv")
VFIELDS = ["timestamp", "stage", "objective", "metric", "covariances", "design", "universe", "horizon",
           "counted_in_family", "note"]


def log_variant(**row):
    if os.environ.get("NO_VARIANT_LOG") == "1":
        return
    key = (row["objective"], row["metric"], row["covariances"], row["design"], str(row["horizon"]))
    if os.path.exists(VLOG):
        old = pd.read_csv(VLOG, dtype=str)
        old = old[old["stage"] != "step0_regression"]  # reruns of existing Sharpe results do not count as tests
        if not old.empty and ((old["objective"] == key[0]) & (old["metric"] == key[1]) & (old["covariances"] == key[2])
                              & (old["design"] == key[3]) & (old["horizon"] == key[4])).any():
            return  # this exact test was already logged (e.g. in the pilot); do not count it twice
    new = not os.path.exists(VLOG)
    with open(VLOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VFIELDS)
        if new:
            w.writeheader()
        w.writerow(dict(timestamp=datetime.now().isoformat(timespec="seconds"), **row))


def universe_of(design):
    return design.split("_")[1] if design.startswith("2p") else design.split("_")[-1]


def fold_paths(design):
    d = os.path.join(HERE, "cache", "step0", design)
    return sorted((os.path.join(d, f) for f in os.listdir(d) if f.startswith("fold") and f.endswith(".pkl")),
                  key=lambda p: int(os.path.basename(p)[4:-4]))


# ------------------------------------------------------------------ covariance methods

_DTWZ = {}


def dtwz_inputs(cache, r):
    """Control (a): return-normalised DTW inputs from cache/dtwz (built by run_dtwz_cache.py)."""
    key = (cache["design"], cache["fold"]["fold_idx"])
    if key not in _DTWZ:
        _DTWZ[key] = L.load_pickle(os.path.join(HERE, "cache", "dtwz", key[0], f"fold{key[1]}.pkl"))["dtw_inputs"]
    return _DTWZ[key].get(r)


def method_inputs(cache, r, method):
    """(stocks, mu, C, vol) for a method in regime r, or None.
    std/dtw: primary. dtwz, volk, invvol: controls (a), (b), (c)."""
    if method == "std":
        si = cache["std_inputs"].get(r)
        return None if si is None else dict(stocks=si["stocks"], mu=si["mu"], C=si["corr"], vol=si["vol"])
    if method == "dtw":
        di = cache["dtw_inputs"].get(r)
        return None if di is None else dict(stocks=di["stocks"], mu=di["mu"], C=di["S_tilde"], vol=di["vol"])
    if method == "dtwz":
        di = dtwz_inputs(cache, r)
        return None if di is None else dict(stocks=di["stocks"], mu=di["mu"], C=di["S_tilde"], vol=di["vol"])
    if method == "volk":
        di = cache["dtw_inputs"].get(r)  # same stocks, means and volatilities as DTW
        if di is None:
            return None
        vol = di["vol"].loc[di["stocks"]]
        return dict(stocks=di["stocks"], mu=di["mu"], C=O.vol_kernel(vol)[0], vol=vol)
    if method == "invvol":
        si = cache["std_inputs"].get(r)  # every stock eligible for the Standard portfolio
        return None if si is None else dict(stocks=si["stocks"], mu=si["mu"], C=None, vol=si["vol"])
    raise KeyError(method)


def method_betas(cache, r, method, mkt_name):
    rets = cache["train_regime_returns"][r]
    mkt = cache["train_regime_market"][r][mkt_name]
    if method == "std":
        return O.sample_betas(rets, mkt, cache["std_inputs"][r]["stocks"]), {}
    if method == "dtw":
        return O.dtw_betas(cache["dtw_inputs"][r], mkt_name, mkt)
    if method == "dtwz":
        return O.dtw_betas(dtwz_inputs(cache, r), mkt_name, mkt)
    if method == "volk":
        di = cache["dtw_inputs"][r]
        return O.vol_kernel(di["vol"].loc[di["stocks"]], market_vol=float(np.nanstd(mkt.values, ddof=1)))
    raise KeyError(method)


def build_fold(cache, tf, methods, markets, warns):
    W, beta_rows, info_rows = {}, [], []
    eval_betas = {m: {} for m in markets}
    tr_end = pd.Timestamp(cache["fold"]["train_end"])
    for r, rets in cache["train_regime_returns"].items():
        for m in markets:
            if rets.index.max() > tr_end or cache["train_regime_market"][r][m].index.max() > tr_end:
                warns.append(f"{tf} {r}: LOOK-AHEAD training data after {tr_end.date()}")
        d = cache["downside_dev"][r]
        inputs = {meth: method_inputs(cache, r, meth) for meth in methods}
        union = sorted(set().union(*[set(x["stocks"]) for x in inputs.values() if x is not None]))
        for m in markets:
            eval_betas[m][r] = O.sample_betas(rets, cache["train_regime_market"][r][m], union)
        betas_by = {}
        for meth, inp in inputs.items():
            if inp is None:
                continue
            if meth == "invvol":
                # objective-independent baseline: the same weights under every objective
                w_iv = O.inverse_vol_weights(inp["vol"].loc[inp["stocks"]])
                for obj in ["sharpe", "sortino", "treynor", "treynor_cap10"] + (["treynor_ownidx"] if len(markets) > 1 else []):
                    W.setdefault((obj, meth), {})[r] = w_iv
                continue
            # Sharpe
            if meth in ("std", "dtw"):
                W.setdefault(("sharpe", meth), {})[r] = cache["weights_sharpe"][meth][r]
                if meth == "std":
                    dev = float((O.optimize_ratio_cov(inp["mu"], cache["std_inputs"][r]["cov"])
                                 - cache["weights_sharpe"]["std"][r]).abs().max())
                    if dev > 1e-6:
                        warns.append(f"{tf} {r}: Sharpe Standard weights differ from cache ({dev:.2e})")
            else:
                vol = inp["vol"].loc[inp["stocks"]].values
                W.setdefault(("sharpe", meth), {})[r] = O.optimize_ratio_cov(
                    inp["mu"], pd.DataFrame(inp["C"].values * np.outer(vol, vol), index=inp["stocks"], columns=inp["stocks"]))
            # Sortino
            Sd = O.downside_cov(d, inp["C"])
            ev = float(np.linalg.eigvalsh(Sd.values).min())
            info_rows.append(dict(fold=tf, regime=r, item=f"Sigma_down_{meth}", min_eig=ev))
            if ev < -1e-12:
                warns.append(f"{tf} {r}: Sigma_down_{meth} not PSD (min eig {ev:.2e})")
            W.setdefault(("sortino", meth), {})[r] = O.optimize_ratio_cov(inp["mu"], Sd)
            # Treynor (primary market NSEI, 40% and 10% caps; own index as robustness)
            for m in markets:
                b, bi = method_betas(cache, r, meth, m)
                betas_by[(meth, m)] = b
                if bi:
                    info_rows.append(dict(fold=tf, regime=r, item=f"S_ext_{meth}_{m}", min_eig=bi["min_eig_ext"]))
                variants = [("treynor", O.MAX_W)] + ([("treynor_cap10", 0.10)] if m == "NSEI" else [])
                for name, cap in variants:
                    key = name if m == "NSEI" else "treynor_ownidx"
                    if m != "NSEI" and name != "treynor":
                        continue
                    wt, inf = O.optimize_treynor(inp["mu"], b, max_w=cap)
                    info_rows.append(dict(fold=tf, regime=r, item=f"{key}_{meth}", **inf))
                    if wt is None:
                        warns.append(f"{tf} {r}: {key} {meth} infeasible")
                    else:
                        W.setdefault((key, meth), {})[r] = wt
        # Estrada (Standard only, clipped)
        if inputs.get("std") is not None:
            E = O.estrada_semicov(rets, inputs["std"]["stocks"])
            if E.isna().any().any():
                warns.append(f"{tf} {r}: Estrada has NaN entries - skipped")
            else:
                ev0 = float(np.linalg.eigvalsh(E.values).min())
                Ec = O.clip_psd(E)
                info_rows.append(dict(fold=tf, regime=r, item="Estrada_std_raw", min_eig=ev0))
                info_rows.append(dict(fold=tf, regime=r, item="Estrada_std_clipped", min_eig=float(np.linalg.eigvalsh(Ec.values).min())))
                W.setdefault(("sortino_estrada_clip", "std"), {})[r] = O.optimize_ratio_cov(inputs["std"]["mu"], Ec)
        # beta diagnostic: each non-standard method vs Standard
        for (meth, m), b in betas_by.items():
            if meth == "std" or ("std", m) not in betas_by:
                continue
            bs = betas_by[("std", m)]
            common = [s for s in b.index if s in bs.index]
            x, y = bs.loc[common], b.loc[common]
            beta_rows.append(dict(fold=tf, regime=r, method=meth, market=m, n=len(common),
                                  corr=float(np.corrcoef(x, y)[0, 1]) if len(common) > 2 else np.nan,
                                  mean_ratio=float((y / x).mean()), median_ratio=float((y / x).median()),
                                  std_min=float(x.min()), std_max=float(x.max()), std_mean=float(x.mean()),
                                  meth_min=float(y.min()), meth_max=float(y.max()), meth_mean=float(y.mean()),
                                  std_n_neg=int((x < 0).sum()), meth_n_neg=int((y < 0).sum())))
    return W, eval_betas, beta_rows, info_rows


def cap_of(obj):
    return 0.10 if obj == "treynor_cap10" else O.MAX_W


def weight_stats(W, tf):
    rows, wrows = [], []
    for (obj, meth), byreg in W.items():
        for r, w in byreg.items():
            ref = W.get(("sharpe", meth), {}).get(r)
            u = w.index.union(ref.index) if ref is not None else w.index
            dw = (w.reindex(u, fill_value=0) - ref.reindex(u, fill_value=0)).abs() if ref is not None else None
            cap = cap_of(obj)
            rows.append(dict(fold=tf, regime=r, objective=obj, method=meth, n_stocks=len(w),
                             n_nonzero=int((w > O.NZ).sum()), n_at_cap=int((w >= cap - 1e-4).sum()),
                             cap=cap, max_weight=float(w.max()),
                             mean_abs_diff_vs_sharpe=float(dw.mean()) if dw is not None else np.nan,
                             half_l1_diff_vs_sharpe=float(dw.sum() / 2) if dw is not None else np.nan))
            wrows += [dict(fold=tf, regime=r, objective=obj, method=meth, stock=s, weight=float(v))
                      for s, v in w.items() if v > O.NZ]
    return rows, wrows


def comparison_pairs(W_keys, methods, own):
    """(objective label, standard key, method key, market for evaluation)"""
    pairs = []
    for meth in methods:
        if meth == "std":
            continue
        for obj in ["sharpe", "sortino", "treynor", "treynor_cap10"]:
            pairs.append((obj, meth, (obj, "std"), (obj, meth), "NSEI"))
        pairs.append(("sortino_estrada_clip", meth, ("sortino_estrada_clip", "std"), ("sortino", meth), "NSEI"))
        if own:
            pairs.append(("treynor_ownidx", meth, ("treynor_ownidx", "std"), ("treynor_ownidx", meth), own))
    return [p for p in pairs if p[2] in W_keys and p[3] in W_keys]


def run(design, horizons, tag, stage, methods=("std", "dtw")):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    own = OWN_INDEX.get(design)
    markets = ["NSEI"] + ([own] if own else [])
    warns, beta_rows, info_rows, ws_rows, w_rows = [], [], [], [], []
    series, mkt_series = {}, {m: [] for m in markets}
    for p in fold_paths(design):
        cache = L.load_pickle(p)
        tf = f"fold{cache['fold']['fold_idx']}({str(cache['fold']['test_start'])[:4]})"
        W, eval_betas, br, ir = build_fold(cache, tf, methods, markets, warns)
        beta_rows += br
        info_rows += ir
        a, b = weight_stats(W, tf)
        ws_rows += a
        w_rows += b
        for m in markets:
            mkt_series[m].append(cache["test_market"][m])
        for key, byreg in W.items():
            mk = own if key[0] == "treynor_ownidx" else "NSEI"
            s, bser = O.portfolio_returns(byreg, cache, eval_betas[mk])
            series.setdefault(key, []).append((s, bser))
            if key in (("sharpe", "std"), ("sharpe", "dtw")):
                ref = cache["std_series"] if key[1] == "std" else cache["dtw_series"]
                dev = float((s - ref).abs().max())
                if dev > 1e-12:
                    warns.append(f"{tf}: Sharpe {key[1]} test returns differ from Step 0 ({dev:.2e})")
        log(f"  {design} {tf} done")

    ws = pd.DataFrame(ws_rows)
    for _, r in ws[ws.objective.str.startswith("treynor")].iterrows():
        if r["n_nonzero"] <= max(3, int(np.ceil(1 / r["cap"]))) or r["n_at_cap"] >= 2:
            warns.append(f"DEGENERATE {r['objective']} optimum: {r['fold']} {r['regime']} {r['method']}: "
                         f"{r['n_nonzero']} non-zero, {r['n_at_cap']} at the {r['cap']:.0%} cap")
    inf = pd.DataFrame(info_rows)
    if "beta_constraint_binds" in inf:
        for _, r in inf[inf["beta_constraint_binds"] == True].iterrows():  # noqa: E712
            warns.append(f"BETA CONSTRAINT BINDS (w'beta = {r['w_beta']:.4f}): {r['fold']} {r['regime']} {r['item']}")

    pooled = {k: (pd.concat([x[0] for x in v]).sort_index(), pd.concat([x[1] for x in v]).sort_index())
              for k, v in series.items()}
    mkt_pooled = {m: pd.concat(v).sort_index() for m, v in mkt_series.items()}
    met_rows, reg_rows = [], []
    for obj, meth, skey, mkey, mk in comparison_pairs(set(pooled), methods, own):
        s_ret, m_ret = pooled[skey][0], pooled[mkey][0]
        common = s_ret.index.intersection(m_ret.index)
        reg = O.alpha_regression(m_ret.loc[common] - s_ret.loc[common], mkt_pooled[mk])
        reg_rows.append(dict(design=design, objective=obj, method=meth, market=mk, role=ROLE[obj], **reg))
        for h in horizons:
            c = O.compare(*pooled[skey], *pooled[mkey], h=h, seed=BOOT_SEED)
            for metric in ["sharpe", "sortino", "treynor"]:
                bt = c["boot"][metric] if c["boot"] else dict(p_one=np.nan, p_two=np.nan, ci=(np.nan, np.nan))
                fam = (design, h) in PRIMARY_FAMILY and obj in FAMILY_OBJECTIVES and metric == obj and meth == "dtw"
                met_rows.append(dict(design=design, universe=universe_of(design), horizon=h, objective=obj,
                                     method=meth, metric=metric, matched=(metric == obj.split("_")[0]),
                                     role=ROLE[obj], primary_family=fam, market=mk,
                                     std=c["std"][metric], alt=c["dtw"][metric], gap=c["dtw"][metric] - c["std"][metric],
                                     p_one=bt["p_one"], p_two=bt["p_two"], ci_lo=bt["ci"][0], ci_hi=bt["ci"][1],
                                     n_obs=c.get("n_obs"), n_blocks=c.get("n_blocks"),
                                     std_ann_vol=c["std"]["ann_vol"], alt_ann_vol=c["dtw"]["ann_vol"],
                                     std_avg_beta=c["std"]["avg_beta"], alt_avg_beta=c["dtw"]["avg_beta"]))
                log_variant(stage=stage, objective=obj, metric=metric, covariances=f"std,{meth}", design=design,
                            universe=universe_of(design), horizon=h, counted_in_family="yes" if fam else "no",
                            note=f"{ROLE[obj]}; {'matched' if metric == obj.split('_')[0] else 'cross-evaluated'}")
    met, reg = pd.DataFrame(met_rows), pd.DataFrame(reg_rows)
    for name, df in [("weights", pd.DataFrame(w_rows)), ("weight_stats", ws), ("betas", pd.DataFrame(beta_rows)),
                     ("solver_psd_info", inf), ("metrics", met), ("alpha_regression", reg)]:
        df.to_csv(os.path.join(OUT, f"{tag}_{name}.csv"), index=False)
    with open(os.path.join(OUT, f"{tag}_warnings.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(warns) if warns else "no warnings")
    L.save_pickle(dict(design=design, horizons=horizons, methods=methods, metrics=met, regression=reg,
                       weight_stats=ws, betas=pd.DataFrame(beta_rows), solver_info=inf, warnings=warns,
                       pooled=pooled, market_pooled=mkt_pooled, boot_seed=BOOT_SEED),
                  os.path.join(OUT, f"{tag}.pkl"))
    log(f"{design}: done in {time.time() - t0:.0f}s, {len(warns)} warnings")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pilot":
        run("2p_N50_B", [1], "pilot_v2_2p_N50_B_1d", "pilot")
    elif mode == "full":
        for design in (sys.argv[2:] or DESIGN_ORDER):
            tag = f"full_{design}"
            if os.path.exists(os.path.join(OUT, f"{tag}.pkl")):
                log(f"{design}: already done, skipping")
                continue
            run(design, [1, 5, 22] if design.startswith("2p") else [1], tag, "full")
        log("FULL RUN DONE")
    elif mode == "controls":
        # controls (a) dtwz, (b) volk, (c) invvol, each vs Standard; new labelled variants
        for design in (sys.argv[2:] or DESIGN_ORDER):
            tag = f"ctrl_{design}"
            if os.path.exists(os.path.join(OUT, f"{tag}.pkl")):
                log(f"{design}: controls already done, skipping")
                continue
            run(design, [1, 5, 22] if design.startswith("2p") else [1], tag, "controls",
                methods=("std", "dtwz", "volk", "invvol"))
        log("CONTROLS DONE")
