# -*- coding: utf-8 -*-
"""
Step 0 regression run: rebuild every existing Sharpe result from the frozen snapshot with
the shared module, write per-fold caches, and compare against the original result pickles.

Resumable: a fold whose cache file exists is loaded instead of recomputed.
Usage: python run_step0.py [design ...]      (default: all reproducible designs)
Outputs (all new files inside experiments_v2/):
  cache/step0/<design>/fold<k>.pkl     per-fold caches (training-only inputs, weights, test data)
  results/step0_<design>.pkl           pooled results per design
  results/step0_regression.csv         new vs original, per test and horizon
  logs/step0.log                       progress log
"""
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import dtwlib as L

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # original project folder (read only)
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))
BOOT_SEED = 20261021

FOLDS_EXP = [(f"2015-01-01", f"{y-1}-12-31", f"{y}-01-01", f"{y}-12-31") for y in range(2020, 2026)]
FOLDS_A = [("2008-01-01", f"{y-1}-12-31", f"{y}-01-01", f"{y}-12-31") for y in range(2013, 2020)]
FOLDS_B = [("2020-01-01", f"{y-1}-12-31", f"{y}-01-01", f"{y}-12-31") for y in range(2021, 2026)]
FOLDS_SINGLE = [("2015-01-01", "2021-12-31", "2022-01-01", "2026-08-09")]

DESIGNS = {
    # name: stocks frame, regime index, market series for later Treynor work, folds, kind, nan check, original pickle
    "single_N50": dict(stocks="stocks_N50_2015", regime_idx="index_NSEI_2015", markets={"NSEI": "index_NSEI_2015"},
                       folds=FOLDS_SINGLE, kind="single", nan_check=False, orig="dtw_baseline_checkpoint_v3.pkl"),
    "exp_N50": dict(stocks="stocks_N50_2015", regime_idx="index_NSEI_2015", markets={"NSEI": "index_NSEI_2015"},
                    folds=FOLDS_EXP, kind="expanding", nan_check=False, orig="expanding_window_results.pkl"),
    "exp_BANK": dict(stocks="stocks_BANK_2015", regime_idx="index_NSEBANK_2015",
                     markets={"NSEI": "index_NSEI_2015", "NSEBANK": "index_NSEBANK_2015"},
                     folds=FOLDS_EXP, kind="expanding", nan_check=False, orig="expanding_window_bank_results.pkl"),
    "2p_N50_A": dict(stocks="stocks_N50_A", regime_idx="index_NSEI_2008", markets={"NSEI": "index_NSEI_2008"},
                     folds=FOLDS_A, kind="twoperiod", nan_check=True, orig=("two_period_results.pkl", "period_a")),
    "2p_N50_B": dict(stocks="stocks_N50_B", regime_idx="index_NSEI_2020", markets={"NSEI": "index_NSEI_2020"},
                     folds=FOLDS_B, kind="twoperiod", nan_check=True, strict="2020-01-01",
                     orig=("two_period_results.pkl", "period_b")),
    "2p_BANK_A": dict(stocks="stocks_BANK_A", regime_idx="index_NSEBANK_2008",
                      markets={"NSEI": "index_NSEI_2008", "NSEBANK": "index_NSEBANK_2008"},
                      folds=FOLDS_A, kind="twoperiod", nan_check=True, orig=("two_period_bank_results.pkl", "period_a")),
    "2p_BANK_B": dict(stocks="stocks_BANK_B", regime_idx="index_NSEBANK_2020",
                      markets={"NSEI": "index_NSEI_2020", "NSEBANK": "index_NSEBANK_2020"},
                      folds=FOLDS_B, kind="twoperiod", nan_check=True, strict="2020-01-01",
                      orig=("two_period_bank_results.pkl", "period_b")),
}
# exp_MID is not reproducible: NIFTY_MIDCAP_100.NS (its regime index) is no longer served by Yahoo.
ORDER = ["2p_BANK_A", "2p_BANK_B", "exp_BANK", "single_N50", "exp_N50", "2p_N50_B", "2p_N50_A"]

os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
_logf = open(os.path.join(HERE, "logs", "step0.log"), "a", encoding="utf-8")
_t0 = time.time()


def log(*a):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')} | +{(time.time() - _t0) / 60:6.1f}min] " + " ".join(map(str, a))
    print(line)
    sys.stdout.flush()
    _logf.write(line + "\n")
    _logf.flush()


def log_variant(**row):
    path = os.path.join(HERE, "variants_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "stage", "objective", "metric", "covariances", "design",
                                          "universe", "horizon", "counted_in_family", "note"])
        if new:
            w.writeheader()
        w.writerow(dict(timestamp=datetime.now().isoformat(timespec="seconds"), **row))


def evaluate(design, spec, records):
    """Pool folds and test exactly as the original script for this design kind did."""
    rng = np.random.RandomState(BOOT_SEED)
    pooled_std = pd.concat([r['std_series'] for r in records]).sort_index()
    pooled_dtw = pd.concat([r['dtw_series'] for r in records]).sort_index()
    out = {}
    if spec["kind"] == "twoperiod":
        idx = pooled_std.index.intersection(pooled_dtw.index)
        s, d = pooled_std.loc[idx], pooled_dtw.loc[idx]
        horizons = [1, 5, 22]
    else:
        s, d = pooled_std, pooled_dtw  # expanding/single: metrics on the unintersected series
        horizons = [1]
    for h in horizons:
        ppy = 252 / h
        sh, dh = L.aggregate_to_horizon(s, h), L.aggregate_to_horizon(d, h)
        sm, dm = L.compute_metrics_dict(sh, ppy), L.compute_metrics_dict(dh, ppy)
        obs, p, lo, hi, _, n, nb = L.bootstrap_sharpe_diff(dh, sh, block_size=20, periods_per_year=ppy, rng=rng)
        out[h] = dict(std_sharpe=sm['sharpe'], dtw_sharpe=dm['sharpe'], gap=dm['sharpe'] - sm['sharpe'],
                      boot_diff=obs, p_one=p, p_two=L.two_sided(p), ci=(lo, hi), n_obs=n, n_blocks=nb)
    return out, pooled_std, pooled_dtw


def load_original(spec):
    o = spec["orig"]
    if isinstance(o, tuple):
        d = L.load_pickle(os.path.join(ROOT, o[0]))[o[1]]
        folds = [(f['std_series'], f['dtw_series']) for f in d['fold_results']]
        hz = {h: dict(std=v['std_metrics']['sharpe'], dtw=v['dtw_metrics']['sharpe'], p=v['bootstrap']['p_value'])
              for h, v in d['horizon_results'].items()}
    elif o.startswith("expanding"):
        d = L.load_pickle(os.path.join(ROOT, o))
        folds = [(f['std_series'], f['dtw_series']) for f in d['fold_results']]
        hz = {1: dict(std=d['pooled_std_metrics']['sharpe'], dtw=d['pooled_dtw_metrics']['sharpe'],
                      p=d['bootstrap']['p_value'])}
    else:
        d = L.load_pickle(os.path.join(ROOT, o))
        s, dd = d['portfolio_daily_returns_causal'], d['portfolio_daily_returns_dtw']
        folds = [(s, dd)]
        hz = {1: dict(std=L.compute_metrics_dict(s)['sharpe'], dtw=L.compute_metrics_dict(dd)['sharpe'],
                      p=d['significance_test']['p_value'])}
    return folds, hz


def series_diff(new, old):
    idx = new.index.union(old.index)
    a, b = new.reindex(idx), old.reindex(idx)
    both = a.notna() & b.notna()
    return dict(max_abs=float((a[both] - b[both]).abs().max()) if both.any() else np.nan,
                n_new=int(new.size), n_old=int(old.size), n_only_one=int((a.notna() ^ b.notna()).sum()))


def run_design(design):
    spec = DESIGNS[design]
    prices = L.load_frame(spec["stocks"])
    index_price = L.load_index(spec["regime_idx"])
    if spec.get("strict"):
        first = min(prices.first_valid_index(), index_price.first_valid_index())
        assert first >= pd.Timestamp(spec["strict"]), f"strict separation violated: {first}"
    log_returns = np.log(prices / prices.shift(1))
    market_features, bench_log_ret = L.build_market_features(index_price)
    markets = {}
    for name, key in spec["markets"].items():
        px = L.load_index(key)
        markets[name] = np.log(px / px.shift(1))
    log(f"=== {design}: stocks {prices.shape}, features {market_features.shape} "
        f"{market_features.index.min().date()}..{market_features.index.max().date()}, markets {list(markets)}")

    records = []
    for k, (a, b, c, d) in enumerate(spec["folds"], start=1):
        path = os.path.join(HERE, "cache", "step0", design, f"fold{k}.pkl")
        if os.path.exists(path):
            cache = L.load_pickle(path)
            rec = cache["fold"] | dict(std_series=cache["std_series"], dtw_series=cache["dtw_series"],
                                       bench_series=cache["bench_series"])
            log(f"  fold {k}: loaded from cache")
        else:
            rec, cache = L.run_fold(k, a, b, c, d, market_features, log_returns, bench_log_ret, markets,
                                    sigma_frac=0.5, distance="fastdtw_l1", std_nan_check=spec["nan_check"],
                                    max_workers=MAX_WORKERS, log=log)
            cache["design"] = design
            L.save_pickle(cache, path)
            log(f"  fold {k} done in {rec['elapsed_sec'] / 60:.1f} min (DTW {rec['dtw_wall_time']:.0f}s); "
                f"map={rec['regime_map']} pooled={rec['regime_pooled_days']} warnings={rec['warnings']}")
        records.append(rec)

    res, pooled_std, pooled_dtw = evaluate(design, spec, records)
    L.save_pickle(dict(design=design, spec=spec, records=records, horizon_results=res,
                       pooled_std=pooled_std, pooled_dtw=pooled_dtw, boot_seed=BOOT_SEED),
                  os.path.join(HERE, "results", f"step0_{design}.pkl"))

    orig_folds, orig_hz = load_original(spec)
    rows = []
    for k, (rec, (os_, od)) in enumerate(zip(records, orig_folds), start=1):
        ds, dd = series_diff(rec['std_series'], os_), series_diff(rec['dtw_series'], od)
        log(f"  fold {k} daily-return check: std max|diff|={ds['max_abs']:.2e} (n {ds['n_new']} vs {ds['n_old']}), "
            f"dtw max|diff|={dd['max_abs']:.2e} (n {dd['n_new']} vs {dd['n_old']})")
        rows.append(dict(design=design, level=f"fold{k}", horizon=1, std_new=np.nan, std_old=np.nan,
                         dtw_new=np.nan, dtw_old=np.nan, p_new=np.nan, p_old=np.nan,
                         std_series_maxabs=ds['max_abs'], dtw_series_maxabs=dd['max_abs'],
                         n_days_new=ds['n_new'], n_days_old=ds['n_old']))
    for h, r in res.items():
        o = orig_hz[h]
        match = round(r['std_sharpe'], 3) == round(o['std'], 3) and round(r['dtw_sharpe'], 3) == round(o['dtw'], 3)
        log(f"  [{h}d] Sharpe std {r['std_sharpe']:.4f} (orig {o['std']:.4f}) | dtw {r['dtw_sharpe']:.4f} "
            f"(orig {o['dtw']:.4f}) | p_one {r['p_one']:.4f} (orig {o['p']:.4f}) | 3dp match={match}")
        rows.append(dict(design=design, level="pooled", horizon=h, std_new=r['std_sharpe'], std_old=o['std'],
                         dtw_new=r['dtw_sharpe'], dtw_old=o['dtw'], p_new=r['p_one'], p_old=o['p'],
                         std_series_maxabs=np.nan, dtw_series_maxabs=np.nan, n_days_new=np.nan, n_days_old=np.nan))
        log_variant(stage="step0_regression", objective="sharpe", metric="sharpe", covariances="std,dtw",
                    design=design, universe=design.split("_")[-1] if "2p" not in design else design.split("_")[1],
                    horizon=h, counted_in_family="no", note="rerun of existing result from snapshot")
    path = os.path.join(HERE, "results", "step0_regression.csv")
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main():
    designs = sys.argv[1:] or ORDER
    log(f"Step 0 regression run: designs={designs}, MAX_WORKERS={MAX_WORKERS}, snapshot={L.snapshot_dir()}")
    for design in designs:
        t = time.time()
        run_design(design)
        log(f"=== {design} finished in {(time.time() - t) / 60:.1f} min")
    log("ALL DONE")


if __name__ == "__main__":
    main()
