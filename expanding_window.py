# -*- coding: utf-8 -*-
"""
Expanding-window walk-forward validation, 6 folds.
DRY_RUN=1 env var -> subset stock universe to 6 tickers for a fast end-to-end logic check.
Otherwise runs the full 50-stock pipeline for real.

DTW-per-episode computation is parallelized across processes (it's embarrassingly
parallel - each episode's pairwise DTW matrix is independent of every other episode's).
Everything else (HMM fit, causal_predict, backtest) stays single-process - those are
either fast or inherently sequential (causal_predict has to decode day t before day t+1).

Windows note: ProcessPoolExecutor uses spawn on this platform, which re-imports this
module in every worker process. Anything that must NOT re-run in a worker (the yfinance
download, the fold loop) has to live inside main() and only get called from the
`if __name__ == "__main__":` guard - top-level module code outside that guard runs in
every worker on import, guard or no guard.
"""
import os
import sys
import time
import pickle
import itertools

import yfinance as yf
import pandas as pd
import numpy as np

from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from concurrent.futures import ProcessPoolExecutor, as_completed

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
CKPT_PATH = os.environ.get("CKPT_PATH", "expanding_window_results.pkl")
LOG_PATH = os.environ.get("LOG_PATH", "expanding_window_log.txt")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))

SEEDS = [0, 7, 21, 42, 99]
FEATURE_COLS = ['mean_return', 'rolling_vol_20d', 'drawdown_60d', 'momentum_5d']

_log_file = open(LOG_PATH, "a", encoding="utf-8")

def log(*a):
    msg = " ".join(str(x) for x in a)
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    sys.stdout.flush()
    _log_file.write(line + "\n")
    _log_file.flush()

# =====================================================================================
# REUSED FUNCTIONS (verbatim from the notebook)
# =====================================================================================

def smooth_regimes(regime_series, window=60):
    smoothed = regime_series.copy()
    for i in range(window, len(regime_series)):
        window_slice = regime_series.iloc[i-window:i]
        smoothed.iloc[i] = window_slice.mode()[0]
    return smoothed

def causal_predict(model, features):
    n = len(features)
    states = np.zeros(n, dtype=int)
    for t in range(1, n + 1):
        partial_states = model.predict(features[:t])
        states[t-1] = partial_states[-1]
    return states

def extract_episodes(regime_series, min_length=10, gap_tolerance=3):
    dates = regime_series.index
    labels = regime_series.values
    raw_runs = []
    start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_idx]:
            raw_runs.append({
                'regime': labels[start_idx], 'start': dates[start_idx],
                'end': dates[i-1], 'length': i - start_idx
            })
            start_idx = i
    if not raw_runs:
        return pd.DataFrame(columns=['regime', 'start', 'end', 'length'])
    bridged = [raw_runs[0]]
    for run in raw_runs[1:]:
        bridged.append(run)
    merged = []
    i = 0
    while i < len(bridged):
        current = bridged[i].copy()
        j = i + 1
        while j + 1 < len(bridged) and bridged[j]['length'] <= gap_tolerance and bridged[j+1]['regime'] == current['regime']:
            current['end'] = bridged[j+1]['end']
            current['length'] = (dates.get_loc(current['end']) - dates.get_loc(current['start'])) + 1
            j += 2
        merged.append(current)
        i = j if j > i + 1 else i + 1
    episodes = [ep for ep in merged if ep['length'] >= min_length]
    return pd.DataFrame(episodes)

def negative_sharpe(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -port_return / port_vol

def optimize_portfolio(mean_returns, cov_matrix, max_weight=0.40):
    n = len(mean_returns)
    init_guess = np.array([1/n] * n)
    bounds = tuple((0, max_weight) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(
        negative_sharpe, init_guess,
        args=(mean_returns.values, cov_matrix.values),
        method='SLSQP', bounds=bounds, constraints=constraints
    )
    return pd.Series(result.x, index=mean_returns.index)

def compute_episode_dtw_matrix(episode_returns, min_valid_ratio=0.9):
    n_days = len(episode_returns)
    valid_counts = episode_returns.notna().sum()
    included_stocks = valid_counts[valid_counts >= min_valid_ratio * n_days].index.tolist()
    if len(included_stocks) < 2:
        return None, []
    clean_returns = episode_returns[included_stocks].dropna()
    n = len(included_stocks)
    dtw_matrix = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        series_i = clean_returns.iloc[:, i].values.reshape(-1, 1)
        series_j = clean_returns.iloc[:, j].values.reshape(-1, 1)
        distance, _ = fastdtw(series_i, series_j, dist=euclidean)
        dtw_matrix[i, j] = distance
        dtw_matrix[j, i] = distance
    return pd.DataFrame(dtw_matrix, index=included_stocks, columns=included_stocks), included_stocks

def _dtw_episode_worker(payload):
    """Runs in a worker process. Must be module-level (picklable by qualified name) and
    must not touch the shared log file - stdout/log lines from workers don't reliably
    interleave with the main process's, so workers just return results and the main
    process logs after collecting them."""
    idx, ep_returns, regime = payload
    t0 = time.time()
    dtw_matrix, included_stocks = compute_episode_dtw_matrix(ep_returns)
    return idx, regime, dtw_matrix, included_stocks, time.time() - t0

def aggregate_regime_dtw(episode_dtw_results, target_regime):
    regime_episodes = {k: v for k, v in episode_dtw_results.items() if v['regime'] == target_regime}
    if not regime_episodes:
        return None, []
    stock_sets = [set(ep['stocks']) for ep in regime_episodes.values()]
    common_stocks = sorted(set.intersection(*stock_sets))
    if len(common_stocks) < 2:
        return None, []
    matrices = []
    for ep in regime_episodes.values():
        aligned = ep['dtw_matrix'].loc[common_stocks, common_stocks]
        matrices.append(aligned.values)
    avg_matrix = np.mean(matrices, axis=0)
    avg_dtw_df = pd.DataFrame(avg_matrix, index=common_stocks, columns=common_stocks)
    return avg_dtw_df, common_stocks

def dtw_to_covariance(dtw_matrix, stock_returns, sigma_frac=0.5):
    dtw_vals = dtw_matrix.values
    base_median = np.median(dtw_vals[dtw_vals > 0])
    sigma = sigma_frac * base_median
    similarity = np.exp(-(dtw_vals ** 2) / (2 * sigma ** 2))
    eigvals, eigvecs = np.linalg.eigh(similarity)
    eigvals_clipped = np.clip(eigvals, a_min=1e-8, a_max=None)
    similarity_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    stock_vols = stock_returns.std().values
    vol_outer = np.outer(stock_vols, stock_vols)
    dtw_cov = similarity_psd * vol_outer
    dtw_cov_df = pd.DataFrame(dtw_cov, index=dtw_matrix.index, columns=dtw_matrix.columns)
    return dtw_cov_df, sigma

def compute_metrics_dict(daily_log_returns):
    total_return = np.exp(daily_log_returns.sum()) - 1
    ann_return = daily_log_returns.mean() * 252
    ann_vol = daily_log_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    cum_ret = daily_log_returns.cumsum()
    running_max = cum_ret.cummax()
    max_dd = (cum_ret - running_max).min()
    return dict(total_return=total_return, ann_return=ann_return, ann_vol=ann_vol,
                sharpe=sharpe, max_dd=max_dd, n_days=len(daily_log_returns))

def print_metrics(m, label):
    log(f"  {label}: Sharpe={m['sharpe']:.3f} AnnRet={m['ann_return']*100:.2f}% "
        f"AnnVol={m['ann_vol']*100:.2f}% MaxDD={m['max_dd']*100:.2f}% n={m['n_days']}")

def bootstrap_sharpe_diff(returns_a, returns_b, n_bootstrap=5000, block_size=20):
    common_idx = returns_a.index.intersection(returns_b.index)
    a = returns_a.loc[common_idx].values
    b = returns_b.loc[common_idx].values
    n = len(a)
    observed_diff = (a.mean() / a.std()) * np.sqrt(252) - (b.mean() / b.std()) * np.sqrt(252)
    diffs = []
    n_blocks = n // block_size
    for _ in range(n_bootstrap):
        block_starts = np.random.randint(0, n - block_size, n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in block_starts])
        a_sample = a[idx]
        b_sample = b[idx]
        sharpe_a = (a_sample.mean() / a_sample.std()) * np.sqrt(252)
        sharpe_b = (b_sample.mean() / b_sample.std()) * np.sqrt(252)
        diffs.append(sharpe_a - sharpe_b)
    diffs = np.array(diffs)
    p_value = (diffs <= 0).mean()
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    return observed_diff, p_value, ci_lower, ci_upper, diffs

def build_regime_map(train_features, train_states):
    """Sort observed states by mean_return; two lowest -> Bear, next -> Sideways, highest -> Bull.
    Degrades gracefully (with a warning) if the HMM never actually assigns the argmax state
    to one or more of the 4 components over this fold's train window."""
    df = train_features.copy()
    df['state'] = train_states
    means = df.groupby('state')['mean_return'].mean().sort_values()
    states_sorted = means.index.tolist()
    warn = None
    k = len(states_sorted)
    if k >= 4:
        regime_map = {states_sorted[0]: 'Bear', states_sorted[1]: 'Bear',
                       states_sorted[2]: 'Sideways', states_sorted[3]: 'Bull'}
    elif k == 3:
        warn = f"only 3 of 4 states observed in train predict() argmax (states={states_sorted}) - using 1 Bear state instead of merging 2"
        regime_map = {states_sorted[0]: 'Bear', states_sorted[1]: 'Sideways', states_sorted[2]: 'Bull'}
    elif k == 2:
        warn = f"only 2 of 4 states observed in train argmax (states={states_sorted}) - NO Sideways regime possible this fold"
        regime_map = {states_sorted[0]: 'Bear', states_sorted[1]: 'Bull'}
    else:
        warn = f"only {k} state observed in train argmax - entire fold collapses to one regime, results not meaningful"
        regime_map = {states_sorted[0]: 'Bull'}
    return regime_map, warn

# =====================================================================================
# MAIN (everything that must not re-execute in a worker process lives here)
# =====================================================================================

def main():
    t_script_start = time.time()

    log("=== Data pipeline: downloading stock universe ===")

    nifty50_symbols = [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
        "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
        "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
        "HINDUNILVR", "ICICIBANK", "ITC", "INFY", "INDIGO",
        "JSWSTEEL", "JIOFIN", "KOTAKBANK", "LT", "M&M",
        "MARUTI", "MAXHEALTH", "NTPC", "NESTLEIND", "ONGC",
        "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
        "SUNPHARMA", "TCS", "TATACONSUM", "TMPV", "TATASTEEL",
        "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"
    ]

    if DRY_RUN:
        nifty50_symbols = nifty50_symbols[:6]
        log("DRY_RUN: subsetting to", nifty50_symbols)

    nifty50_tickers = [s.replace("&", "%26") + ".NS" if s == "M&M" else s + ".NS" for s in nifty50_symbols]

    start_date = "2015-01-01"
    end_date = "2026-08-09"  # same vintage as the rest of the notebook, for consistency

    data = yf.download(nifty50_tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    log("Batch download shape:", data.shape)

    missing = data.isna().sum()
    failed_tickers = missing[missing == len(data)].index.tolist()
    retry_data = {}
    for t in failed_tickers:
        for attempt in range(3):
            try:
                df = yf.Ticker(t).history(start=start_date, end=end_date, auto_adjust=True)['Close']
                if len(df) > 0:
                    retry_data[t] = df
                    break
            except Exception as e:
                log(f"{t} attempt {attempt+1} failed: {e}")
            time.sleep(2)

    for t, series in retry_data.items():
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        data[t] = series.reindex(data.index)

    data = data.rename(columns={"M%26M.NS": "M&M.NS"})
    log("Final missing total:", int(data.isna().sum().sum()))

    log_returns = np.log(data / data.shift(1))

    nifty_index = yf.download("^NSEI", start=start_date, end=end_date, auto_adjust=True)['Close'].squeeze()
    nifty_log_ret = np.log(nifty_index / nifty_index.shift(1))

    roll_vol = nifty_log_ret.rolling(20).std()
    cum_ret = nifty_log_ret.cumsum()
    rolling_max = cum_ret.rolling(60).max()
    drawdown = (cum_ret - rolling_max)
    momentum = nifty_log_ret.rolling(5).mean()

    market_features = pd.DataFrame({
        'mean_return': nifty_log_ret,
        'rolling_vol_20d': roll_vol,
        'drawdown_60d': drawdown,
        'momentum_5d': momentum
    }).dropna()

    log("market_features shape:", market_features.shape)

    # =================================================================================
    # FOLD DEFINITIONS
    # =================================================================================

    folds_spec = [
        ("2015-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2015-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2015-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ]

    if DRY_RUN:
        folds_spec = folds_spec[:2]
        log("DRY_RUN: subsetting to", len(folds_spec), "folds")

    fold_results = []

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds_spec, start=1):
        t_fold_start = time.time()
        log(f"\n{'='*70}\nFOLD {fold_idx}/{len(folds_spec)}: train {tr_start}..{tr_end}  test {te_start}..{te_end}\n{'='*70}")
        warnings_this_fold = []

        train_features = market_features.loc[tr_start:tr_end, FEATURE_COLS]
        test_features = market_features.loc[te_start:te_end, FEATURE_COLS]
        log(f"  train days={len(train_features)}  test days={len(test_features)}")

        # --- HMM multi-seed fit on train only ---
        # A GaussianHMM(covariance_type="full") EM fit can occasionally drive one state's
        # estimated covariance to become non-positive-definite mid-iteration (a real risk,
        # not a data-quality issue - hmmlearn raises LinAlgError/ValueError when that
        # happens). Skip that seed and keep going rather than losing the whole fold.
        best_model, best_score = None, -np.inf
        n_seed_failures = 0
        for seed in SEEDS:
            m = GaussianHMM(n_components=4, covariance_type="full", n_iter=5000, tol=1e-4, random_state=seed)
            try:
                m.fit(train_features.values)
                score = m.score(train_features.values)
            except (np.linalg.LinAlgError, ValueError) as e:
                n_seed_failures += 1
                log(f"  seed={seed}: FIT FAILED ({type(e).__name__}: {e}) - skipping this seed")
                continue
            log(f"  seed={seed}: converged={m.monitor_.converged}, log-likelihood={score:.2f}")
            if score > best_score:
                best_score = score
                best_model = m
        if best_model is None:
            raise RuntimeError(f"Fold {fold_idx}: all {len(SEEDS)} HMM seeds failed to fit - cannot proceed")
        if n_seed_failures:
            w = f"{n_seed_failures}/{len(SEEDS)} HMM seeds failed to fit (non-PD covariance) - best model chosen from the remaining {len(SEEDS)-n_seed_failures}"
            warnings_this_fold.append(w)
            log("  WARNING:", w)
        model = best_model
        train_states = model.predict(train_features.values)

        regime_map, warn = build_regime_map(train_features, train_states)
        if warn:
            warnings_this_fold.append(warn)
            log("  WARNING:", warn)
        log("  regime_map:", regime_map)

        train_regime_raw = pd.Series(train_states, index=train_features.index).map(regime_map)
        train_regime_smoothed = smooth_regimes(train_regime_raw, window=60)

        t0 = time.time()
        test_states = causal_predict(model, test_features.values)
        log(f"  causal_predict on test ({len(test_features)} days) took {time.time()-t0:.1f}s")
        test_regime_raw = pd.Series(test_states, index=test_features.index).map(regime_map)
        test_regime_smoothed = smooth_regimes(test_regime_raw, window=60)

        log("  train regime dist:", train_regime_smoothed.value_counts().to_dict())
        log("  test regime dist:", test_regime_smoothed.value_counts().to_dict())

        # --- episode extraction (train only) ---
        train_episodes = extract_episodes(train_regime_smoothed, min_length=10, gap_tolerance=3)
        log(f"  train episodes: {len(train_episodes)} total,",
            train_episodes['regime'].value_counts().to_dict() if len(train_episodes) else {})

        regime_pooled_days = {}
        for regime in ['Bull', 'Sideways', 'Bear']:
            eps = train_episodes[train_episodes['regime'] == regime] if len(train_episodes) else train_episodes
            pooled = int(eps['length'].sum()) if len(eps) else 0
            regime_pooled_days[regime] = pooled
            n_eps = len(eps)
            if pooled == 0:
                w = f"{regime}: 0 episodes / 0 pooled days - regime totally absent from train, no frozen weights possible"
                warnings_this_fold.append(w)
                log("  WARNING:", w)
            elif pooled < 100:
                w = f"{regime}: only {pooled} pooled days across {n_eps} episode(s) - thin, treat this regime's result with caution"
                warnings_this_fold.append(w)
                log("  WARNING:", w)
            else:
                log(f"  {regime}: {pooled} pooled days across {n_eps} episodes - OK")

        train_regime_returns = {}
        for regime in ['Bull', 'Sideways', 'Bear']:
            if regime_pooled_days[regime] == 0:
                continue
            eps = train_episodes[train_episodes['regime'] == regime]
            date_ranges = [log_returns.loc[row['start']:row['end']] for _, row in eps.iterrows()]
            train_regime_returns[regime] = pd.concat(date_ranges)

        # --- standard covariance frozen weights ---
        frozen_weights_std = {}
        for regime, rets in train_regime_returns.items():
            valid_counts = rets.notna().sum()
            included = valid_counts[valid_counts >= 30].index.tolist()
            if len(included) < 2:
                warnings_this_fold.append(f"{regime}: <2 stocks pass min_periods=30 for standard cov - skipped")
                continue
            mean_ret = rets[included].mean()
            cov = rets[included].cov(min_periods=30)
            frozen_weights_std[regime] = optimize_portfolio(mean_ret, cov)
        log("  standard frozen weights built for:", list(frozen_weights_std.keys()))

        # --- DTW covariance frozen weights (expensive step - parallelized across episodes) ---
        t_dtw0 = time.time()
        tasks = [(idx, log_returns.loc[ep['start']:ep['end']], ep['regime']) for idx, ep in train_episodes.iterrows()]
        n_workers = min(MAX_WORKERS, max(1, len(tasks)))
        log(f"  DTW loop: dispatching {len(tasks)} episodes across {n_workers} worker processes")

        episode_dtw_results = {}
        n_completed = 0
        if tasks:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_dtw_episode_worker, task): task[0] for task in tasks}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        ridx, regime, dtw_matrix, included_stocks, elapsed = fut.result()
                    except Exception as e:
                        log(f"    episode {idx}: WORKER FAILED: {type(e).__name__}: {e}")
                        continue
                    n_completed += 1
                    if dtw_matrix is not None:
                        episode_dtw_results[ridx] = {'regime': regime, 'dtw_matrix': dtw_matrix, 'stocks': included_stocks}
                    log(f"    episode {ridx} ({regime}) done in {elapsed:.1f}s "
                        f"[{n_completed}/{len(tasks)} complete, {time.time()-t_dtw0:.0f}s elapsed]")
        log(f"  DTW episode loop: {len(episode_dtw_results)}/{len(train_episodes)} usable episodes, "
            f"{n_completed}/{len(tasks)} completed, wall time {time.time()-t_dtw0:.1f}s")

        frozen_weights_dtw = {}
        for regime in ['Bull', 'Sideways', 'Bear']:
            if regime_pooled_days.get(regime, 0) == 0:
                continue
            avg_dtw, stocks = aggregate_regime_dtw(episode_dtw_results, regime)
            if avg_dtw is None:
                warnings_this_fold.append(f"{regime}: DTW aggregation failed (no common stocks across episodes) - skipped for DTW")
                continue
            stock_rets = train_regime_returns[regime][stocks]
            dtw_cov, sigma_used = dtw_to_covariance(avg_dtw, stock_rets, sigma_frac=0.5)
            mean_ret = stock_rets.mean()
            frozen_weights_dtw[regime] = optimize_portfolio(mean_ret, dtw_cov)
            log(f"  {regime}: DTW cov built, sigma={sigma_used:.4f}, {len(stocks)} stocks")
        log(f"  DTW frozen weights built for: {list(frozen_weights_dtw.keys())}")

        # --- walk-forward backtest over this fold's single test year ---
        test_returns = log_returns.loc[test_features.index]
        regime_sequence = test_regime_smoothed.reindex(test_returns.index)

        std_rets, dtw_rets, bench_rets = [], [], []
        std_skipped, dtw_skipped = 0, 0
        for date in test_returns.index:
            regime = regime_sequence.loc[date]
            bench_rets.append((date, nifty_log_ret.loc[date] if date in nifty_log_ret.index else np.nan))

            if pd.isna(regime) or regime not in frozen_weights_std:
                std_skipped += 1
            else:
                w = frozen_weights_std[regime]
                day_returns = test_returns.loc[date, w.index].fillna(0)
                std_rets.append((date, float(np.dot(w, day_returns))))

            if pd.isna(regime) or regime not in frozen_weights_dtw:
                dtw_skipped += 1
            else:
                w = frozen_weights_dtw[regime]
                day_returns = test_returns.loc[date, w.index].fillna(0)
                dtw_rets.append((date, float(np.dot(w, day_returns))))

        std_series = pd.Series(dict(std_rets)).sort_index()
        dtw_series = pd.Series(dict(dtw_rets)).sort_index()
        bench_series = pd.Series(dict(bench_rets)).sort_index().dropna()

        if std_skipped:
            w = f"standard: {std_skipped}/{len(test_returns)} test days skipped (regime unavailable in frozen weights)"
            warnings_this_fold.append(w); log("  WARNING:", w)
        if dtw_skipped:
            w = f"DTW: {dtw_skipped}/{len(test_returns)} test days skipped (regime unavailable in frozen weights)"
            warnings_this_fold.append(w); log("  WARNING:", w)

        std_metrics = compute_metrics_dict(std_series) if len(std_series) else None
        dtw_metrics = compute_metrics_dict(dtw_series) if len(dtw_series) else None
        bench_metrics = compute_metrics_dict(bench_series) if len(bench_series) else None

        if std_metrics: print_metrics(std_metrics, "STANDARD")
        if dtw_metrics: print_metrics(dtw_metrics, "DTW")
        if bench_metrics: print_metrics(bench_metrics, "BENCHMARK")

        fold_record = dict(
            fold_idx=fold_idx, train_start=tr_start, train_end=tr_end, test_start=te_start, test_end=te_end,
            train_days=len(train_features), test_days=len(test_features),
            regime_map=regime_map, best_score=best_score,
            train_regime_dist=train_regime_smoothed.value_counts().to_dict(),
            test_regime_dist=test_regime_smoothed.value_counts().to_dict(),
            n_train_episodes=len(train_episodes),
            episode_regime_counts=train_episodes['regime'].value_counts().to_dict() if len(train_episodes) else {},
            regime_pooled_days=regime_pooled_days,
            std_series=std_series, dtw_series=dtw_series, bench_series=bench_series,
            std_metrics=std_metrics, dtw_metrics=dtw_metrics, bench_metrics=bench_metrics,
            warnings=warnings_this_fold,
            std_skipped=std_skipped, dtw_skipped=dtw_skipped,
            elapsed_sec=time.time() - t_fold_start,
        )
        fold_results.append(fold_record)
        log(f"  Fold {fold_idx} done in {fold_record['elapsed_sec']/60:.1f} min. "
            f"Total elapsed so far: {(time.time()-t_script_start)/60:.1f} min")

        # checkpoint after every fold
        with open(CKPT_PATH, "wb") as f:
            pickle.dump({'fold_results': fold_results, 'complete': False}, f)

    # =================================================================================
    # POOLING + FINAL SIGNIFICANCE TEST
    # =================================================================================

    log(f"\n{'='*70}\nPOOLING RESULTS ACROSS {len(fold_results)} FOLDS\n{'='*70}")

    pooled_std = pd.concat([fr['std_series'] for fr in fold_results]).sort_index()
    pooled_dtw = pd.concat([fr['dtw_series'] for fr in fold_results]).sort_index()
    pooled_bench = pd.concat([fr['bench_series'] for fr in fold_results]).sort_index()

    log("Pooled n_days: std=", len(pooled_std), "dtw=", len(pooled_dtw), "bench=", len(pooled_bench))

    pooled_std_metrics = compute_metrics_dict(pooled_std)
    pooled_dtw_metrics = compute_metrics_dict(pooled_dtw)
    pooled_bench_metrics = compute_metrics_dict(pooled_bench)
    print_metrics(pooled_std_metrics, "POOLED STANDARD")
    print_metrics(pooled_dtw_metrics, "POOLED DTW")
    print_metrics(pooled_bench_metrics, "POOLED BENCHMARK")

    t0 = time.time()
    obs_diff, p_val, ci_low, ci_high, diffs = bootstrap_sharpe_diff(pooled_dtw, pooled_std)
    log(f"Bootstrap ({time.time()-t0:.1f}s): observed_diff={obs_diff:.3f} p={p_val:.4f} CI=[{ci_low:.3f},{ci_high:.3f}]")

    per_fold_table = []
    for fr in fold_results:
        sd = fr['std_metrics']['sharpe'] if fr['std_metrics'] else np.nan
        dd = fr['dtw_metrics']['sharpe'] if fr['dtw_metrics'] else np.nan
        bd = fr['bench_metrics']['sharpe'] if fr['bench_metrics'] else np.nan
        per_fold_table.append(dict(fold=fr['fold_idx'], test_year=fr['test_start'][:4],
                                    std_sharpe=sd, dtw_sharpe=dd, bench_sharpe=bd, gap=dd-sd))
    per_fold_df = pd.DataFrame(per_fold_table)
    log("\nPer-fold Sharpe gap table:\n" + per_fold_df.to_string(index=False))

    final = dict(
        fold_results=fold_results,
        pooled_std=pooled_std, pooled_dtw=pooled_dtw, pooled_bench=pooled_bench,
        pooled_std_metrics=pooled_std_metrics, pooled_dtw_metrics=pooled_dtw_metrics, pooled_bench_metrics=pooled_bench_metrics,
        bootstrap=dict(observed_diff=obs_diff, p_value=p_val, ci=(ci_low, ci_high)),
        per_fold_df=per_fold_df,
        complete=True,
        total_elapsed_sec=time.time() - t_script_start,
    )
    with open(CKPT_PATH, "wb") as f:
        pickle.dump(final, f)

    log(f"\nALL DONE. Total elapsed: {(time.time()-t_script_start)/60:.1f} min. Saved to {CKPT_PATH}")
    _log_file.close()


if __name__ == "__main__":
    main()
