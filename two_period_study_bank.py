# -*- coding: utf-8 -*-
"""
Two-period, multi-horizon extension of the Nifty 50 DTW regime-conditional covariance
study. Period A = 2008-2019 ("calm era", 43-stock filtered universe). Period B =
2020-2026 ("crisis/recent era", full 50-stock universe, STRICT training separation -
no data before 2020-01-01 enters any Period B computation, including rolling-feature
warmup). All shared functions copied verbatim from the notebook / prior scripts.
"""
import os
import sys
import time
import pickle
import itertools
import traceback

import yfinance as yf
import pandas as pd
import numpy as np

from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from concurrent.futures import ProcessPoolExecutor, as_completed

CKPT_PATH = os.environ.get("CKPT_PATH", "two_period_results.pkl")
LOG_PATH = os.environ.get("LOG_PATH", "two_period_log.txt")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

SEEDS = [0, 7, 21, 42, 99]
FEATURE_COLS = ['mean_return', 'rolling_vol_20d', 'drawdown_60d', 'momentum_5d']

_log_file = open(LOG_PATH, "a", encoding="utf-8")
_t_script_start = time.time()

def log(*a):
    msg = " ".join(str(x) for x in a)
    elapsed = (time.time() - _t_script_start) / 60
    line = f"[{time.strftime('%H:%M:%S')} | +{elapsed:6.1f}min] {msg}"
    print(line); sys.stdout.flush()
    _log_file.write(line + "\n"); _log_file.flush()

def stage(label):
    log(f"--- STAGE START: {label} ---")
    return time.time()

def stage_end(label, t0):
    log(f"--- STAGE END: {label} ({time.time()-t0:.1f}s) ---")

# =====================================================================================
# REUSED FUNCTIONS (verbatim from the notebook / prior Bank+Midcap scripts)
# =====================================================================================

def smooth_regimes(regime_series, window=60):
    smoothed = regime_series.copy()
    for i in range(window, len(regime_series)):
        smoothed.iloc[i] = regime_series.iloc[i-window:i].mode()[0]
    return smoothed

def causal_predict(model, features):
    n = len(features)
    states = np.zeros(n, dtype=int)
    for t in range(1, n + 1):
        states[t-1] = model.predict(features[:t])[-1]
    return states

def extract_episodes(regime_series, min_length=10, gap_tolerance=3):
    dates = regime_series.index
    labels = regime_series.values
    raw_runs = []
    start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_idx]:
            raw_runs.append({'regime': labels[start_idx], 'start': dates[start_idx],
                              'end': dates[i-1], 'length': i - start_idx})
            start_idx = i
    if not raw_runs:
        return pd.DataFrame(columns=['regime', 'start', 'end', 'length'])
    bridged = [raw_runs[0]] + raw_runs[1:]
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
    return pd.DataFrame([ep for ep in merged if ep['length'] >= min_length])

def negative_sharpe(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -port_return / port_vol

def optimize_portfolio(mean_returns, cov_matrix, max_weight=0.40):
    n = len(mean_returns)
    result = minimize(
        negative_sharpe, np.array([1/n] * n),
        args=(mean_returns.values, cov_matrix.values),
        method='SLSQP', bounds=tuple((0, max_weight) for _ in range(n)),
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
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
    matrices = [ep['dtw_matrix'].loc[common_stocks, common_stocks].values for ep in regime_episodes.values()]
    avg_matrix = np.mean(matrices, axis=0)
    return pd.DataFrame(avg_matrix, index=common_stocks, columns=common_stocks), common_stocks

def dtw_to_covariance(dtw_matrix, stock_returns, sigma_frac=0.5):
    dtw_vals = dtw_matrix.values
    base_median = np.median(dtw_vals[dtw_vals > 0])
    sigma = sigma_frac * base_median
    similarity = np.exp(-(dtw_vals ** 2) / (2 * sigma ** 2))
    eigvals, eigvecs = np.linalg.eigh(similarity)
    eigvals_clipped = np.clip(eigvals, a_min=1e-8, a_max=None)
    similarity_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    stock_vols = stock_returns.std().values
    dtw_cov = similarity_psd * np.outer(stock_vols, stock_vols)
    return pd.DataFrame(dtw_cov, index=dtw_matrix.index, columns=dtw_matrix.columns), sigma

def evaluate_sigma(dtw_matrix, sigma_frac, stock_returns):
    dtw_vals = dtw_matrix.values
    base_median = np.median(dtw_vals[dtw_vals > 0])
    sigma = sigma_frac * base_median
    similarity = np.exp(-(dtw_vals ** 2) / (2 * sigma ** 2))
    eigvals, eigvecs = np.linalg.eigh(similarity)
    eigvals_clipped = np.clip(eigvals, a_min=1e-8, a_max=None)
    similarity_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    stock_vols = stock_returns.std().values
    cov = similarity_psd * np.outer(stock_vols, stock_vols)
    cond_number = np.linalg.eigvalsh(cov)[-1] / np.linalg.eigvalsh(cov)[0]
    off_diag = similarity_psd[np.triu_indices_from(similarity_psd, k=1)]
    return sigma, cond_number, off_diag.std()

def compute_metrics_dict(daily_log_returns, periods_per_year=252):
    total_return = np.exp(daily_log_returns.sum()) - 1
    ann_return = daily_log_returns.mean() * periods_per_year
    ann_vol = daily_log_returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    cum_ret = daily_log_returns.cumsum()
    max_dd = (cum_ret - cum_ret.cummax()).min()
    return dict(total_return=total_return, ann_return=ann_return, ann_vol=ann_vol,
                sharpe=sharpe, max_dd=max_dd, n_days=len(daily_log_returns))

def print_metrics(m, label):
    log(f"  {label}: Sharpe={m['sharpe']:.3f} AnnRet={m['ann_return']*100:.2f}% "
        f"AnnVol={m['ann_vol']*100:.2f}% MaxDD={m['max_dd']*100:.2f}% n={m['n_days']}")

def bootstrap_sharpe_diff(returns_a, returns_b, n_bootstrap=5000, block_size=20, periods_per_year=252):
    common_idx = returns_a.index.intersection(returns_b.index)
    a = returns_a.loc[common_idx].values
    b = returns_b.loc[common_idx].values
    n = len(a)
    if n <= block_size:
        return np.nan, np.nan, np.nan, np.nan, None, n, 0
    observed_diff = (a.mean() / a.std()) * np.sqrt(periods_per_year) - (b.mean() / b.std()) * np.sqrt(periods_per_year)
    diffs = []
    n_blocks = n // block_size
    for _ in range(n_bootstrap):
        block_starts = np.random.randint(0, n - block_size, n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in block_starts])
        a_sample, b_sample = a[idx], b[idx]
        sharpe_a = (a_sample.mean() / a_sample.std()) * np.sqrt(periods_per_year)
        sharpe_b = (b_sample.mean() / b_sample.std()) * np.sqrt(periods_per_year)
        diffs.append(sharpe_a - sharpe_b)
    diffs = np.array(diffs)
    p_value = (diffs <= 0).mean()
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    return observed_diff, p_value, ci_lower, ci_upper, diffs, n, n_blocks

def build_regime_map(train_features, train_states):
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
        warn = f"only 3 of 4 states observed (states={states_sorted})"
        regime_map = {states_sorted[0]: 'Bear', states_sorted[1]: 'Sideways', states_sorted[2]: 'Bull'}
    elif k == 2:
        warn = f"only 2 of 4 states observed (states={states_sorted}) - no Sideways"
        regime_map = {states_sorted[0]: 'Bear', states_sorted[1]: 'Bull'}
    else:
        warn = f"only {k} state observed - fold collapses to one regime"
        regime_map = {states_sorted[0]: 'Bull'}
    return regime_map, warn

def aggregate_to_horizon(daily_returns, horizon):
    """Sum consecutive non-overlapping blocks of `horizon` daily log returns (log
    returns are additive across time, so summing is the correct aggregation)."""
    if horizon == 1:
        return daily_returns
    n = (len(daily_returns) // horizon) * horizon
    trimmed = daily_returns.iloc[:n]
    grouped = trimmed.groupby(np.arange(len(trimmed)) // horizon).sum()
    # index by the LAST date in each block, for readability if ever printed
    block_end_dates = trimmed.index[horizon - 1::horizon]
    grouped.index = block_end_dates
    return grouped


# =====================================================================================
# CORE FOLD RUNNER - shared by both periods
# =====================================================================================

def run_fold(fold_idx, n_folds, tr_start, tr_end, te_start, te_end,
             market_features, log_returns, index_log_ret, sigma_frac,
             prior_dtw_times):
    """Runs one HMM-fit -> regime-detect -> episode-extract -> covariance-build ->
    backtest fold. Returns a fold_record dict. Raises only if truly unrecoverable
    (caller wraps in try/except per the 'don't discard a failed fold silently' spec)."""
    t_fold_start = time.time()
    log(f"{'='*70}")
    log(f"FOLD {fold_idx}/{n_folds}: train {tr_start}..{tr_end}  test {te_start}..{te_end}")
    log(f"{'='*70}")
    warnings_this_fold = []

    train_features = market_features.loc[tr_start:tr_end, FEATURE_COLS]
    test_features = market_features.loc[te_start:te_end, FEATURE_COLS]
    log(f"  train days={len(train_features)}  test days={len(test_features)}")
    if len(train_features) < 150:
        w = f"train set is only {len(train_features)} days - well below every other fold built in this project (usually 1100+)"
        warnings_this_fold.append(w)
        log("  WARNING (THIN FOLD):", w)

    t0 = stage(f"fold {fold_idx} HMM fit")
    best_model, best_score, n_seed_failures = None, -np.inf, 0
    seed_scores = []
    for seed in SEEDS:
        m = GaussianHMM(n_components=4, covariance_type="full", n_iter=5000, tol=1e-4, random_state=seed)
        try:
            m.fit(train_features.values)
            score = m.score(train_features.values)
        except (np.linalg.LinAlgError, ValueError) as e:
            n_seed_failures += 1
            log(f"  seed={seed}: FIT FAILED ({type(e).__name__}: {e})")
            continue
        log(f"  seed={seed}: converged={m.monitor_.converged}, log-likelihood={score:.2f}")
        seed_scores.append(score)
        if score > best_score:
            best_score, best_model = score, m
    stage_end(f"fold {fold_idx} HMM fit", t0)
    if best_model is None:
        raise RuntimeError(f"Fold {fold_idx}: all {len(SEEDS)} HMM seeds failed to fit - regime detection impossible")
    if n_seed_failures:
        w = f"{n_seed_failures}/{len(SEEDS)} HMM seeds failed to fit"
        warnings_this_fold.append(w); log("  WARNING:", w)
    if len(seed_scores) >= 2 and (max(seed_scores) - min(seed_scores)) / abs(max(seed_scores)) > 0.05:
        w = f"seed log-likelihoods spread widely ({min(seed_scores):.1f} to {max(seed_scores):.1f}) - fit looks less stable than usual"
        warnings_this_fold.append(w); log("  WARNING:", w)
    model = best_model
    train_states = model.predict(train_features.values)

    regime_map, warn = build_regime_map(train_features, train_states)
    if warn:
        warnings_this_fold.append(warn); log("  WARNING:", warn)
    log("  regime_map:", regime_map)

    train_regime_smoothed = smooth_regimes(
        pd.Series(train_states, index=train_features.index).map(regime_map), window=60)
    test_states = causal_predict(model, test_features.values)
    test_regime_smoothed = smooth_regimes(
        pd.Series(test_states, index=test_features.index).map(regime_map), window=60)

    log("  train regime dist:", train_regime_smoothed.value_counts().to_dict())
    log("  test regime dist:", test_regime_smoothed.value_counts().to_dict())

    t0 = stage(f"fold {fold_idx} episode extraction")
    train_episodes = extract_episodes(train_regime_smoothed, min_length=10, gap_tolerance=3)
    stage_end(f"fold {fold_idx} episode extraction", t0)
    log(f"  train episodes: {len(train_episodes)} total,",
        train_episodes['regime'].value_counts().to_dict() if len(train_episodes) else {})
    n_regimes_present = train_episodes['regime'].nunique() if len(train_episodes) else 0
    if n_regimes_present < 3:
        w = f"only {n_regimes_present}/3 regimes have any surviving episodes at all"
        warnings_this_fold.append(w); log("  WARNING (DEGENERATE):", w)

    regime_pooled_days = {}
    for regime in ['Bull', 'Sideways', 'Bear']:
        eps = train_episodes[train_episodes['regime'] == regime] if len(train_episodes) else train_episodes
        pooled = int(eps['length'].sum()) if len(eps) else 0
        regime_pooled_days[regime] = pooled
        n_eps = len(eps)
        if pooled == 0:
            w = f"{regime}: 0 pooled days - regime absent from train"
            warnings_this_fold.append(w); log("  WARNING:", w)
        elif pooled < 100:
            w = f"{regime}: only {pooled} pooled days across {n_eps} episode(s) - thin"
            warnings_this_fold.append(w); log("  WARNING:", w)
        else:
            log(f"  {regime}: {pooled} pooled days across {n_eps} episodes - OK")

    train_regime_returns = {}
    for regime in ['Bull', 'Sideways', 'Bear']:
        if regime_pooled_days[regime] == 0:
            continue
        eps = train_episodes[train_episodes['regime'] == regime]
        date_ranges = [log_returns.loc[row['start']:row['end']] for _, row in eps.iterrows()]
        train_regime_returns[regime] = pd.concat(date_ranges)

    frozen_weights_std = {}
    for regime, rets in train_regime_returns.items():
        valid_counts = rets.notna().sum()
        included = valid_counts[valid_counts >= 30].index.tolist()
        if len(included) < 2:
            warnings_this_fold.append(f"{regime}: <2 stocks pass min_periods=30 for standard cov")
            continue
        cov = rets[included].cov(min_periods=30)
        if cov.isna().any().any():
            warnings_this_fold.append(f"{regime}: standard covariance has NaN entries even after min_periods filter - degenerate")
            continue
        frozen_weights_std[regime] = optimize_portfolio(rets[included].mean(), cov)
    log("  standard frozen weights built for:", list(frozen_weights_std.keys()),
        "| stock counts:", {k: len(v) for k, v in frozen_weights_std.items()})

    t0 = stage(f"fold {fold_idx} DTW loop ({len(train_episodes)} episodes)")
    tasks = [(idx, log_returns.loc[ep['start']:ep['end']], ep['regime']) for idx, ep in train_episodes.iterrows()]
    episode_dtw_results = {}
    n_completed = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(tasks)))) as executor:
            futures = {executor.submit(_dtw_episode_worker, task): task[0] for task in tasks}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    ridx, regime, dtw_matrix, included_stocks, elapsed = fut.result()
                except Exception as e:
                    log(f"    episode {idx}: WORKER FAILED: {e}")
                    continue
                n_completed += 1
                if dtw_matrix is not None:
                    episode_dtw_results[ridx] = {'regime': regime, 'dtw_matrix': dtw_matrix, 'stocks': included_stocks}
                if n_completed % 5 == 0 or n_completed == len(tasks):
                    log(f"    ...episode {n_completed}/{len(tasks)} done, {time.time()-t0:.0f}s elapsed")
    dtw_elapsed = time.time() - t0
    stage_end(f"fold {fold_idx} DTW loop", t0)
    if prior_dtw_times and dtw_elapsed > 2 * max(prior_dtw_times):
        w = f"this fold's DTW loop took {dtw_elapsed:.0f}s, more than 2x any prior fold in this run ({max(prior_dtw_times):.0f}s max so far) - flagging as unusually slow"
        warnings_this_fold.append(w); log("  WARNING (SLOW FOLD):", w)
    prior_dtw_times.append(dtw_elapsed)
    log(f"  DTW episode loop: {len(episode_dtw_results)}/{len(train_episodes)} usable, "
        f"{n_completed}/{len(tasks)} completed, wall time {dtw_elapsed:.1f}s")

    frozen_weights_dtw = {}
    for regime in ['Bull', 'Sideways', 'Bear']:
        if regime_pooled_days.get(regime, 0) == 0:
            continue
        avg_dtw, stocks = aggregate_regime_dtw(episode_dtw_results, regime)
        if avg_dtw is None:
            warnings_this_fold.append(f"{regime}: DTW aggregation failed - no common stocks across episodes")
            continue
        stock_rets = train_regime_returns[regime][stocks]
        dtw_cov, sigma_used = dtw_to_covariance(avg_dtw, stock_rets, sigma_frac=sigma_frac)
        if np.isnan(dtw_cov.values).any():
            warnings_this_fold.append(f"{regime}: DTW covariance has NaN entries - degenerate")
            continue
        frozen_weights_dtw[regime] = optimize_portfolio(stock_rets.mean(), dtw_cov)
        log(f"  {regime}: DTW cov built, sigma={sigma_used:.4f}, {len(stocks)} stocks")
    log("  DTW frozen weights built for:", list(frozen_weights_dtw.keys()))

    if n_regimes_present < 3 or len(frozen_weights_std) < 3 or len(frozen_weights_dtw) < 3:
        w = (f"fold {fold_idx} could not build all 3 regimes' frozen weights for both methods "
             f"(std={list(frozen_weights_std.keys())}, dtw={list(frozen_weights_dtw.keys())}) - "
             f"proceeding, but test days in a missing regime will be skipped, not filled in")
        warnings_this_fold.append(w); log("  WARNING:", w)

    cap_binds = []
    for wd in [frozen_weights_std, frozen_weights_dtw]:
        for w in wd.values():
            cap_binds.append(1 if w.max() >= 0.395 else 0)

    test_returns = log_returns.loc[test_features.index]
    regime_sequence = test_regime_smoothed.reindex(test_returns.index)

    std_rets, dtw_rets, bench_rets = [], [], []
    std_skipped = dtw_skipped = 0
    for date in test_returns.index:
        regime = regime_sequence.loc[date]
        bench_rets.append((date, index_log_ret.loc[date] if date in index_log_ret.index else np.nan))
        if pd.isna(regime) or regime not in frozen_weights_std:
            std_skipped += 1
        else:
            w = frozen_weights_std[regime]
            std_rets.append((date, float(np.dot(w, test_returns.loc[date, w.index].fillna(0)))))
        if pd.isna(regime) or regime not in frozen_weights_dtw:
            dtw_skipped += 1
        else:
            w = frozen_weights_dtw[regime]
            dtw_rets.append((date, float(np.dot(w, test_returns.loc[date, w.index].fillna(0)))))

    std_series = pd.Series(dict(std_rets)).sort_index()
    dtw_series = pd.Series(dict(dtw_rets)).sort_index()
    bench_series = pd.Series(dict(bench_rets)).sort_index().dropna()

    if std_skipped or dtw_skipped:
        w = f"std_skipped={std_skipped}, dtw_skipped={dtw_skipped} test days (regime unavailable)"
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
        regime_map=regime_map, n_train_episodes=len(train_episodes),
        episode_regime_counts=train_episodes['regime'].value_counts().to_dict() if len(train_episodes) else {},
        regime_pooled_days=regime_pooled_days,
        std_series=std_series, dtw_series=dtw_series, bench_series=bench_series,
        std_metrics=std_metrics, dtw_metrics=dtw_metrics, bench_metrics=bench_metrics,
        warnings=warnings_this_fold, std_skipped=std_skipped, dtw_skipped=dtw_skipped,
        cap_binds=cap_binds, dtw_wall_time=dtw_elapsed,
        elapsed_sec=time.time() - t_fold_start, failed=False,
    )
    log(f"  Fold {fold_idx} done in {(time.time()-t_fold_start)/60:.1f} min.")
    return fold_record


def run_sigma_sweep(label, calib_train_features, log_returns):
    t0 = stage(f"{label} sigma sweep calibration")
    best_model, best_score = None, -np.inf
    for seed in SEEDS:
        m = GaussianHMM(n_components=4, covariance_type="full", n_iter=5000, tol=1e-4, random_state=seed)
        try:
            m.fit(calib_train_features.values)
            score = m.score(calib_train_features.values)
        except (np.linalg.LinAlgError, ValueError) as e:
            log(f"  calib seed={seed}: FIT FAILED: {e}")
            continue
        if score > best_score:
            best_score, best_model = score, m
    calib_states = best_model.predict(calib_train_features.values)
    calib_regime_map, _ = build_regime_map(calib_train_features, calib_states)
    calib_regime_smoothed = smooth_regimes(
        pd.Series(calib_states, index=calib_train_features.index).map(calib_regime_map), window=60)
    calib_episodes = extract_episodes(calib_regime_smoothed, min_length=10, gap_tolerance=3)
    log(f"  calibration episodes: {len(calib_episodes)},", calib_episodes['regime'].value_counts().to_dict())

    calib_regime_returns, calib_dtw_matrices, calib_dtw_stocks = {}, {}, {}
    for regime in ['Bull', 'Sideways', 'Bear']:
        eps = calib_episodes[calib_episodes['regime'] == regime]
        if len(eps) == 0:
            continue
        date_ranges = [log_returns.loc[row['start']:row['end']] for _, row in eps.iterrows()]
        calib_regime_returns[regime] = pd.concat(date_ranges)
        tasks = [(idx, log_returns.loc[ep['start']:ep['end']], regime) for idx, ep in eps.iterrows()]
        episode_dtw = {}
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(tasks)))) as executor:
            futures = {executor.submit(_dtw_episode_worker, task): task[0] for task in tasks}
            n_done = 0
            for fut in as_completed(futures):
                ridx, rregime, dtw_matrix, stocks, _ = fut.result()
                n_done += 1
                if dtw_matrix is not None:
                    episode_dtw[ridx] = {'regime': rregime, 'dtw_matrix': dtw_matrix, 'stocks': stocks}
                if n_done % 5 == 0 or n_done == len(tasks):
                    log(f"    [{regime}] ...{n_done}/{len(tasks)} calibration episodes done")
        avg_dtw, stocks = aggregate_regime_dtw(episode_dtw, regime)
        if avg_dtw is not None:
            calib_dtw_matrices[regime] = avg_dtw
            calib_dtw_stocks[regime] = stocks

    sigma_sweep_results = {}
    for regime in calib_dtw_matrices:
        stocks = calib_dtw_stocks[regime]
        log(f"  {regime} regime - sigma sensitivity ({len(stocks)} stocks):")
        log(f"  {'sigma_frac':>10} {'sigma':>10} {'cond_number':>12} {'off_diag_std':>13}")
        rows = []
        for frac in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
            sigma, cond, off_std = evaluate_sigma(calib_dtw_matrices[regime], frac, calib_regime_returns[regime][stocks])
            rows.append((frac, sigma, cond, off_std))
            log(f"  {frac:>10.2f} {sigma:>10.4f} {cond:>12.1f} {off_std:>13.6f}")
        sigma_sweep_results[regime] = rows
    stage_end(f"{label} sigma sweep calibration", t0)
    return sigma_sweep_results


def run_period(label, folds_spec, universe_tickers, feature_start_date, feature_end_date,
               strict_no_lookback_before=None, index_ticker="^NSEI"):
    """
    universe_tickers: list of .NS tickers for this period's stock universe.
    feature_start_date/feature_end_date: date range to pull index+stock data over.
    strict_no_lookback_before: if set, asserts the pulled data's first row is exactly
    this date (i.e. no earlier data is present anywhere in this period's pipeline).
    index_ticker: yfinance ticker for the regime-detection index (^NSEI for Nifty 50,
    ^NSEBANK for Nifty Bank).
    """
    t_period_start = time.time()
    log(f"\n{'#'*70}\n# PERIOD {label}\n{'#'*70}")

    t0 = stage(f"{label} data pull")
    data = yf.download(universe_tickers, start=feature_start_date, end=feature_end_date, auto_adjust=True)['Close']
    log(f"  batch download shape: {data.shape}")
    missing = data.isna().sum()
    failed_tickers = missing[missing == len(data)].index.tolist()
    if failed_tickers:
        log(f"  retrying {len(failed_tickers)} flaky tickers:", failed_tickers)
        retry_data = {}
        for t in failed_tickers:
            for attempt in range(3):
                try:
                    df = yf.Ticker(t).history(start=feature_start_date, end=feature_end_date, auto_adjust=True)['Close']
                    if len(df) > 0:
                        retry_data[t] = df
                        break
                except Exception as e:
                    log(f"    {t} attempt {attempt+1} failed: {e}")
                time.sleep(2)
        for t, series in retry_data.items():
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            data[t] = series.reindex(data.index)
    data = data.rename(columns={"M%26M.NS": "M&M.NS"})
    log(f"  final missing total: {int(data.isna().sum().sum())}")

    index_price = None
    for attempt in range(3):
        try:
            index_price = yf.download(index_ticker, start=feature_start_date, end=feature_end_date, auto_adjust=True)['Close'].squeeze()
            if len(index_price) > 0:
                break
        except Exception as e:
            log(f"  {index_ticker} attempt {attempt+1} failed: {e}")
        time.sleep(3)
    log(f"  {index_ticker} shape: {index_price.shape}, first={index_price.first_valid_index()}, last={index_price.last_valid_index()}")

    if strict_no_lookback_before is not None:
        actual_first = min(data.first_valid_index() or pd.Timestamp.max, index_price.first_valid_index() or pd.Timestamp.max)
        assert actual_first >= pd.Timestamp(strict_no_lookback_before), (
            f"STRICT SEPARATION VIOLATED: data starts at {actual_first}, before required {strict_no_lookback_before}")
        log(f"  STRICT SEPARATION CHECK PASSED: earliest data point is {actual_first}, "
            f"on/after required {strict_no_lookback_before}")
    stage_end(f"{label} data pull", t0)

    log_returns = np.log(data / data.shift(1))
    index_log_ret = np.log(index_price / index_price.shift(1))
    roll_vol = index_log_ret.rolling(20).std()
    cum_ret = index_log_ret.cumsum()
    drawdown = cum_ret - cum_ret.rolling(60).max()
    momentum = index_log_ret.rolling(5).mean()
    market_features = pd.DataFrame({
        'mean_return': index_log_ret, 'rolling_vol_20d': roll_vol,
        'drawdown_60d': drawdown, 'momentum_5d': momentum
    }).dropna()
    log(f"  market_features shape: {market_features.shape}, "
        f"first={market_features.index.min()}, last={market_features.index.max()}")
    if strict_no_lookback_before is not None:
        assert market_features.index.min() >= pd.Timestamp(strict_no_lookback_before), (
            "STRICT SEPARATION VIOLATED after rolling-feature construction")

    # sigma sweep, calibrated on this period's OWN first ~half of available fold-1 train years
    # (can't reuse the 2015-2021 calibration window used elsewhere - it doesn't exist inside
    # either period's own boundaries)
    calib_tr_start, calib_tr_end = folds_spec[len(folds_spec)//2][0], folds_spec[len(folds_spec)//2][1]
    log(f"  sigma sweep calibration window: {calib_tr_start}..{calib_tr_end} (middle fold's train window)")
    calib_train_features = market_features.loc[calib_tr_start:calib_tr_end, FEATURE_COLS]
    sigma_sweep_results = run_sigma_sweep(label, calib_train_features, log_returns)

    # decide sigma_frac from the sweep: pick the frac (from the swept grid) that keeps
    # condition number lowest while off_diag_std is still meaningfully above ~0 (i.e. not
    # the trivial near-identity solution) - same criterion used qualitatively for Bank/Midcap.
    # Concretely: among fracs with off_diag_std >= 0.05 (real differentiation), pick the
    # one with the lowest median condition number across regimes.
    candidate_fracs = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
    frac_scores = {}
    for frac in candidate_fracs:
        conds, off_stds = [], []
        for regime, rows in sigma_sweep_results.items():
            for f, sigma, cond, off_std in rows:
                if f == frac:
                    conds.append(cond); off_stds.append(off_std)
        if conds:
            frac_scores[frac] = (np.median(off_stds), np.median(conds))
    viable = {f: sc for f, sc in frac_scores.items() if sc[0] >= 0.05}
    if viable:
        chosen_sigma_frac = min(viable, key=lambda f: viable[f][1])
    else:
        chosen_sigma_frac = 0.5
    log(f"  sigma_frac candidates and (median off_diag_std, median cond): {frac_scores}")
    log(f"  CHOSEN sigma_frac for {label}: {chosen_sigma_frac} "
        f"({'matches default 0.5' if chosen_sigma_frac == 0.5 else 'DIFFERS from default 0.5'})")

    fold_results = []
    prior_dtw_times = []
    for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds_spec, start=1):
        try:
            fr = run_fold(i, len(folds_spec), tr_start, tr_end, te_start, te_end,
                          market_features, log_returns, index_log_ret, chosen_sigma_frac,
                          prior_dtw_times)
        except Exception as e:
            tb = traceback.format_exc()
            log(f"  FOLD {i} FAILED OUTRIGHT: {type(e).__name__}: {e}")
            log(f"  Traceback:\n{tb}")
            fr = dict(fold_idx=i, train_start=tr_start, train_end=tr_end, test_start=te_start, test_end=te_end,
                      failed=True, error=f"{type(e).__name__}: {e}", warnings=[f"FOLD FAILED: {e}"])
        fold_results.append(fr)
        log(f"  Period {label} total elapsed so far: {(time.time()-t_period_start)/60:.1f} min")
        partial_path = CKPT_PATH + f".partial_{label}"
        with open(partial_path, "wb") as f:
            pickle.dump({"label": label, "fold_results_so_far": fold_results, "complete": False}, f)

    good_folds = [fr for fr in fold_results if not fr.get('failed')]
    log(f"\n{label}: {len(good_folds)}/{len(fold_results)} folds succeeded")

    pooled = {}
    if good_folds:
        pooled_std = pd.concat([fr['std_series'] for fr in good_folds]).sort_index()
        pooled_dtw = pd.concat([fr['dtw_series'] for fr in good_folds]).sort_index()
        pooled_bench = pd.concat([fr['bench_series'] for fr in good_folds]).sort_index()

        log(f"\n{'='*70}\nPOOLING + MULTI-HORIZON EVALUATION: {label}\n{'='*70}")
        log(f"pooled n_days (1-day): std={len(pooled_std)} dtw={len(pooled_dtw)} bench={len(pooled_bench)}")

        horizon_results = {}
        common_idx = pooled_std.index.intersection(pooled_dtw.index)
        pooled_std_c = pooled_std.loc[common_idx]
        pooled_dtw_c = pooled_dtw.loc[common_idx]

        for horizon in [1, 5, 22]:
            t0 = stage(f"{label} horizon={horizon}d aggregation + significance test")
            ppy = 252 / horizon
            std_h = aggregate_to_horizon(pooled_std_c, horizon)
            dtw_h = aggregate_to_horizon(pooled_dtw_c, horizon)
            bench_h = aggregate_to_horizon(pooled_bench, horizon)

            std_m = compute_metrics_dict(std_h, periods_per_year=ppy)
            dtw_m = compute_metrics_dict(dtw_h, periods_per_year=ppy)
            bench_m = compute_metrics_dict(bench_h, periods_per_year=ppy)
            print_metrics(std_m, f"[{horizon}d] STANDARD")
            print_metrics(dtw_m, f"[{horizon}d] DTW")
            print_metrics(bench_m, f"[{horizon}d] BENCHMARK")

            obs_diff, p_val, ci_lo, ci_hi, diffs, n_used, n_blocks = bootstrap_sharpe_diff(
                dtw_h, std_h, block_size=20, periods_per_year=ppy)
            log(f"  [{horizon}d] bootstrap: n_obs={n_used}, n_blocks/resample={n_blocks}, "
                f"observed_diff={obs_diff:.3f}, p={p_val:.4f}, CI=[{ci_lo:.3f},{ci_hi:.3f}]")
            if n_blocks and n_blocks < 10:
                w = f"[{horizon}d] bootstrap resamples only {n_blocks} blocks of 20 from {n_used} observations - low resolution, treat p-value with real caution"
                log("  WARNING:", w)

            horizon_results[horizon] = dict(
                std_metrics=std_m, dtw_metrics=dtw_m, bench_metrics=bench_m,
                gap=dtw_m['sharpe'] - std_m['sharpe'],
                bootstrap=dict(observed_diff=obs_diff, p_value=p_val, ci=(ci_lo, ci_hi), n_obs=n_used, n_blocks=n_blocks),
            )
            stage_end(f"{label} horizon={horizon}d", t0)

        pooled = dict(pooled_std=pooled_std, pooled_dtw=pooled_dtw, pooled_bench=pooled_bench,
                      horizon_results=horizon_results)

    per_fold_table = []
    for fr in fold_results:
        if fr.get('failed'):
            per_fold_table.append(dict(fold=fr['fold_idx'], test_year=fr['test_start'][:4], failed=True))
            continue
        sd = fr['std_metrics']['sharpe'] if fr['std_metrics'] else np.nan
        dd = fr['dtw_metrics']['sharpe'] if fr['dtw_metrics'] else np.nan
        bd = fr['bench_metrics']['sharpe'] if fr['bench_metrics'] else np.nan
        per_fold_table.append(dict(fold=fr['fold_idx'], test_year=fr['test_start'][:4],
                                    std_sharpe=sd, dtw_sharpe=dd, bench_sharpe=bd, gap=dd-sd, failed=False))
    per_fold_df = pd.DataFrame(per_fold_table)
    log(f"\n{label} per-fold Sharpe gap table (1-day):\n" + per_fold_df.to_string(index=False))

    result = dict(
        label=label, fold_results=fold_results, sigma_sweep_results=sigma_sweep_results,
        chosen_sigma_frac=chosen_sigma_frac, per_fold_df=per_fold_df,
        n_universe=len(universe_tickers), total_elapsed_sec=time.time() - t_period_start,
        **pooled,
    )
    log(f"\nPERIOD {label} COMPLETE. Total elapsed: {(time.time()-t_period_start)/60:.1f} min")
    return result


# =====================================================================================
if __name__ == "__main__":
    log("=== TWO-PERIOD, MULTI-HORIZON STUDY STARTING (NIFTY BANK) ===")

    # Verified live Nifty Bank list (from the notebook's existing Bank section, NOT
    # re-derived here) - 14 stocks, no Bandhan Bank (removed from the index since the
    # original list-verification pass), Canara/Union/Yes Bank included.
    bank_symbols = ["AUBANK", "AXISBANK", "BANKBARODA", "CANBK", "FEDERALBNK", "HDFCBANK",
                    "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK", "PNB", "SBIN",
                    "UNIONBANK", "YESBANK"]
    bank_tickers = [s + ".NS" for s in bank_symbols]

    log("Checking ^NSEBANK earliest available date...")
    idx_probe = None
    for attempt in range(3):
        try:
            idx_probe = yf.download("^NSEBANK", start="2000-01-01", end="2026-08-09", auto_adjust=True)['Close'].squeeze()
            if len(idx_probe) > 0:
                break
        except Exception as e:
            log(f"  ^NSEBANK probe attempt {attempt+1} failed: {e}")
        time.sleep(3)
    log(f"^NSEBANK first valid: {idx_probe.first_valid_index()}, last valid: {idx_probe.last_valid_index()}")
    # Matches Nifty 50's own 2007-09-17 start almost exactly - round up to the same
    # 2008-01-01 Period A start used for Nifty 50, so fold years line up for comparison.
    period_a_start = "2008-01-01"

    log(f"Checking Period A universe coverage (>= {period_a_start}, cutoff 2009-01-01 for 'qualifies')...")
    probe = yf.download(bank_tickers, start=period_a_start, end="2009-06-01", auto_adjust=True)['Close']
    cutoff = pd.Timestamp("2009-01-01")
    qualify, exclude = [], []
    for t in bank_tickers:
        fv = probe[t].first_valid_index() if t in probe.columns else None
        # Retry individually before concluding a stock is genuinely too-young-listed -
        # the batch endpoint has repeatedly shown flaky false-negatives in this project
        # (e.g. KOTAKBANK, a bank listed well before 2008, initially came back empty here).
        if fv is None:
            for attempt in range(3):
                try:
                    df = yf.Ticker(t).history(start=period_a_start, end="2009-06-01", auto_adjust=True)['Close']
                    if len(df) > 0:
                        fv = df.index.min()
                        if fv.tz is not None:
                            fv = fv.tz_localize(None)
                        log(f"  {t}: batch download came back empty, retry succeeded ({len(df)} rows, first {fv})")
                        break
                except Exception as e:
                    log(f"  {t} retry attempt {attempt+1} failed: {e}")
                time.sleep(2)
        if fv is not None and fv <= cutoff:
            qualify.append(t)
        else:
            exclude.append((t, fv))
    log(f"Period A qualifying universe: {len(qualify)}/{len(bank_tickers)}")
    log(f"Excluded (listed too late): {exclude}")

    # Bank is a much smaller universe to begin with (14 vs Nifty 50's 50), so the
    # "is this too thin" threshold is proportionally tighter: 8 stocks, not 30.
    if len(qualify) >= 8:
        log(f"DECISION: {len(qualify)} >= 8 threshold -> using CURRENT-CONSTITUENTS-FILTERED "
            f"approach for Period A (not a historical-constituent-list fallback).")
        period_a_tickers = qualify
    else:
        log(f"DECISION: only {len(qualify)} < 8 -> would need a historical constituent list "
            f"fallback. THIS BRANCH WAS NOT NEEDED IN THIS RUN.")
        raise NotImplementedError("Historical fallback not needed - qualifying universe >= 8")

    if DRY_RUN:
        period_a_tickers = period_a_tickers[:6]
        log(f"DRY_RUN: Period A universe truncated to {period_a_tickers}")

    # Same fold years as the Nifty 50 Period A study - ^NSEBANK's start date (2007-09-17)
    # is close enough to ^NSEI's (2007-09-17, identical) that no boundary adjustment is needed.
    folds_a = [
        ("2008-01-01", "2012-12-31", "2013-01-01", "2013-12-31"),
        ("2008-01-01", "2013-12-31", "2014-01-01", "2014-12-31"),
        ("2008-01-01", "2014-12-31", "2015-01-01", "2015-12-31"),
        ("2008-01-01", "2015-12-31", "2016-01-01", "2016-12-31"),
        ("2008-01-01", "2016-12-31", "2017-01-01", "2017-12-31"),
        ("2008-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
        ("2008-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
    ]
    if DRY_RUN:
        folds_a = folds_a[:2]

    result_a = run_period("A_2008_2019_calm_BANK", folds_a, period_a_tickers,
                          feature_start_date="2008-01-01", feature_end_date="2020-01-01",
                          strict_no_lookback_before=None, index_ticker="^NSEBANK")
    with open(CKPT_PATH, "wb") as f:
        pickle.dump({"period_a": result_a, "complete_a": True, "complete_b": False}, f)
    log("Period A checkpoint saved.")

    # ---- Period B: full 14-stock universe, STRICT separation from 2020-01-01 ----
    folds_b = [
        ("2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ]
    if DRY_RUN:
        folds_b = folds_b[:2]
        period_b_tickers_for_dry = bank_tickers[:6]
    else:
        period_b_tickers_for_dry = bank_tickers

    result_b = run_period("B_2020_2026_recent_BANK", folds_b, period_b_tickers_for_dry,
                          feature_start_date="2020-01-01", feature_end_date="2026-08-09",
                          strict_no_lookback_before="2020-01-01", index_ticker="^NSEBANK")

    final = {"period_a": result_a, "period_b": result_b, "complete_a": True, "complete_b": True}
    with open(CKPT_PATH, "wb") as f:
        pickle.dump(final, f)
    log(f"\n=== ALL DONE. Total elapsed: {(time.time()-_t_script_start)/60:.1f} min ===")
    _log_file.close()
