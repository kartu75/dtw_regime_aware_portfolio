# -*- coding: utf-8 -*-
"""
Same expanding-window pipeline as expanding_window.py, applied to the Nifty Bank
universe instead of Nifty 50. All the shared functions are copied verbatim from the
notebook (not reimplemented) so methodology stays identical; the only new pieces are
the Bank-specific data pull and the fresh sigma sweep for this smaller universe.
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

CKPT_PATH = os.environ.get("CKPT_PATH", "expanding_window_bank_results.pkl")
LOG_PATH = os.environ.get("LOG_PATH", "expanding_window_bank_log.txt")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))

SEEDS = [0, 7, 21, 42, 99]
FEATURE_COLS = ['mean_return', 'rolling_vol_20d', 'drawdown_60d', 'momentum_5d']

_log_file = open(LOG_PATH, "a", encoding="utf-8")

def log(*a):
    msg = " ".join(str(x) for x in a)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line); sys.stdout.flush()
    _log_file.write(line + "\n"); _log_file.flush()

# =====================================================================================
# REUSED FUNCTIONS (verbatim from the notebook - identical to the Nifty 50 run)
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

def compute_metrics_dict(daily_log_returns):
    total_return = np.exp(daily_log_returns.sum()) - 1
    ann_return = daily_log_returns.mean() * 252
    ann_vol = daily_log_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    cum_ret = daily_log_returns.cumsum()
    max_dd = (cum_ret - cum_ret.cummax()).min()
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
        a_sample, b_sample = a[idx], b[idx]
        sharpe_a = (a_sample.mean() / a_sample.std()) * np.sqrt(252)
        sharpe_b = (b_sample.mean() / b_sample.std()) * np.sqrt(252)
        diffs.append(sharpe_a - sharpe_b)
    diffs = np.array(diffs)
    p_value = (diffs <= 0).mean()
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    return observed_diff, p_value, ci_lower, ci_upper, diffs

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

# =====================================================================================
def main():
    t_script_start = time.time()

    log("=== Nifty Bank data pipeline ===")
    log("NOTE: constituent list verified fresh from NSE archives (ind_niftybanklist.csv) "
        "on the day this ran, NOT hardcoded from the task description - the description's "
        "guessed list included Bandhan Bank (no longer a constituent) and omitted Canara "
        "Bank, Union Bank of India, and Yes Bank (all three currently ARE constituents).")

    bank_symbols = ["AUBANK", "AXISBANK", "BANKBARODA", "CANBK", "FEDERALBNK", "HDFCBANK",
                    "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK", "PNB", "SBIN",
                    "UNIONBANK", "YESBANK"]
    bank_tickers = [s + ".NS" for s in bank_symbols]
    log(f"Nifty Bank constituents ({len(bank_symbols)}):", bank_symbols)

    start_date = "2015-01-01"
    end_date = "2026-08-09"

    data = yf.download(bank_tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
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

    log("Final missing total:", int(data.isna().sum().sum()))
    for t in bank_tickers:
        fv = data[t].first_valid_index()
        n_missing = data[t].isna().sum()
        log(f"  {t}: first valid {fv}, missing days {n_missing} ({n_missing/len(data)*100:.1f}%)")

    log_returns = np.log(data / data.shift(1))

    # Nifty Bank index for regime detection (not the broad Nifty 50 index)
    for attempt in range(3):
        try:
            bank_index = yf.download("^NSEBANK", start=start_date, end=end_date, auto_adjust=True)['Close'].squeeze()
            if len(bank_index) > 0:
                break
        except Exception as e:
            log(f"^NSEBANK attempt {attempt+1} failed: {e}")
        time.sleep(3)
    log("^NSEBANK shape:", bank_index.shape)

    bank_log_ret = np.log(bank_index / bank_index.shift(1))
    roll_vol = bank_log_ret.rolling(20).std()
    cum_ret = bank_log_ret.cumsum()
    drawdown = cum_ret - cum_ret.rolling(60).max()
    momentum = bank_log_ret.rolling(5).mean()

    market_features = pd.DataFrame({
        'mean_return': bank_log_ret, 'rolling_vol_20d': roll_vol,
        'drawdown_60d': drawdown, 'momentum_5d': momentum
    }).dropna()
    log("market_features (bank) shape:", market_features.shape)

    # =================================================================================
    # FRESH SIGMA SWEEP for this smaller (~14-stock) universe
    # Calibrated on the same train window (2015-2021) as the original Nifty 50 sigma=0.5
    # calibration, for a like-for-like comparison.
    # =================================================================================
    log(f"\n{'='*70}\nSIGMA SWEEP - Nifty Bank universe, train 2015-01-01..2021-12-31\n{'='*70}")

    calib_train_features = market_features.loc["2015-01-01":"2021-12-31", FEATURE_COLS]
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
    calib_regime_raw = pd.Series(calib_states, index=calib_train_features.index).map(calib_regime_map)
    calib_regime_smoothed = smooth_regimes(calib_regime_raw, window=60)
    calib_episodes = extract_episodes(calib_regime_smoothed, min_length=10, gap_tolerance=3)
    log(f"  calibration episodes: {len(calib_episodes)},", calib_episodes['regime'].value_counts().to_dict())

    calib_regime_returns = {}
    calib_dtw_matrices, calib_dtw_stocks = {}, {}
    for regime in ['Bull', 'Sideways', 'Bear']:
        eps = calib_episodes[calib_episodes['regime'] == regime]
        if len(eps) == 0:
            continue
        date_ranges = [log_returns.loc[row['start']:row['end']] for _, row in eps.iterrows()]
        calib_regime_returns[regime] = pd.concat(date_ranges)
        episode_dtw = {}
        for idx, ep in eps.iterrows():
            dtw_matrix, stocks = compute_episode_dtw_matrix(log_returns.loc[ep['start']:ep['end']])
            if dtw_matrix is not None:
                episode_dtw[idx] = {'regime': regime, 'dtw_matrix': dtw_matrix, 'stocks': stocks}
        avg_dtw, stocks = aggregate_regime_dtw(episode_dtw, regime)
        if avg_dtw is not None:
            calib_dtw_matrices[regime] = avg_dtw
            calib_dtw_stocks[regime] = stocks

    sigma_sweep_results = {}
    for regime in calib_dtw_matrices:
        stocks = calib_dtw_stocks[regime]
        log(f"\n  {regime} regime - sigma sensitivity ({len(stocks)} stocks):")
        log(f"  {'sigma_frac':>10} {'sigma':>10} {'cond_number':>12} {'off_diag_std':>13}")
        rows = []
        for frac in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
            sigma, cond, off_std = evaluate_sigma(calib_dtw_matrices[regime], frac, calib_regime_returns[regime][stocks])
            rows.append((frac, sigma, cond, off_std))
            log(f"  {frac:>10.2f} {sigma:>10.4f} {cond:>12.1f} {off_std:>13.6f}")
        sigma_sweep_results[regime] = rows

    # =================================================================================
    # FOLD DEFINITIONS - identical years to the Nifty 50 run
    # =================================================================================
    folds_spec = [
        ("2015-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2015-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2015-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ]

    folds_limit = int(os.environ.get("FOLDS_LIMIT", "0"))
    if folds_limit:
        folds_spec = folds_spec[:folds_limit]
        log(f"FOLDS_LIMIT set: running only first {folds_limit} fold(s) for a sanity check")

    fold_results = []
    cap_bind_count, cap_total_count = 0, 0  # 40% weight cap binding diagnostic

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds_spec, start=1):
        t_fold_start = time.time()
        log(f"\n{'='*70}\nFOLD {fold_idx}/6: train {tr_start}..{tr_end}  test {te_start}..{te_end}\n{'='*70}")
        warnings_this_fold = []

        train_features = market_features.loc[tr_start:tr_end, FEATURE_COLS]
        test_features = market_features.loc[te_start:te_end, FEATURE_COLS]
        log(f"  train days={len(train_features)}  test days={len(test_features)}")

        best_model, best_score = None, -np.inf
        n_seed_failures = 0
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
            if score > best_score:
                best_score, best_model = score, m
        if best_model is None:
            raise RuntimeError(f"Fold {fold_idx}: all seeds failed")
        if n_seed_failures:
            w = f"{n_seed_failures}/{len(SEEDS)} HMM seeds failed to fit"
            warnings_this_fold.append(w); log("  WARNING:", w)
        model = best_model
        train_states = model.predict(train_features.values)

        regime_map, warn = build_regime_map(train_features, train_states)
        if warn:
            warnings_this_fold.append(warn); log("  WARNING:", warn)
        log("  regime_map:", regime_map)

        train_regime_raw = pd.Series(train_states, index=train_features.index).map(regime_map)
        train_regime_smoothed = smooth_regimes(train_regime_raw, window=60)

        test_states = causal_predict(model, test_features.values)
        test_regime_raw = pd.Series(test_states, index=test_features.index).map(regime_map)
        test_regime_smoothed = smooth_regimes(test_regime_raw, window=60)

        log("  train regime dist:", train_regime_smoothed.value_counts().to_dict())
        log("  test regime dist:", test_regime_smoothed.value_counts().to_dict())

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
            frozen_weights_std[regime] = optimize_portfolio(rets[included].mean(), rets[included].cov(min_periods=30))
        log("  standard frozen weights built for:", list(frozen_weights_std.keys()))

        # DTW loop, parallelized (small universe - 91 pairs max vs 1225 for Nifty 50,
        # so this should be much faster per episode than the Nifty 50 run)
        t_dtw0 = time.time()
        tasks = [(idx, log_returns.loc[ep['start']:ep['end']], ep['regime']) for idx, ep in train_episodes.iterrows()]
        n_workers = min(MAX_WORKERS, max(1, len(tasks)))
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
                        log(f"    episode {idx}: WORKER FAILED: {e}")
                        continue
                    n_completed += 1
                    if dtw_matrix is not None:
                        episode_dtw_results[ridx] = {'regime': regime, 'dtw_matrix': dtw_matrix, 'stocks': included_stocks}
        log(f"  DTW episode loop: {len(episode_dtw_results)}/{len(train_episodes)} usable, "
            f"{n_completed}/{len(tasks)} completed, wall time {time.time()-t_dtw0:.1f}s")

        frozen_weights_dtw = {}
        for regime in ['Bull', 'Sideways', 'Bear']:
            if regime_pooled_days.get(regime, 0) == 0:
                continue
            avg_dtw, stocks = aggregate_regime_dtw(episode_dtw_results, regime)
            if avg_dtw is None:
                warnings_this_fold.append(f"{regime}: DTW aggregation failed - no common stocks")
                continue
            stock_rets = train_regime_returns[regime][stocks]
            dtw_cov, sigma_used = dtw_to_covariance(avg_dtw, stock_rets, sigma_frac=0.5)
            frozen_weights_dtw[regime] = optimize_portfolio(stock_rets.mean(), dtw_cov)
            log(f"  {regime}: DTW cov built, sigma={sigma_used:.4f}, {len(stocks)} stocks")
        log("  DTW frozen weights built for:", list(frozen_weights_dtw.keys()))

        # 40% cap diagnostic: how often does the top holding actually sit at/near the cap?
        for wd in [frozen_weights_std, frozen_weights_dtw]:
            for w in wd.values():
                cap_total_count += 1
                if w.max() >= 0.395:
                    cap_bind_count += 1

        test_returns = log_returns.loc[test_features.index]
        regime_sequence = test_regime_smoothed.reindex(test_returns.index)

        std_rets, dtw_rets, bench_rets = [], [], []
        std_skipped = dtw_skipped = 0
        for date in test_returns.index:
            regime = regime_sequence.loc[date]
            bench_rets.append((date, bank_log_ret.loc[date] if date in bank_log_ret.index else np.nan))
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
            w = f"std_skipped={std_skipped}, dtw_skipped={dtw_skipped} test days"
            warnings_this_fold.append(w); log("  WARNING:", w)

        std_metrics = compute_metrics_dict(std_series) if len(std_series) else None
        dtw_metrics = compute_metrics_dict(dtw_series) if len(dtw_series) else None
        bench_metrics = compute_metrics_dict(bench_series) if len(bench_series) else None
        if std_metrics: print_metrics(std_metrics, "STANDARD")
        if dtw_metrics: print_metrics(dtw_metrics, "DTW")
        if bench_metrics: print_metrics(bench_metrics, "BENCHMARK")

        fold_results.append(dict(
            fold_idx=fold_idx, train_start=tr_start, train_end=tr_end, test_start=te_start, test_end=te_end,
            regime_map=regime_map, n_train_episodes=len(train_episodes),
            episode_regime_counts=train_episodes['regime'].value_counts().to_dict() if len(train_episodes) else {},
            regime_pooled_days=regime_pooled_days,
            std_series=std_series, dtw_series=dtw_series, bench_series=bench_series,
            std_metrics=std_metrics, dtw_metrics=dtw_metrics, bench_metrics=bench_metrics,
            warnings=warnings_this_fold, std_skipped=std_skipped, dtw_skipped=dtw_skipped,
            elapsed_sec=time.time() - t_fold_start,
        ))
        log(f"  Fold {fold_idx} done in {(time.time()-t_fold_start)/60:.1f} min. "
            f"Total elapsed: {(time.time()-t_script_start)/60:.1f} min")

        with open(CKPT_PATH, "wb") as f:
            pickle.dump({'fold_results': fold_results, 'sigma_sweep_results': sigma_sweep_results, 'complete': False}, f)

    log(f"\n40% weight-cap diagnostic: bound (>=0.395) in {cap_bind_count}/{cap_total_count} "
        f"frozen-weight vectors ({cap_bind_count/max(cap_total_count,1)*100:.1f}%)")

    # =================================================================================
    log(f"\n{'='*70}\nPOOLING RESULTS ACROSS {len(fold_results)} FOLDS (NIFTY BANK)\n{'='*70}")
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

    obs_diff, p_val, ci_low, ci_high, diffs = bootstrap_sharpe_diff(pooled_dtw, pooled_std)
    log(f"Bootstrap: observed_diff={obs_diff:.3f} p={p_val:.4f} CI=[{ci_low:.3f},{ci_high:.3f}]")

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
        fold_results=fold_results, sigma_sweep_results=sigma_sweep_results,
        pooled_std=pooled_std, pooled_dtw=pooled_dtw, pooled_bench=pooled_bench,
        pooled_std_metrics=pooled_std_metrics, pooled_dtw_metrics=pooled_dtw_metrics, pooled_bench_metrics=pooled_bench_metrics,
        bootstrap=dict(observed_diff=obs_diff, p_value=p_val, ci=(ci_low, ci_high)),
        per_fold_df=per_fold_df, cap_bind_count=cap_bind_count, cap_total_count=cap_total_count,
        bank_symbols=bank_symbols, complete=True, total_elapsed_sec=time.time() - t_script_start,
    )
    with open(CKPT_PATH, "wb") as f:
        pickle.dump(final, f)
    log(f"\nALL DONE. Total elapsed: {(time.time()-t_script_start)/60:.1f} min. Saved to {CKPT_PATH}")
    _log_file.close()


if __name__ == "__main__":
    main()
