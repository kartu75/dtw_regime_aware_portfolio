# -*- coding: utf-8 -*-
"""
Shared pipeline for the DTW regime-conditional covariance study (experiments_v2).

Functions marked "verbatim" are copied from two_period_study.py / expanding_window*.py /
dtw_baseline_v1_nifty50.ipynb (identical logic across all of them; only formatting differed).
The DTW distance is swappable through DTW_DISTANCES; the default "fastdtw_l1" is the exact
call used by every existing result: fastdtw(x, y, dist=euclidean) on 1-D log returns.

New in this module (not in the original code):
  - data loading from the frozen snapshot instead of yfinance
  - a seedable RNG for the block bootstrap (the original used the unseeded global RNG)
  - per-fold caches of every training-only input needed by later objectives
  - stock-to-market DTW distances per episode (computed separately, so the stock-stock
    DTW matrix is exactly the original one)
"""
import itertools
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean

SEEDS = [0, 7, 21, 42, 99]
FEATURE_COLS = ['mean_return', 'rolling_vol_20d', 'drawdown_60d', 'momentum_5d']
REGIMES = ['Bull', 'Sideways', 'Bear']
HERE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================================
# Data (frozen snapshot)
# =====================================================================================

def snapshot_dir(name="snapshot_20261021"):
    return os.path.join(HERE, "data", name)


def load_frame(key, snap=None):
    """Read a snapshot frame and apply the documented corrections in <snapshot>/corrections.json
    (dates to drop). The parquet files themselves are never modified."""
    snap = snap or snapshot_dir()
    df = pd.read_parquet(os.path.join(snap, f"{key}.parquet"))
    corr_path = os.path.join(snap, "corrections.json")
    if os.path.exists(corr_path):
        import json
        with open(corr_path, encoding="utf-8") as f:
            corr = json.load(f)
        drop = [pd.Timestamp(d) for d in corr.get(key, {}).get("drop_dates", [])]
        if drop:
            df = df.drop(index=[d for d in drop if d in df.index])
    return df


def load_index(key, snap=None):
    return load_frame(key, snap).iloc[:, 0]


def build_market_features(index_price):
    """verbatim feature construction (all scripts and the notebook)."""
    index_log_ret = np.log(index_price / index_price.shift(1))
    roll_vol = index_log_ret.rolling(20).std()
    cum_ret = index_log_ret.cumsum()
    drawdown = cum_ret - cum_ret.rolling(60).max()
    momentum = index_log_ret.rolling(5).mean()
    market_features = pd.DataFrame({
        'mean_return': index_log_ret, 'rolling_vol_20d': roll_vol,
        'drawdown_60d': drawdown, 'momentum_5d': momentum
    }).dropna()
    return market_features, index_log_ret


# =====================================================================================
# Regimes and episodes (verbatim)
# =====================================================================================

def smooth_regimes(regime_series, window=60):
    smoothed = regime_series.copy()
    for i in range(window, len(regime_series)):
        smoothed.iloc[i] = regime_series.iloc[i - window:i].mode()[0]
    return smoothed


def causal_predict(model, features):
    n = len(features)
    states = np.zeros(n, dtype=int)
    for t in range(1, n + 1):
        states[t - 1] = model.predict(features[:t])[-1]
    return states


def extract_episodes(regime_series, min_length=10, gap_tolerance=3):
    dates = regime_series.index
    labels = regime_series.values
    raw_runs = []
    start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_idx]:
            raw_runs.append({'regime': labels[start_idx], 'start': dates[start_idx],
                             'end': dates[i - 1], 'length': i - start_idx})
            start_idx = i
    if not raw_runs:
        return pd.DataFrame(columns=['regime', 'start', 'end', 'length'])
    bridged = [raw_runs[0]] + raw_runs[1:]
    merged = []
    i = 0
    while i < len(bridged):
        current = bridged[i].copy()
        j = i + 1
        while j + 1 < len(bridged) and bridged[j]['length'] <= gap_tolerance and bridged[j + 1]['regime'] == current['regime']:
            current['end'] = bridged[j + 1]['end']
            current['length'] = (dates.get_loc(current['end']) - dates.get_loc(current['start'])) + 1
            j += 2
        merged.append(current)
        i = j if j > i + 1 else i + 1
    return pd.DataFrame([ep for ep in merged if ep['length'] >= min_length])


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


def fit_hmm_multiseed(train_X, log=print):
    best_model, best_score, failures, scores = None, -np.inf, 0, []
    for seed in SEEDS:
        m = GaussianHMM(n_components=4, covariance_type="full", n_iter=5000, tol=1e-4, random_state=seed)
        try:
            m.fit(train_X)
            score = m.score(train_X)
        except (np.linalg.LinAlgError, ValueError) as e:
            failures += 1
            log(f"  seed={seed}: FIT FAILED ({type(e).__name__}: {e})")
            continue
        scores.append(score)
        if score > best_score:
            best_score, best_model = score, m
    return best_model, best_score, failures, scores


# =====================================================================================
# Optimizer (verbatim)
# =====================================================================================

def negative_sharpe(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -port_return / port_vol


def optimize_portfolio(mean_returns, cov_matrix, max_weight=0.40):
    n = len(mean_returns)
    result = minimize(
        negative_sharpe, np.array([1 / n] * n),
        args=(mean_returns.values, cov_matrix.values),
        method='SLSQP', bounds=tuple((0, max_weight) for _ in range(n)),
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )
    return pd.Series(result.x, index=mean_returns.index)


# =====================================================================================
# DTW (swappable distance) and kernel
# =====================================================================================

def _fastdtw_l1(x, y):
    """The original call: fastdtw on (n,1) arrays with scipy euclidean = |x-y| for 1-D."""
    distance, _ = fastdtw(x.reshape(-1, 1), y.reshape(-1, 1), dist=euclidean)
    return distance


DTW_DISTANCES = {"fastdtw_l1": _fastdtw_l1}


def compute_episode_dtw_matrix(episode_returns, min_valid_ratio=0.9, distance="fastdtw_l1"):
    """verbatim inclusion/cleaning rules; distance function looked up by name."""
    dist_fn = DTW_DISTANCES[distance]
    n_days = len(episode_returns)
    valid_counts = episode_returns.notna().sum()
    included_stocks = valid_counts[valid_counts >= min_valid_ratio * n_days].index.tolist()
    if len(included_stocks) < 2:
        return None, [], None
    clean_returns = episode_returns[included_stocks].dropna()
    n = len(included_stocks)
    dtw_matrix = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d = dist_fn(clean_returns.iloc[:, i].values, clean_returns.iloc[:, j].values)
        dtw_matrix[i, j] = d
        dtw_matrix[j, i] = d
    return pd.DataFrame(dtw_matrix, index=included_stocks, columns=included_stocks), included_stocks, clean_returns


def compute_episode_market_dtw(clean_returns, market_returns, distance="fastdtw_l1"):
    """DTW distance between each included stock and each market series, on the same cleaned
    episode rows used for the stock-stock matrix (minus any day the market itself is missing).
    Computed separately so the stock-stock matrix is untouched."""
    dist_fn = DTW_DISTANCES[distance]
    out = {}
    for name, m in market_returns.items():
        mk = m.reindex(clean_returns.index)
        ok = mk.notna().values
        vals = {c: dist_fn(clean_returns[c].values[ok], mk.values[ok]) for c in clean_returns.columns}
        out[name] = dict(dist=pd.Series(vals), n_days=int(ok.sum()))
    return out


def dtw_episode_worker(payload):
    idx, ep_returns, regime, market_returns, distance = payload
    t0 = time.time()
    dtw_matrix, included, clean = compute_episode_dtw_matrix(ep_returns, distance=distance)
    mkt = compute_episode_market_dtw(clean, market_returns, distance) if dtw_matrix is not None else None
    return idx, regime, dtw_matrix, included, mkt, time.time() - t0


def aggregate_regime_dtw(episode_dtw_results, target_regime):
    """verbatim."""
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


def kernel_similarity_psd(dtw_vals, sigma_frac=0.5, base_median=None):
    """Steps 1-2 of dtw_to_covariance (verbatim): Gaussian kernel with
    sigma = sigma_frac * median(positive distances), then eigenvalue clipping at 1e-8."""
    if base_median is None:
        base_median = np.median(dtw_vals[dtw_vals > 0])
    sigma = sigma_frac * base_median
    similarity = np.exp(-(dtw_vals ** 2) / (2 * sigma ** 2))
    eigvals, eigvecs = np.linalg.eigh(similarity)
    eigvals_clipped = np.clip(eigvals, a_min=1e-8, a_max=None)
    similarity_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return similarity_psd, sigma, base_median


def dtw_to_covariance(dtw_matrix, stock_returns, sigma_frac=0.5):
    """verbatim result: S_tilde * outer(sigma, sigma)."""
    similarity_psd, sigma, _ = kernel_similarity_psd(dtw_matrix.values, sigma_frac)
    stock_vols = stock_returns.std().values
    dtw_cov = similarity_psd * np.outer(stock_vols, stock_vols)
    return pd.DataFrame(dtw_cov, index=dtw_matrix.index, columns=dtw_matrix.columns), sigma


# =====================================================================================
# Metrics, horizons, bootstrap
# =====================================================================================

def compute_metrics_dict(daily_log_returns, periods_per_year=252):
    """verbatim (two-period version; identical to the others at periods_per_year=252)."""
    total_return = np.exp(daily_log_returns.sum()) - 1
    ann_return = daily_log_returns.mean() * periods_per_year
    ann_vol = daily_log_returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    cum_ret = daily_log_returns.cumsum()
    max_dd = (cum_ret - cum_ret.cummax()).min()
    return dict(total_return=total_return, ann_return=ann_return, ann_vol=ann_vol,
                sharpe=sharpe, max_dd=max_dd, n_days=len(daily_log_returns))


def aggregate_to_horizon(daily_returns, horizon):
    """verbatim: sum consecutive non-overlapping blocks of `horizon` daily log returns."""
    if horizon == 1:
        return daily_returns
    n = (len(daily_returns) // horizon) * horizon
    trimmed = daily_returns.iloc[:n]
    grouped = trimmed.groupby(np.arange(len(trimmed)) // horizon).sum()
    grouped.index = trimmed.index[horizon - 1::horizon]
    return grouped


def bootstrap_sharpe_diff(returns_a, returns_b, n_bootstrap=5000, block_size=20, periods_per_year=252, rng=None):
    """verbatim algorithm; rng=None uses the unseeded global RNG exactly as the original code."""
    randint = np.random.randint if rng is None else rng.randint
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
        block_starts = randint(0, n - block_size, n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in block_starts])
        a_sample, b_sample = a[idx], b[idx]
        sharpe_a = (a_sample.mean() / a_sample.std()) * np.sqrt(periods_per_year)
        sharpe_b = (b_sample.mean() / b_sample.std()) * np.sqrt(periods_per_year)
        diffs.append(sharpe_a - sharpe_b)
    diffs = np.array(diffs)
    p_value = (diffs <= 0).mean()
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    return observed_diff, p_value, ci_lower, ci_upper, diffs, n, n_blocks


def two_sided(p):
    return min(1.0, 2 * min(p, 1 - p))


# =====================================================================================
# One fold: regimes -> episodes -> covariances -> frozen weights -> test returns (+ cache)
# =====================================================================================

def run_fold(fold_idx, tr_start, tr_end, te_start, te_end, market_features, log_returns,
             bench_log_ret, market_returns, sigma_frac=0.5, distance="fastdtw_l1",
             std_nan_check=True, max_workers=10, log=print):
    """
    Mirrors run_fold() in two_period_study.py (std_nan_check=True) and the fold loop in
    expanding_window*.py / the notebook single split (std_nan_check=False). Returns
    (fold_record, cache). market_returns = {name: daily log return series} for the
    stock-to-market DTW distances and betas used later (never used for the Sharpe weights).
    """
    t_fold = time.time()
    warns = []
    train_features = market_features.loc[tr_start:tr_end, FEATURE_COLS]
    test_features = market_features.loc[te_start:te_end, FEATURE_COLS]
    log(f"FOLD {fold_idx}: train {tr_start}..{tr_end} ({len(train_features)} d)  test {te_start}..{te_end} ({len(test_features)} d)")

    model, best_score, n_fail, scores = fit_hmm_multiseed(train_features.values, log)
    if model is None:
        raise RuntimeError(f"fold {fold_idx}: all HMM seeds failed")
    if n_fail:
        warns.append(f"{n_fail}/{len(SEEDS)} HMM seeds failed to fit")
    train_states = model.predict(train_features.values)
    regime_map, w = build_regime_map(train_features, train_states)
    if w:
        warns.append(w)
    train_raw = pd.Series(train_states, index=train_features.index).map(regime_map)
    train_smoothed = smooth_regimes(train_raw, window=60)
    test_states = causal_predict(model, test_features.values)
    test_raw = pd.Series(test_states, index=test_features.index).map(regime_map)
    test_smoothed = smooth_regimes(test_raw, window=60)

    train_episodes = extract_episodes(train_smoothed, min_length=10, gap_tolerance=3)
    regime_pooled_days = {}
    for regime in REGIMES:
        eps = train_episodes[train_episodes['regime'] == regime] if len(train_episodes) else train_episodes
        pooled = int(eps['length'].sum()) if len(eps) else 0
        regime_pooled_days[regime] = pooled
        if pooled == 0:
            warns.append(f"{regime}: 0 pooled days - regime absent from train")
        elif pooled < 100:
            warns.append(f"{regime}: only {pooled} pooled days across {len(eps)} episode(s) - thin")

    train_regime_returns, train_regime_market = {}, {}
    for regime in REGIMES:
        if regime_pooled_days[regime] == 0:
            continue
        eps = train_episodes[train_episodes['regime'] == regime]
        train_regime_returns[regime] = pd.concat([log_returns.loc[r['start']:r['end']] for _, r in eps.iterrows()])
        train_regime_market[regime] = {k: pd.concat([s.loc[r['start']:r['end']] for _, r in eps.iterrows()])
                                       for k, s in market_returns.items()}

    # --- standard covariance (verbatim) ---
    w_std, std_inputs = {}, {}
    for regime, rets in train_regime_returns.items():
        valid_counts = rets.notna().sum()
        included = valid_counts[valid_counts >= 30].index.tolist()
        if len(included) < 2:
            warns.append(f"{regime}: <2 stocks pass min_periods=30 for standard cov")
            continue
        cov = rets[included].cov(min_periods=30)
        if std_nan_check and cov.isna().any().any():
            warns.append(f"{regime}: standard covariance has NaN entries - degenerate")
            continue
        mu = rets[included].mean()
        w_std[regime] = optimize_portfolio(mu, cov)
        vol = np.sqrt(np.diag(cov.values))
        std_inputs[regime] = dict(stocks=included, mu=mu, cov=cov,
                                  vol=pd.Series(vol, index=included),
                                  corr=pd.DataFrame(cov.values / np.outer(vol, vol), index=included, columns=included))

    # --- DTW (stock-stock verbatim; stock-market extra) ---
    t0 = time.time()
    tasks = [(idx, log_returns.loc[ep['start']:ep['end']], ep['regime'],
              {k: s.loc[ep['start']:ep['end']] for k, s in market_returns.items()}, distance)
             for idx, ep in train_episodes.iterrows()]
    ep_results = {}
    if tasks:
        with ProcessPoolExecutor(max_workers=min(max_workers, max(1, len(tasks)))) as ex:
            futs = {ex.submit(dtw_episode_worker, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                ridx, regime, mat, stocks, mkt, _ = fut.result()
                if mat is not None:
                    ep_results[ridx] = {'regime': regime, 'dtw_matrix': mat, 'stocks': stocks, 'market': mkt}
    # deterministic episode order for the average (the original used completion order;
    # this only changes floating-point summation order)
    ep_results = dict(sorted(ep_results.items()))
    dtw_secs = time.time() - t0

    w_dtw, dtw_inputs = {}, {}
    for regime in REGIMES:
        if regime_pooled_days.get(regime, 0) == 0:
            continue
        avg_dtw, stocks = aggregate_regime_dtw(ep_results, regime)
        if avg_dtw is None:
            warns.append(f"{regime}: DTW aggregation failed - no common stocks across episodes")
            continue
        stock_rets = train_regime_returns[regime][stocks]
        dtw_cov, sigma_used = dtw_to_covariance(avg_dtw, stock_rets, sigma_frac=sigma_frac)
        if np.isnan(dtw_cov.values).any():
            warns.append(f"{regime}: DTW covariance has NaN entries - degenerate")
            continue
        mu = stock_rets.mean()
        w_dtw[regime] = optimize_portfolio(mu, dtw_cov)
        s_psd, sigma_k, base_med = kernel_similarity_psd(avg_dtw.values, sigma_frac)
        # stock-to-market distances averaged over the same episodes (simple mean, as for stocks)
        reg_eps = [v for v in ep_results.values() if v['regime'] == regime]
        mkt_avg = {name: pd.concat([e['market'][name]['dist'].loc[stocks] for e in reg_eps], axis=1).mean(axis=1)
                   for name in market_returns}
        dtw_inputs[regime] = dict(stocks=stocks, mu=mu, vol=stock_rets.std(), avg_dtw=avg_dtw,
                                  S_tilde=pd.DataFrame(s_psd, index=stocks, columns=stocks),
                                  kernel_sigma=sigma_k, base_median=base_med, cov=dtw_cov,
                                  market_dtw=mkt_avg)

    # --- walk-forward over the test period (verbatim) ---
    test_returns = log_returns.loc[test_features.index]
    regime_seq = test_smoothed.reindex(test_returns.index)
    std_rets, dtw_rets, bench_rets = [], [], []
    std_skip = dtw_skip = 0
    for date in test_returns.index:
        regime = regime_seq.loc[date]
        bench_rets.append((date, bench_log_ret.loc[date] if date in bench_log_ret.index else np.nan))
        if pd.isna(regime) or regime not in w_std:
            std_skip += 1
        else:
            w = w_std[regime]
            std_rets.append((date, float(np.dot(w, test_returns.loc[date, w.index].fillna(0)))))
        if pd.isna(regime) or regime not in w_dtw:
            dtw_skip += 1
        else:
            w = w_dtw[regime]
            dtw_rets.append((date, float(np.dot(w, test_returns.loc[date, w.index].fillna(0)))))
    std_series = pd.Series(dict(std_rets), dtype=float).sort_index()
    dtw_series = pd.Series(dict(dtw_rets), dtype=float).sort_index()
    bench_series = pd.Series(dict(bench_rets), dtype=float).sort_index().dropna()
    if std_skip or dtw_skip:
        warns.append(f"std_skipped={std_skip}, dtw_skipped={dtw_skip} test days (regime unavailable)")

    record = dict(fold_idx=fold_idx, train_start=tr_start, train_end=tr_end, test_start=te_start, test_end=te_end,
                  train_days=len(train_features), test_days=len(test_features), regime_map=regime_map,
                  n_train_episodes=len(train_episodes), regime_pooled_days=regime_pooled_days,
                  std_series=std_series, dtw_series=dtw_series, bench_series=bench_series,
                  std_metrics=compute_metrics_dict(std_series) if len(std_series) else None,
                  dtw_metrics=compute_metrics_dict(dtw_series) if len(dtw_series) else None,
                  warnings=warns, std_skipped=std_skip, dtw_skipped=dtw_skip,
                  hmm_best_score=best_score, hmm_seed_scores=scores, dtw_wall_time=dtw_secs,
                  elapsed_sec=time.time() - t_fold)

    # downside deviation per stock (MAR = 0, the existing code uses rf = 0), training only
    def downside(rets):
        return np.sqrt((np.minimum(rets, 0.0) ** 2).mean())

    cache = dict(
        fold=record | dict(std_series=None, dtw_series=None, bench_series=None),
        hmm_model=model, regime_map=regime_map,
        train_raw=train_raw, train_smoothed=train_smoothed, test_raw=test_raw, test_smoothed=test_smoothed,
        episodes=train_episodes, regime_pooled_days=regime_pooled_days,
        train_regime_returns=train_regime_returns, train_regime_market=train_regime_market,
        downside_dev={r: downside(x) for r, x in train_regime_returns.items()},
        std_inputs=std_inputs, dtw_inputs=dtw_inputs,
        episode_dtw={k: dict(regime=v['regime'], dtw_matrix=v['dtw_matrix'], stocks=v['stocks'], market=v['market'])
                     for k, v in ep_results.items()},
        weights_sharpe=dict(std=w_std, dtw=w_dtw),
        test_returns=test_returns, test_market={k: s.reindex(test_returns.index) for k, s in market_returns.items()},
        test_regime_seq=regime_seq, bench_series=bench_series,
        std_series=std_series, dtw_series=dtw_series,
        distance=distance, sigma_frac=sigma_frac, std_nan_check=std_nan_check,
    )
    return record, cache


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
