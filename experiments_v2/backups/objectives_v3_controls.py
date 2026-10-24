# -*- coding: utf-8 -*-
"""
Alternative risk-adjusted objectives (Sortino, Treynor) on top of the Step 0 per-fold caches.
Only the objective changes: regimes, episodes, means, volatilities, correlations, DTW matrices
and S_tilde all come from the cache built by run_step0.py.

Conventions (mirroring the existing Sharpe code): rf = MAR = 0, daily log returns,
annualisation sqrt(252/h), horizons from non-overlapping blocks of daily log returns.

Definitions (training data of fold k, regime R only):
  Sharpe   max w'mu / sqrt(w' Sigma w),        Sigma_std = D_s C_std D_s,  Sigma_dtw = D_s S~ D_s
  Sortino  max w'mu / sqrt(w' Sigma_down w),   Sigma_down = D_d C D_d,  d_i = sqrt(mean(min(r_i,0)^2))
           robustness: Estrada semicovariance  E[min(r_i,0) min(r_j,0)] (Standard only)
  Treynor  max w'mu / w'beta  s.t. w'beta >= 0.05,  beta_i = Sigma[i,m] / Sigma[m,m]
           Standard: sample covariance of [stocks, market] on the regime's training days
           DTW: extended (N+1) DTW matrix (stock-market DTW averaged over the same episodes),
                same kernel bandwidth h as the stock-only matrix, eigenvalue clipping, vol scaling
  All: sum w = 1, 0 <= w_i <= 0.4, weights frozen over the test period.
Evaluation on realised test returns: Sharpe, Sortino (downside deviation of test returns, MAR 0),
Treynor (annualised mean / average training-period sample beta of the portfolio held each day).
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

import dtwlib as L

MAX_W = 0.40
MIN_BETA = 0.05
NZ = 1e-4  # weight threshold for counting a holding as non-zero


# ------------------------------------------------------------------ training inputs

def downside_cov(d, corr):
    """D_d C D_d"""
    dv = d.loc[corr.index].values
    return pd.DataFrame(corr.values * np.outer(dv, dv), index=corr.index, columns=corr.columns)


def estrada_semicov(rets, stocks):
    """E[min(r_i,0) min(r_j,0)] over pairwise-complete training days (min 30), MAR 0."""
    neg = np.minimum(rets[stocks], 0.0)
    m = neg.notna().astype(float)
    num = neg.fillna(0.0).T @ neg.fillna(0.0)
    cnt = m.T @ m
    out = num / cnt.where(cnt >= 30)
    return out


def sample_betas(rets, mkt, stocks):
    """beta_i = cov(r_i, r_m) / var(r_m) from the sample covariance of [stocks, market] on the regime's
    training days (pairwise-complete, min 30 obs), i.e. Sigma[i,m]/Sigma[m,m] of the extended Standard matrix."""
    df = rets[stocks].copy()
    df["__MKT__"] = mkt.reindex(df.index).values
    cov = df.cov(min_periods=30)
    return cov.loc[stocks, "__MKT__"] / cov.loc["__MKT__", "__MKT__"]


def dtw_betas(dtw_in, mkt_name, mkt_rets, sigma_frac=0.5):
    """Extended (N+1) DTW matrix -> kernel with the stock-only bandwidth -> clip -> scale -> beta."""
    stocks = dtw_in["stocks"]
    D = dtw_in["avg_dtw"].loc[stocks, stocks].values
    dm = dtw_in["market_dtw"][mkt_name].loc[stocks].values
    n = len(stocks)
    Dx = np.zeros((n + 1, n + 1))
    Dx[:n, :n] = D
    Dx[:n, n] = dm
    Dx[n, :n] = dm
    S_ext, h, _ = L.kernel_similarity_psd(Dx, sigma_frac, base_median=dtw_in["base_median"])
    vol = np.append(dtw_in["vol"].loc[stocks].values, np.nanstd(mkt_rets.values, ddof=1))
    Sig = S_ext * np.outer(vol, vol)
    beta = pd.Series(Sig[:n, n] / Sig[n, n], index=stocks)
    return beta, dict(h=h, S_im=pd.Series(S_ext[:n, n], index=stocks), S_mm=S_ext[n, n],
                      min_eig_ext=float(np.linalg.eigvalsh(S_ext).min()))


def vol_kernel(vol, sigma_frac=0.5, market_vol=None):
    """Control (b): Gaussian kernel on volatility differences |s_i - s_j| with the DTW bandwidth rule
    (h = 0.5 x median positive distance), eigenvalue clipping at 1e-8. If market_vol is given, the
    (N+1) extended matrix is built with the stock-only h (as for DTW) and stock betas are returned."""
    v = vol.values
    D = np.abs(v[:, None] - v[None, :])
    S, h, base = L.kernel_similarity_psd(D, sigma_frac)
    S = pd.DataFrame(S, index=vol.index, columns=vol.index)
    if market_vol is None:
        return S, h
    n = len(v)
    Dx = np.zeros((n + 1, n + 1))
    Dx[:n, :n] = D
    Dx[:n, n] = Dx[n, :n] = np.abs(v - market_vol)
    Sx, _, _ = L.kernel_similarity_psd(Dx, sigma_frac, base_median=base)
    vv = np.append(v, market_vol)
    Sig = Sx * np.outer(vv, vv)
    return pd.Series(Sig[:n, n] / Sig[n, n], index=vol.index), dict(min_eig_ext=float(np.linalg.eigvalsh(Sx).min()))


def inverse_vol_weights(vol, max_w=MAX_W):
    """Control (c): w_i proportional to 1/sigma_i, capped at max_w with the excess redistributed."""
    w = (1.0 / vol) / (1.0 / vol).sum()
    for _ in range(100):
        over = w > max_w
        if not over.any():
            break
        excess = (w[over] - max_w).sum()
        w[over] = max_w
        w[~over] += excess * w[~over] / w[~over].sum()
    return w


# ------------------------------------------------------------------ optimisers

def optimize_ratio_cov(mu, cov):
    """Sharpe and Sortino: the original optimize_portfolio (SLSQP, equal-weight start, 40% cap)."""
    return L.optimize_portfolio(mu, cov, max_weight=MAX_W)


def clip_psd(M, floor=1e-8):
    """Eigenvalue clipping exactly as for S_tilde in dtw_to_covariance (floor 1e-8)."""
    vals, vecs = np.linalg.eigh(M.values)
    out = vecs @ np.diag(np.clip(vals, a_min=floor, a_max=None)) @ vecs.T
    return pd.DataFrame(out, index=M.index, columns=M.columns)


def optimize_treynor(mu, beta, max_w=MAX_W):
    """max w'mu / w'beta, sum w = 1, 0<=w<=max_w (0.4 primary, 0.1 robustness), w'beta >= 0.05.
    SLSQP (as for the other objectives) cross-checked with the exact Charnes-Cooper LP;
    the better of the two is returned with a record of which was used."""
    stocks = mu.index
    m, b = mu.values, beta.loc[stocks].values
    n = len(m)
    info = dict(method=None, slsqp_obj=np.nan, lp_obj=np.nan, lp_status=None, feasible=True)
    # exact LP: y = t w,  max m'y  s.t. b'y = 1, 1'y - t = 0, y_i - 0.4 t <= 0, 0.05 t - b'y <= 0
    c = np.append(-m, 0.0)
    A_eq = np.vstack([np.append(b, 0.0), np.append(np.ones(n), -1.0)])
    b_eq = np.array([1.0, 0.0])
    if n * max_w < 1 - 1e-12:
        info["feasible"] = False
        return None, info
    A_ub = np.hstack([np.eye(n), -max_w * np.ones((n, 1))])
    A_ub = np.vstack([A_ub, np.append(-b, MIN_BETA)])
    b_ub = np.zeros(n + 1)
    lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (n + 1), method="highs")
    info["lp_status"] = lp.status
    w_lp = None
    if lp.status == 0 and lp.x[-1] > 0:
        w_lp = lp.x[:n] / lp.x[-1]
        info["lp_obj"] = float(m @ w_lp / (b @ w_lp))
    res = minimize(lambda w: -(w @ m) / (w @ b), np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0, max_w)] * n,
                   constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                {'type': 'ineq', 'fun': lambda w: w @ b - MIN_BETA}])
    w_sq = res.x
    ok_sq = res.success and abs(w_sq.sum() - 1) < 1e-6 and w_sq @ b >= MIN_BETA - 1e-6 and w_sq.min() >= -1e-8
    if ok_sq:
        info["slsqp_obj"] = float(m @ w_sq / (b @ w_sq))
    if w_lp is None and not ok_sq:
        info["feasible"] = False
        return None, info
    if w_lp is not None and (not ok_sq or info["lp_obj"] > info["slsqp_obj"] + 1e-9 * max(1, abs(info["lp_obj"]))):
        w, info["method"] = w_lp, "LP"
    else:
        w, info["method"] = w_sq, "SLSQP"
    w = np.clip(w, 0, None)
    w = w / w.sum()
    info["w_beta"] = float(w @ b)
    info["beta_constraint_binds"] = bool(w @ b <= MIN_BETA + 1e-6)
    return pd.Series(w, index=stocks), info


# ------------------------------------------------------------------ evaluation

def portfolio_returns(weights_by_regime, cache, betas_by_regime=None):
    """Walk-forward exactly as the original: weights of the day's regime, NaN stock returns -> 0,
    days whose regime has no weights are skipped. Also returns the daily training-period
    portfolio beta (w' beta_sample of the regime held) for the Treynor metric."""
    tr, seq = cache["test_returns"], cache["test_regime_seq"]
    rets, betas = [], []
    for date in tr.index:
        r = seq.loc[date]
        if pd.isna(r) or r not in weights_by_regime:
            continue
        w = weights_by_regime[r]
        rets.append((date, float(np.dot(w, tr.loc[date, w.index].fillna(0)))))
        if betas_by_regime is not None:
            betas.append((date, float(np.dot(w, betas_by_regime[r].reindex(w.index).values))))
    s = pd.Series(dict(rets), dtype=float).sort_index()
    b = pd.Series(dict(betas), dtype=float).sort_index() if betas_by_regime is not None else None
    return s, b


def _agg(r, b, h):
    """Non-overlapping h-day blocks: returns summed (log), beta averaged."""
    if h == 1:
        return r, b
    n = (len(r) // h) * h
    g = np.arange(n) // h
    return r[:n].reshape(-1, h).sum(axis=1), b[:n].reshape(-1, h).mean(axis=1)


def metric_values(r, b, ppy, ddof=1):
    mean = r.mean()
    sd = r.std(ddof=ddof)
    dd = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return dict(sharpe=mean / sd * np.sqrt(ppy) if sd > 0 else np.nan,
                sortino=mean / dd * np.sqrt(ppy) if dd > 0 else np.nan,
                treynor=mean * ppy / b.mean() if b.mean() != 0 else np.nan,
                ann_return=mean * ppy, ann_vol=sd * np.sqrt(ppy), downside_dev_ann=dd * np.sqrt(ppy),
                avg_beta=b.mean(), n_obs=len(r))


def compare(std_r, std_b, dtw_r, dtw_b, h, seed, n_boot=5000, block=20):
    """Point metrics (pandas-style ddof=1, as compute_metrics_dict) and a paired block bootstrap
    (numpy ddof=0 inside resamples, as the original) of DTW - Standard for all three metrics.
    Days are aligned on the dates both portfolios traded; the same resampled days are used for
    every metric and both portfolios."""
    idx = std_r.index.intersection(dtw_r.index)
    sr, sb = std_r.loc[idx].values, std_b.loc[idx].values
    dr, db = dtw_r.loc[idx].values, dtw_b.loc[idx].values
    sr, sb = _agg(sr, sb, h)
    dr, db = _agg(dr, db, h)
    ppy = 252 / h
    ms, md = metric_values(sr, sb, ppy), metric_values(dr, db, ppy)
    out = dict(std=ms, dtw=md)
    n = len(sr)
    if n <= block:
        out["boot"] = None
        return out
    rng = np.random.RandomState(seed)
    nb = n // block
    keys = ("sharpe", "sortino", "treynor")
    diffs = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        starts = rng.randint(0, n - block, nb)
        ix = np.concatenate([np.arange(s, s + block) for s in starts])
        a = metric_values(dr[ix], db[ix], ppy, ddof=0)
        c = metric_values(sr[ix], sb[ix], ppy, ddof=0)
        for k in keys:
            diffs[k][i] = a[k] - c[k]
    boot = {}
    for k in keys:
        p = float((diffs[k] <= 0).mean())
        lo, hi = np.percentile(diffs[k], [2.5, 97.5])
        boot[k] = dict(p_one=p, p_two=L.two_sided(p), ci=(lo, hi))
    out["boot"] = boot
    out["n_obs"], out["n_blocks"] = n, nb
    return out


def alpha_regression(diff, mkt, hac_lags=5):
    """OLS of the daily DTW-minus-Standard return series on the market return:
    diff_t = a + b * mkt_t + e_t. Reports annualised alpha (x252), beta and the alpha p-value
    with Newey-West (HAC, 5 lags) standard errors, plus the plain OLS p-value."""
    import statsmodels.api as sm
    df = pd.concat([diff.rename("d"), mkt.rename("m")], axis=1).dropna()
    X = sm.add_constant(df["m"].values)
    ols = sm.OLS(df["d"].values, X).fit()
    hac = sm.OLS(df["d"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return dict(alpha_ann=float(ols.params[0] * 252), beta=float(ols.params[1]),
                alpha_p_hac=float(hac.pvalues[0]), alpha_p_ols=float(ols.pvalues[0]),
                beta_p_hac=float(hac.pvalues[1]), r2=float(ols.rsquared), n_days=int(len(df)),
                mean_diff_ann=float(df["d"].mean() * 252))


def bh_adjust(pvals):
    """Benjamini-Hochberg adjusted p-values (step-up, monotone)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * m / rank)
        adj[i] = prev
    return adj
