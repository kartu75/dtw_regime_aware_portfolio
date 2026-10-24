# -*- coding: utf-8 -*-
"""Where does the small DTW-portfolio drift (vs the original runs) come from?
Uses the original single-split checkpoint (read only): its train episodes, averaged DTW
matrices and frozen weights. Recomputes from the frozen snapshot for one regime and compares
(1) returns data, (2) DTW distances, (3) weights from the original vs recomputed matrix."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dtwlib as L

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ck = L.load_pickle(os.path.join(ROOT, "dtw_baseline_checkpoint_v3.pkl"))
regime = sys.argv[1] if len(sys.argv) > 1 else "Sideways"

prices = L.load_frame("stocks_N50_2015")
log_returns = np.log(prices / prices.shift(1))
eps = ck["train_episodes"]
eps_r = eps[eps["regime"] == regime]
print(f"{regime}: {len(eps_r)} episodes, {int(eps_r['length'].sum())} days")

# (2) DTW distances per episode, averaged over common stocks (same as original)
res = {}
for idx, ep in eps_r.iterrows():
    mat, stocks, _ = L.compute_episode_dtw_matrix(log_returns.loc[ep["start"]:ep["end"]])
    if mat is not None:
        res[idx] = dict(regime=regime, dtw_matrix=mat, stocks=stocks)
avg_new, stocks_new = L.aggregate_regime_dtw(res, regime)
avg_old = ck["regime_dtw_matrices"][regime]
common = [s for s in avg_old.index if s in avg_new.index]
print(f"stocks: old {len(avg_old)}, new {len(avg_new)}, common {len(common)}")
diff = (avg_new.loc[common, common] - avg_old.loc[common, common]).abs()
rel = diff / avg_old.loc[common, common].replace(0, np.nan)
print(f"DTW avg matrix: max|diff|={diff.values.max():.3e}, max rel={np.nanmax(rel.values):.3e}, "
      f"share of pairs with rel diff > 1e-6: {(rel.values > 1e-6).mean():.3f}")

# (1) the input returns: standard weights from original episodes with snapshot data vs original weights
train_rets = pd.concat([log_returns.loc[r["start"]:r["end"]] for _, r in eps_r.iterrows()])
vc = train_rets.notna().sum()
inc = vc[vc >= 30].index.tolist()
w_std_new = L.optimize_portfolio(train_rets[inc].mean(), train_rets[inc].cov(min_periods=30))
w_std_old = ck["frozen_weights"][regime]
print(f"standard weights: max|diff|={(w_std_new - w_std_old.reindex(w_std_new.index)).abs().max():.3e}")

# (3) DTW weights: original matrix vs recomputed matrix, same returns
stock_rets = train_rets[stocks_new]
cov_new, _ = L.dtw_to_covariance(avg_new, stock_rets)
cov_oldmat, _ = L.dtw_to_covariance(avg_old.loc[stocks_new, stocks_new], stock_rets)
w_new = L.optimize_portfolio(stock_rets.mean(), cov_new)
w_oldmat = L.optimize_portfolio(stock_rets.mean(), cov_oldmat)
w_old = ck["frozen_weights_dtw"][regime].reindex(w_new.index)
print(f"DTW weights: recomputed vs original max|diff|={(w_new - w_old).abs().max():.3e}; "
      f"original-matrix-with-snapshot-returns vs original max|diff|={(w_oldmat - w_old).abs().max():.3e}")
print("top weights new:", w_new.sort_values(ascending=False).head(5).round(4).to_dict())
print("top weights old:", w_old.sort_values(ascending=False).head(5).round(4).to_dict())
