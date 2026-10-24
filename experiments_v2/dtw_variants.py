# -*- coding: utf-8 -*-
"""
Exact DTW variants, vectorised over many series pairs at once (anti-diagonal recursion, so memory is
O(pairs x T) instead of O(pairs x T^2)). No window constraint (full warping, as in the primary spec).

  exactL1  R[p,q] = |x_p - y_q| + min(R[p-1,q], R[p,q-1], R[p-1,q-1]);  distance = R[T,T]
           (the quantity FastDTW approximates; exact <= FastDTW always)
  dtw2     same recursion with cost (x_p - y_q)^2;  distance = sqrt(R[T,T])
  sdtw     soft-DTW (Cuturi & Blondel 2017) with cost (x_p - y_q)^2 and soft-min
           softmin_g(a,b,c) = -g log(e^{-a/g} + e^{-b/g} + e^{-c/g});
           used as the divergence D(x,y) = sdtw(x,y) - (sdtw(x,x) + sdtw(y,y)) / 2  (>= 0)
GAMMA RULE (fixed 2026-10-21, before any soft-DTW result was computed):
  per episode, gamma = 0.1 x median over stock pairs and days of the squared same-day difference
  (x_{i,t} - x_{j,t})^2, computed on the series the distance is applied to (raw or normalised).
"""
import numpy as np

GAMMA_FRAC = 0.1
# Sakoe-Chiba band half-widths for windowed DTW, FIXED 2026-10-21 before any windowed result was computed.
BAND_WIDTHS = (1, 2, 3, 5)


def _recursion(A, B, kind, gamma=None, band=None):
    """A, B: (P, T) arrays (row k of A is compared with row k of B). Returns (P,) distances.
    band = w restricts the path to |p - q| <= w (Sakoe-Chiba); None = unconstrained."""
    P, T = A.shape
    INF = np.inf
    prev2 = np.full((P, T + 2), INF)
    prev2[:, 0] = 0.0                      # diagonal k = 0: R[0,0] = 0
    prev1 = np.full((P, T + 2), INF)       # diagonal k = 1: R[0,1] = R[1,0] = inf
    for k in range(2, 2 * T + 1):
        cur = np.full((P, T + 2), INF)
        lo, hi = max(1, k - T), min(T, k - 1)
        if band is not None:               # |p - q| = |2p - k| <= w
            lo, hi = max(lo, -(-(k - band) // 2)), min(hi, (k + band) // 2)
            if lo > hi:
                prev2, prev1 = prev1, cur
                continue
        p = np.arange(lo, hi + 1)
        a = A[:, p - 1]
        b = B[:, k - p - 1]
        cost = np.abs(a - b) if kind == "exactL1" else (a - b) ** 2
        up, left, diag = prev1[:, p - 1], prev1[:, p], prev2[:, p - 1]
        if kind == "sdtw":
            z = -np.logaddexp(np.logaddexp(-up / gamma, -left / gamma), -diag / gamma) * gamma
        else:
            z = np.minimum(np.minimum(up, left), diag)
        cur[:, p] = cost + z
        prev2, prev1 = prev1, cur
    out = prev1[:, T]
    return np.sqrt(out) if kind == "dtw2" else out


def episode_gamma(X):
    """0.1 x median squared same-day difference over all stock pairs (X: n x T)."""
    n = X.shape[0]
    i, j = np.triu_indices(n, 1)
    return GAMMA_FRAC * float(np.median((X[i] - X[j]) ** 2))


def pairwise(X, kind, gamma=None, chunk=400, band=None):
    """Symmetric (n x n) distance matrix between the rows of X (n x T)."""
    n = X.shape[0]
    i, j = np.triu_indices(n, 1)
    D = np.zeros((n, n))
    vals = np.empty(len(i))
    for s in range(0, len(i), chunk):
        sl = slice(s, s + chunk)
        vals[sl] = _recursion(X[i[sl]], X[j[sl]], kind, gamma, band)
    if kind == "sdtw":
        self_ = _recursion(X, X, kind, gamma, band)
        vals = vals - 0.5 * (self_[i] + self_[j])
    D[i, j] = vals
    D[j, i] = vals
    return D


def to_series(X, y, kind, gamma=None, band=None):
    """Distances between every row of X (n x T) and one series y (T,)."""
    Y = np.broadcast_to(y, X.shape)
    d = _recursion(X, np.ascontiguousarray(Y), kind, gamma, band)
    if kind == "sdtw":
        d = d - 0.5 * (_recursion(X, X, kind, gamma, band) + _recursion(y[None, :], y[None, :], kind, gamma, band)[0])
    return d


def path_lags(A, B):
    """Unconstrained exact L1 DTW between rows of A and B (P x T), keeping the full accumulated-cost
    matrices for backtracking. Returns the median of (q - p) along each optimal warping path (P,).
    Convention: a positive value means day p of A is matched to a LATER day q of B, i.e. B follows A."""
    P, T = A.shape
    R = np.full((P, T + 1, T + 1), np.inf)
    R[:, 0, 0] = 0.0
    for k in range(2, 2 * T + 1):                       # anti-diagonals p + q = k
        p = np.arange(max(1, k - T), min(T, k - 1) + 1)
        q = k - p
        c = np.abs(A[:, p - 1] - B[:, q - 1])
        R[:, p, q] = c + np.minimum(np.minimum(R[:, p - 1, q], R[:, p, q - 1]), R[:, p - 1, q - 1])
    lags = np.empty(P)
    for k in range(P):
        p, q, steps = T, T, []
        while p > 0 and q > 0:
            steps.append(q - p)
            cand = (R[k, p - 1, q - 1], R[k, p - 1, q], R[k, p, q - 1])
            m = int(np.argmin(cand))
            p, q = (p - 1, q - 1) if m == 0 else (p - 1, q) if m == 1 else (p, q - 1)
        lags[k] = np.median(steps)
    return lags


def xcorr_lag(a, b, max_lag=5):
    """Lag k in [-max_lag, max_lag] maximising corr(a_t, b_{t+k}); positive k means b follows a."""
    best, best_k = -np.inf, 0
    T = len(a)
    for k in range(-max_lag, max_lag + 1):
        x, y = (a[:T - k], b[k:]) if k >= 0 else (a[-k:], b[:T + k])
        if len(x) < 5 or x.std() == 0 or y.std() == 0:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if r > best:
            best, best_k = r, k
    return best_k, best
