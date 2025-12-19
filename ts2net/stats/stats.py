"""
stats.py - Statistical functions for time series and network analysis.

This module provides a comprehensive set of statistical functions for analyzing
time series data and networks, including correlation analysis, surrogate data
generation, recurrence analysis, and network motif analysis.
"""

from __future__ import annotations
from typing import Tuple, List, Optional, Union, Dict, Any
from ..core.stats_summary import CorrSigResult, CCFResult
import numpy as np
import math
import networkx as nx
from math import sqrt
import scipy.stats as st

# Try to import Rust functions, fall back to Python if not available
try:
    from ..core.core_rust import (
        fnn as _fnn_rs,
        cao as _cao_rs,
        surrogate_phase as _surr_phase_rs,
        iaaft as _iaaft_rs,
        corr_perm as _corr_perm_rs,
    )
except ImportError:
    # Rust functions not available, will use Python fallbacks
    _fnn_rs = None
    _cao_rs = None
    _surr_phase_rs = None
    _iaaft_rs = None
    _corr_perm_rs = None


def corr_t_pvalue(r: float, n: int) -> float:
    r = _clip_r(r)
    df = n - 2
    if df <= 0:
        return 1.0
    t = abs(r) * sqrt(df / max(1e-12, 1.0 - r * r))
    return float(2.0 * (1.0 - st.t.cdf(t, df)))


def corr_perm_pvalue(
    x: np.ndarray, y: np.ndarray, n_perm: int = 1000, rng=None
) -> float:
    """Correlation permutation test p-value. Uses Rust implementation if available."""
    if _corr_perm_rs is not None:
        seed = int(rng) if rng is not None else 3363
        return _corr_perm_rs(x, y, n_perm, seed)
    # Python fallback
    rng = np.random.default_rng(None if rng is None else rng)
    x = _nz(x)
    y = _nz(y)
    r_obs = float(np.corrcoef(x, y)[0, 1])
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r = float(np.corrcoef(x, yp)[0, 1])
        if abs(r) >= abs(r_obs):
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def ccf_perm_pvalue(
    x: np.ndarray, y: np.ndarray, max_lag: int = 20, n_perm: int = 1000, rng=None
) -> float:
    rng = np.random.default_rng(None if rng is None else rng)
    x = _nz(x)
    y = _nz(y)

    def best_r(a, b):
        n = len(a)
        br = -1.0
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                aa, bb = a[lag:], b[: n - lag]
            else:
                aa, bb = a[: n + lag], b[-lag:]
            if aa.size < 4:
                continue
            br = max(br, float(np.corrcoef(aa, bb)[0, 1]))
        return br

    r_obs = best_r(x, y)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r = best_r(x, yp)
        if r >= r_obs:
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


# ---------- helpers ----------


def _clip_r(r: float) -> float:
    return float(max(min(r, 0.999999), -0.999999))


def _zcrit(alpha: float) -> float:
    # erfcinv-based normal quantile. alpha in (0,1)
    # two-sided use: z = sqrt(2) * erfcinv(alpha)
    from math import erfcinv, sqrt

    return sqrt(2.0) * erfcinv(alpha)


def _nz(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, float)


# ---------- Fisher-z and tests ----------


def fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n < 4:
        raise ValueError("n >= 4 required")
    r = _clip_r(r)
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    zc = _zcrit(alpha)
    lo = z - zc * se
    hi = z + zc * se
    rlo = (math.exp(2 * lo) - 1) / (math.exp(2 * lo) + 1)
    rhi = (math.exp(2 * hi) - 1) / (math.exp(2 * hi) + 1)
    return float(rlo), float(rhi)


def corr_sig(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    x = _nz(x)
    y = _nz(y)
    r = float(np.corrcoef(x, y)[0, 1])
    lo, hi = fisher_z_ci(r, len(x), alpha)
    return {"r": r, "ci": (lo, hi), "sig": not (lo <= 0.0 <= hi)}


def ccf_sig(
    x: np.ndarray, y: np.ndarray, max_lag: int = 20, alpha: float = 0.05
) -> dict:
    x = _nz(x)
    y = _nz(y)
    best_r = -1.0
    best_lag = 0
    n = len(x)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = x[lag:]
            b = y[: n - lag]
        else:
            a = x[: n + lag]
            b = y[-lag:]
        if a.size < 4:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best_r:
            best_r = r
            best_lag = lag
    eff_n = n - abs(best_lag)
    lo, hi = fisher_z_ci(best_r, eff_n, alpha)
    return {"r": best_r, "lag": best_lag, "ci": (lo, hi), "sig": not (lo <= 0.0 <= hi)}


# ---------- multiple testing ----------


def fdr_bh(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p, float).ravel()
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    k = np.where(ranked <= thresh)[0]
    if k.size == 0:
        return np.zeros_like(p, dtype=bool)
    kmax = k.max()
    cut = ranked[kmax]
    out = p <= cut
    return out.reshape(p.shape)


# ---------- surrogates ----------


def surrogate_circular(x: np.ndarray, rng=None) -> np.ndarray:
    rng = np.random.default_rng(None if rng is None else rng)
    n = len(x)
    s = rng.integers(0, n)
    return np.roll(_nz(x), int(s))


def surrogate_phase(x: np.ndarray, rng=None) -> np.ndarray:
    """Phase randomization surrogate. Uses Rust implementation if available."""
    if _surr_phase_rs is not None:
        seed = int(rng) if rng is not None else 3363
        return _surr_phase_rs(x, seed)
    # Python fallback
    rng = np.random.default_rng(None if rng is None else rng)
    x = _nz(x)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    ph = np.angle(X)
    rnd = rng.uniform(-np.pi, np.pi, size=ph.shape)
    rnd[0] = ph[0]
    if x.size % 2 == 0 and rnd.size > 1:
        rnd[-1] = ph[-1]
    Y = mag * np.exp(1j * rnd)
    y = np.fft.irfft(Y, n=x.size)
    return np.real(y)


def iaaft(x: np.ndarray, iters: int = 50, rng=None) -> np.ndarray:
    """Iterative amplitude adjusted Fourier transform. Uses Rust implementation if available."""
    if _iaaft_rs is not None:
        seed = int(rng) if rng is not None else 3363
        return _iaaft_rs(x, iters, seed)
    # Python fallback
    rng = np.random.default_rng(None if rng is None else rng)
    x = _nz(x)
    y = np.sort(x)[np.argsort(rng.standard_normal(x.size))]
    Xmag = np.abs(np.fft.rfft(x))
    for _ in range(iters):
        Y = np.fft.rfft(y)
        Y = Xmag * np.exp(1j * np.angle(Y))
        y = np.fft.irfft(Y, n=x.size)
        y = np.sort(y)[np.argsort(np.argsort(x))]
    return y


# ---------- link significance on distance threshold ----------


def link_significance_from_D(
    D: np.ndarray, eps: float, alpha: float = 0.05, n_surr: int = 200, rng=None
) -> dict:
    rng = np.random.default_rng(None if rng is None else rng)
    D = _nz(D)
    n = D.shape[0]
    A = (D <= eps).astype(int)
    mask = np.triu(np.ones_like(A, dtype=bool), 1)
    pvals = np.ones_like(D, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d_obs = D[i, j]
            cnt = 0
            for _ in range(n_surr):
                perm = rng.permutation(n)
                d_s = D[perm[i], perm[j]]
                if d_s <= d_obs:
                    cnt += 1
            p = (cnt + 1) / (n_surr + 1)
            pvals[i, j] = p
            pvals[j, i] = p
    p_flat = pvals[mask]
    sig_flat = fdr_bh(p_flat, alpha=alpha)
    sig = np.zeros_like(A, dtype=int)
    sig[mask] = sig_flat.astype(int)
    sig = sig + sig.T
    sig = (sig > 0).astype(int) * A
    return {"sig_adj": sig, "pvals": pvals}


# ---------- motif z-scores ----------


def motif_zscore(G: nx.Graph, stat_fn, n_surr: int = 200, rng=None) -> dict:
    rng = np.random.default_rng(None if rng is None else rng)
    val = float(stat_fn(G))
    vals = []
    H = G.copy()
    m = H.number_of_edges()
    tries = 100 * m if m > 0 else 0
    for _ in range(n_surr):
        S = H.copy()
        if m > 0:
            try:
                nx.double_edge_swap(S, nswap=5 * m, max_tries=tries)
            except Exception:
                pass
        vals.append(float(stat_fn(S)))
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1) + 1e-12)
    z = (val - mu) / sd
    return {"value": val, "mu": mu, "sd": sd, "z": float(z)}


# ---------- RQA on recurrence matrices ----------


def rqa_measures(A: np.ndarray, lmin: int = 2) -> dict:
    B = (_nz(A) > 0).astype(int)
    n = B.shape[0]
    RR = float(B.sum() / (n * n))
    diag_lens = []
    for d in range(-n + 1, n):
        line = np.diag(B, k=d)
        run = 0
        for v in line:
            if v:
                run += 1
            elif run:
                diag_lens.append(run)
                run = 0
        if run:
            diag_lens.append(run)
    long = [L for L in diag_lens if L >= lmin]
    DET = 0.0 if len(diag_lens) == 0 else float(sum(long) / max(1, sum(diag_lens)))
    L = 0.0 if len(long) == 0 else float(np.mean(long))
    return {"RR": RR, "DET": DET, "L": L}


# ---------- partial correlation ----------


def partial_corr(X: np.ndarray) -> np.ndarray:
    X = _nz(X)
    X = (X - X.mean(axis=1, keepdims=True)) / (
        X.std(axis=1, ddof=1, keepdims=True) + 1e-12
    )
    S = np.corrcoef(X)
    P = np.linalg.pinv(S)
    D = np.diag(P)
    out = -P / np.sqrt(np.outer(D, D))
    np.fill_diagonal(out, 1.0)
    return out


def partial_corr_sig(R: np.ndarray, n: int, alpha: float = 0.05) -> np.ndarray:
    R = _nz(R)
    p = R.shape[0]
    Z = 0.5 * np.log((1 + R) / (1 - R + 1e-12))
    se = 1.0 / math.sqrt(max(n - p, 4))
    zc = _zcrit(alpha)
    lo = Z - zc * se
    hi = Z + zc * se
    Rlo = (np.exp(2 * lo) - 1) / (np.exp(2 * lo) + 1)
    Rhi = (np.exp(2 * hi) - 1) / (np.exp(2 * hi) + 1)
    sig = ~((Rlo <= 0.0) & (0.0 <= Rhi))
    np.fill_diagonal(sig, False)
    return sig


# ---------- usage shims ----------


def link_mask_from_distance(
    D: np.ndarray, target_density: float
) -> Tuple[np.ndarray, float]:
    D = _nz(D)
    tri = D[np.triu_indices_from(D, 1)]
    eps = float(np.quantile(tri, target_density))
    A = (D <= eps).astype(int)
    np.fill_diagonal(A, 0)
    A = np.maximum(A, A.T)
    return A, eps


def fisher_z_ci(r: float, n: int, alpha: float = 0.05):
    if n < 4:
        raise ValueError("n must be >= 4")
    r = max(min(r, 0.999999), -0.999999)
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    zcrit = abs(
        np.quantile(np.random.standard_normal(1_000_000), alpha / 2)
    )  # simple draw-free replacement below
    # better: exact
    from math import erfcinv, sqrt

    zcrit = sqrt(2) * erfcinv(alpha)
    lo = z - zcrit * se
    hi = z + zcrit * se
    rlo = (math.exp(2 * lo) - 1) / (math.exp(2 * lo) + 1)
    rhi = (math.exp(2 * hi) - 1) / (math.exp(2 * hi) + 1)
    return rlo, rhi


def corr_sig(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    lo, hi = fisher_z_ci(r, len(x), alpha)
    return {"r": r, "ci": (lo, hi), "sig": not (lo <= 0.0 <= hi)}


def ccf_sig(x: np.ndarray, y: np.ndarray, max_lag: int = 20, alpha: float = 0.05):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    best = (-1.0, 0)  # r, lag
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = x[lag:], y[: len(x) - lag]
        else:
            a, b = x[: len(x) + lag], y[-lag:]
        if len(a) < 4:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best[0]:
            best = (r, lag)
    lo, hi = fisher_z_ci(best[0], len(x) - abs(best[1]), alpha)
    return {"r": best[0], "lag": best[1], "ci": (lo, hi), "sig": not (lo <= 0.0 <= hi)}


def surrogate_circular(x: np.ndarray, k: int, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    n = len(x)
    s = rng.integers(0, n)
    return np.roll(x, s)


def surrogate_phase(x: np.ndarray, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    ph = np.angle(X)
    rand = rng.uniform(-np.pi, np.pi, size=ph.shape)
    rand[0] = ph[0]
    if rand.shape[0] > 1 and (len(x) % 2 == 0):
        rand[-1] = ph[-1]
    Y = mag * np.exp(1j * rand)
    y = np.fft.irfft(Y, n=len(x))
    return np.real(y)


def iAAFT(x: np.ndarray, iters: int = 50, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    x = np.asarray(x, float)
    y = np.sort(x)[np.argsort(rng.standard_normal(len(x)))]
    Xmag = np.abs(np.fft.rfft(x))
    for _ in range(iters):
        Y = np.fft.rfft(y)
        Y = Xmag * np.exp(1j * np.angle(Y))
        y = np.fft.irfft(Y, n=len(x))
        y = np.sort(y)[np.argsort(np.argsort(x))]
    return y


def link_significance_from_D(
    D: np.ndarray,
    eps: float,
    alpha: float = 0.05,
    n_surr: int = 200,
    method: str = "circular",
    rng=None,
):
    rng = np.random.default_rng(None if rng is None else rng)
    n = D.shape[0]
    tri = D[np.triu_indices(n, 1)]
    thr = eps
    A = (D <= thr).astype(int)
    # simple null by row/col shuffle on D
    counts = np.zeros_like(A, dtype=float)
    for _ in range(n_surr):
        perm = rng.permutation(n)
        Ds = D[perm][:, perm]
        As = (Ds <= thr).astype(int)
        counts += As
    p = 1.0 - counts / n_surr
    sig = (A == 1) & (p < alpha)
    return sig.astype(int), p


def motif_zscores(G: nx.Graph, stat_fn, n_surr: int = 200, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    val = stat_fn(G)
    vals = []
    H = G.copy()
    for _ in range(n_surr):
        Hs = H.copy()
        try:
            nx.double_edge_swap(
                Hs, nswap=5 * Hs.number_of_edges(), max_tries=100 * Hs.number_of_edges()
            )
        except Exception:
            pass
        vals.append(stat_fn(Hs))
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1) + 1e-12)
    z = (val - mu) / sd
    return {"value": val, "mu": mu, "sd": sd, "z": z}


def rqa_measures(A: np.ndarray):
    B = (A > 0).astype(int)
    n = B.shape[0]
    RR = B.sum() / (n * n)
    diag_lens = []
    for d in range(-n + 1, n):
        line = np.diag(B, k=d)
        run = 0
        for v in line:
            if v:
                run += 1
            elif run:
                diag_lens.append(run)
                run = 0
        if run:
            diag_lens.append(run)
    DET = (
        0.0
        if len(diag_lens) == 0
        else sum(l for l in diag_lens if l >= 2) / max(1, sum(diag_lens))
    )
    L = (
        float(np.mean([l for l in diag_lens if l >= 2]))
        if any(l >= 2 for l in diag_lens)
        else 0.0
    )
    return {"RR": RR, "DET": DET, "L": L}


def corr_sig(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    x = _nz(x)
    y = _nz(y)
    r = float(np.corrcoef(x, y)[0, 1])
    lo, hi = fisher_z_ci(r, len(x), alpha)
    sig = not (lo <= 0.0 <= hi)
    return CorrSigResult(r=r, n=len(x), ci=(lo, hi), alpha=alpha, sig=sig)


def ccf_sig(x: np.ndarray, y: np.ndarray, max_lag: int = 20, alpha: float = 0.05):
    x = _nz(x)
    y = _nz(y)
    best_r = -1.0
    best_lag = 0
    n = len(x)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = x[lag:]
            b = y[: n - lag]
        else:
            a = x[: n + lag]
            b = y[-lag:]
        if a.size < 4:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best_r:
            best_r = r
            best_lag = lag
    eff_n = n - abs(best_lag)
    lo, hi = fisher_z_ci(best_r, eff_n, alpha)
    sig = not (lo <= 0.0 <= hi)
    return CCFResult(
        r=best_r, lag=best_lag, n_eff=eff_n, ci=(lo, hi), alpha=alpha, sig=sig
    )


def false_nearest_neighbors(
    x: np.ndarray, m_max: int = 10, tau: int = 1, Rtol: float = 10.0, Atol: float = 2.0
):
    """False nearest neighbors analysis. Uses Rust implementation if available."""
    if _fnn_rs is not None:
        return _fnn_rs(x, m_max, tau, Rtol, Atol)
    # Python fallback
    x = np.asarray(x, float)
    n = x.size

    def embed(m):
        L = n - (m - 1) * tau
        E = np.empty((L, m))
        for i in range(m):
            E[:, i] = x[i * tau : i * tau + L]
        return E

    out = []
    for m in range(1, m_max):
        Xm = embed(m)
        Xmp = embed(m + 1)
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=2).fit(Xm)
        dist, idx = nn.kneighbors(Xm, n_neighbors=2)
        r = dist[:, 1]
        j = idx[:, 1]
        num = np.linalg.norm(Xmp[np.arange(Xmp.shape[0]), -1] - Xmp[j, -1], ord=2)
        fnn = np.mean((num / r > Rtol) | (np.abs(num) > Atol))
        out.append(fnn)
    return np.array(out)


def cao_e1_e2(x: np.ndarray, m_max: int = 10, tau: int = 1):
    """Cao's method for embedding dimension. Uses Rust implementation if available."""
    if _cao_rs is not None:
        return _cao_rs(x, m_max, tau)
    # Python fallback
    x = np.asarray(x, float)

    def embed(m):
        L = x.size - (m - 1) * tau
        E = np.empty((L, m))
        for i in range(m):
            E[:, i] = x[i * tau : i * tau + L]
        return E

    E1 = []
    E2 = []
    from sklearn.neighbors import NearestNeighbors

    for m in range(1, m_max):
        Xm = embed(m)
        Xmp = embed(m + 1)
        nn = NearestNeighbors(n_neighbors=2).fit(Xm)
        _, idx = nn.kneighbors(Xm, n_neighbors=2)
        j = idx[:, 1]
        num = np.linalg.norm(Xmp - Xmp[j], axis=1)
        den = np.linalg.norm(Xm - Xm[j], axis=1) + 1e-12
        E1.append(np.mean(num / den))
        if m > 1:
            E2.append(np.mean(np.abs(Xmp[:, -1] - Xmp[j, -1])))
    return np.array(E1), np.array(E2)


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    # Use ordinal patterns from core module
    from ..core import _ordinal_patterns
    
    # Apply delay if needed
    if delay > 1:
        x = x[::delay]
    pats = _ordinal_patterns(x, order)
    import math
    K = math.factorial(order)
    counts = np.bincount(pats, minlength=K).astype(float)
    p = counts / counts.sum() if counts.sum() > 0 else counts
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(H / np.log(K))


def rqa_full(A: np.ndarray, lmin: int = 2, vmin: int = 2) -> dict:
    B = (np.asarray(A) > 0).astype(int)
    n = B.shape[0]
    RR = float(B.sum() / (n * n))

    def runs(line):
        out = []
        c = 0
        for v in line:
            if v:
                c += 1
            elif c:
                out.append(c)
                c = 0
        if c:
            out.append(c)
        return out

    diag = []
    vert = []
    for d in range(-n + 1, n):
        diag += runs(np.diag(B, k=d))
    for j in range(B.shape[1]):
        vert += runs(B[:, j])
    DET = 0.0 if not diag else sum(L for L in diag if L >= lmin) / max(1, sum(diag))
    L = (
        0.0
        if not diag
        else (
            float(np.mean([L for L in diag if L >= lmin]))
            if any(L >= lmin for L in diag)
            else 0.0
        )
    )
    Lmax = 0 if not diag else int(max(diag))
    ENTR = 0.0
    dl = [L for L in diag if L >= lmin]
    if dl:
        counts = np.bincount(dl)
        p = counts[counts > 0] / counts.sum()
        ENTR = float(-(p * np.log(p)).sum())
    LAM = 0.0 if not vert else sum(V for V in vert if V >= vmin) / max(1, sum(vert))
    TT = (
        0.0
        if not vert
        else (
            float(np.mean([V for V in vert if V >= vmin]))
            if any(V >= vmin for V in vert)
            else 0.0
        )
    )
    return {
        "RR": RR,
        "DET": DET,
        "L": L,
        "Lmax": Lmax,
        "ENTR": ENTR,
        "LAM": LAM,
        "TT": TT,
    }
