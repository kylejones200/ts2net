from __future__ import annotations
import numpy as np
from typing import Tuple

from .core_rust import moran_i as _moran_rs
from .core_rust import knn as _knn_rs, radius as _radius_rs

try:
    from scipy.sparse import issparse
except Exception:
    issparse = None


def moran_i(y: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    return _moran_rs(y, W)


def _as2d(X: np.ndarray) -> np.ndarray:
    A = np.asarray(X, float)
    if A.ndim != 2:
        raise ValueError("X must be 2-D [n, d]")
    return A


def _as1d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, float).ravel()
    if a.ndim != 1:
        raise ValueError("y must be 1-D")
    return a


def knn_weights(coords: np.ndarray, k: int, row_std: bool = True) -> np.ndarray:
    X = _as2d(coords)
    idx, _ = _knn_rs(X, int(k) + 1)
    n = X.shape[0]
    W = np.zeros((n, n), float)
    for i in range(n):
        for j in idx[i]:
            if j == i:
                continue
            W[i, int(j)] = 1.0
    if row_std:
        s = W.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        W = W / s
    return W


def radius_weights(coords: np.ndarray, eps: float, row_std: bool = True) -> np.ndarray:
    X = _as2d(coords)
    nbrs = _radius_rs(X, float(eps))
    n = X.shape[0]
    W = np.zeros((n, n), float)
    for i, ns in enumerate(nbrs):
        for j in ns:
            if i == j:
                continue
            W[i, int(j)] = 1.0
    if row_std:
        s = W.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        W = W / s
    return W


def graph_weights_from_adj(A, row_std: bool = True) -> np.ndarray:
    if issparse(A):
        W = A.astype(float).toarray()
    else:
        W = np.asarray(A, float)
    np.fill_diagonal(W, 0.0)
    if row_std:
        s = W.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        W = W / s
    return W


def moran_i(y: np.ndarray, W: np.ndarray) -> Tuple[float, float]:
    x = _as1d(y)
    W = np.asarray(W, float)
    n = x.size
    z = x - x.mean()
    S0 = W.sum()
    num = float(z @ W @ z)
    den = float((z * z).sum())
    I = (n / S0) * (num / den) if S0 > 0 and den > 0 else np.nan
    Ez = -1.0 / (n - 1) if n > 1 else np.nan
    var_num = (W + W.T) / 2.0
    S1 = 0.5 * ((var_num**2).sum())
    S2 = float((((W.sum(axis=0) + W.sum(axis=1)) ** 2)).sum())
    b = n**2 - 3 * n + 3
    c = (n - 1) * (n - 2) * (n - 3)
    if den == 0 or S0 == 0 or n < 4:
        return I, np.nan
    EI = Ez
    VI = (n * S1 - n * S2 + 3 * (S0**2)) / (
        (S0**2) * (n - 1) * (n - 2) * (n - 3)
    ) - (1 / ((n - 1) ** 2))
    zscore = (I - EI) / np.sqrt(max(VI, 1e-12))
    return I, zscore


def geary_c(y: np.ndarray, W: np.ndarray) -> Tuple[float, float]:
    x = _as1d(y)
    W = np.asarray(W, float)
    n = x.size
    z = x - x.mean()
    S0 = W.sum()
    num = 0.0
    for i in range(n):
        for j in range(n):
            if W[i, j] == 0:
                continue
            num += W[i, j] * (x[i] - x[j]) ** 2
    den = 2.0 * float((z * z).sum())
    C = ((n - 1) / (2 * S0)) * (num / den) if S0 > 0 and den > 0 else np.nan
    return C, np.nan


def getis_ord_gstar(
    y: np.ndarray, W: np.ndarray, standardize: bool = True
) -> np.ndarray:
    x = _as1d(y)
    W = np.asarray(W, float)
    n = x.size
    mu = x.mean()
    sd = x.std(ddof=1) if standardize else 1.0
    sd = sd if sd > 0 else 1.0
    z = (x - mu) / sd
    G = np.zeros(n, float)
    for i in range(n):
        wi = W[i]
        s = wi.sum()
        num = float((wi * z).sum())
        den = np.sqrt(((n * np.sum(wi**2) - s**2) / (n - 1))) if n > 1 else 1.0
        G[i] = num / den if den > 0 else 0.0
    return G


def local_moran(y: np.ndarray, W: np.ndarray) -> np.ndarray:
    x = _as1d(y)
    W = np.asarray(W, float)
    z = x - x.mean()
    s2 = float((z * z).sum() / len(z))
    Ii = np.zeros_like(z)
    for i in range(len(z)):
        Ii[i] = z[i] * float(np.dot(W[i], z)) / s2 if s2 > 0 else 0.0
    return Ii
