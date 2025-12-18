import numpy as np
from .spatial import knn_weights, moran_i, geary_c, local_moran, getis_ord_gstar

def _coords_grid(n=10):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    return np.c_[xs.ravel(), ys.ravel()]


def test_basic():
    X = _coords_grid(8)
    W = knn_weights(X, k=8)
    y = X[:, 0].astype(float)
    I, z = moran_i(y, W)
    C, _ = geary_c(y, W)
    Ii = local_moran(y, W)
    Gs = getis_ord_gstar(y, W)
    assert np.isfinite(I)
    assert np.isfinite(C)
    assert Ii.shape[0] == y.size
    assert Gs.shape[0] == y.size
