import numpy as np
from ts2net import RecurrenceNetwork


def test_rn_eps_default():
    rng = np.random.default_rng(3363)
    x = np.sin(np.linspace(0, 8 * np.pi, 300)) + 0.05 * rng.standard_normal(300)
    G, A = RecurrenceNetwork(m=3, tau=2, rule="epsilon").fit_transform(x)
    assert A.shape[0] == 300 - (3 - 1) * 2
    assert A.sum() > 0


def test_rn_knn_degree():
    rng = np.random.default_rng(3363)
    x = rng.standard_normal(400)
    m, tau, k = 3, 1, 6
    G, A = RecurrenceNetwork(m=m, tau=tau, rule="knn", k=k).fit_transform(x)
    # Symmetric KNN should yield near 2k edges per node on average after symmetrize
    avg_deg = 2 * A.sum() / A.shape[0]
    assert avg_deg > 4
