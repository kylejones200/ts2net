import numpy as np
from ts2net import HVG, NVG, graph_summary


def test_hvg_monotone_chain():
    x = np.arange(50.0)  # strict increase
    G, A = HVG().fit_transform(x)
    # Only nearest neighbors should connect
    assert A.sum() == 2 * (len(x) - 1)
    degs = [d for _, d in G.degree()]
    assert max(degs) == 2
    assert min(degs) in (1, 2)


def test_nvg_basic_props():
    rng = np.random.default_rng(3363)
    x = rng.normal(size=120)
    G, A = NVG().fit_transform(x)
    s = graph_summary(G)
    assert s["n"] == len(x)
    assert s["m"] > 0
