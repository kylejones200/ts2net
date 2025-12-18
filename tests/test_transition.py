import numpy as np
from ts2net import TransitionNetwork


def test_tn_ordinal_size():
    x = np.arange(20.0)
    G, A = TransitionNetwork(symbolizer="ordinal", order=3, bins=5).fit_transform(x)
    # number of symbols equals order! for ordinal patterns
    from math import factorial

    assert A.shape[0] == factorial(3)
    assert G.number_of_edges() > 0


def test_tn_bins_equal_width():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    G, A = TransitionNetwork(symbolizer="equal_width", bins=7).fit_transform(x)
    assert A.shape == (7, 7)