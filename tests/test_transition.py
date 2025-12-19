import numpy as np
from ts2net.core.transition import TransitionNetwork


def test_tn_ordinal_size():
    x = np.arange(20.0)
    G, A = TransitionNetwork(symbolizer="ordinal", order=3, bins=5).fit_transform(x)
    # Check that we get a valid transition network
    assert A.shape[0] > 0
    assert A.shape[0] == A.shape[1]  # Square matrix
    assert G.number_of_edges() > 0


def test_tn_bins_equal_width():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    G, A = TransitionNetwork(symbolizer="equal_width", bins=7, order=1).fit_transform(x)
    assert A.shape == (7, 7)