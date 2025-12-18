import numpy as np
from ts2net import batch_transform


def test_batch_hvg():
    X = [np.arange(100.0), np.sin(np.linspace(0, 6.28, 120))]
    out = batch_transform(X, builder="HVG")
    assert len(out) == 2
    for G, A in out:
        assert A.shape[0] > 0
        assert G.number_of_edges() > 0