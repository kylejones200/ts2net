import numpy as np
from ts2net import NVG


def test_nvg_sweepline_monotone():
    x = np.arange(60.0)
    G, A = NVG().fit_transform(x)
    assert A.sum() == 2 * (len(x) - 1)