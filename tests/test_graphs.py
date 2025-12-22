"""
Test script to verify graph implementations are working.
"""

import numpy as np
from ts2net.core.visibility import HVG, NVG


def test_graph_implementations():
    """Test HVG and NVG implementations with a simple time series."""
    # Create a simple time series
    ts = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0])

    # Test HVG
    hvg = HVG(weighted=True)
    G_hvg, A_hvg = hvg.fit_transform(ts)

    # Test NVG
    nvg = NVG(weighted=True)
    G_nvg, A_nvg = nvg.fit_transform(ts)


if __name__ == "__main__":
    test_graph_implementations()
