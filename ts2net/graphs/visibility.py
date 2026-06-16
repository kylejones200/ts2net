"""
Multiplex and multiscale visibility graph builders.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .._validation import validate_series
from ..api import HVG, NVG
from .multiplex import MultiplexGraph, multiplex_graph


def multiplex_visibility_graph(
    x: NDArray[np.float64],
    *,
    hvg_kwargs: dict | None = None,
    nvg_kwargs: dict | None = None,
) -> tuple[MultiplexGraph, dict[str, nx.Graph]]:
    """
    Build HVG and NVG layers over the same series.

    Parameters
    ----------
    x : array (n,)
        Input time series.
    hvg_kwargs, nvg_kwargs : dict, optional
        Arguments passed to ``HVG`` / ``NVG`` builders.

    Returns
    -------
    multiplex : MultiplexGraph
        Layers ``"hvg"`` and ``"nvg"``.
    layers : dict
        Raw NetworkX graphs per layer.
    """
    x = validate_series(x, "multiplex_visibility_graph")
    hvg_kwargs = hvg_kwargs or {}
    nvg_kwargs = nvg_kwargs or {"limit": min(len(x), 2000)}

    G_hvg = HVG(**hvg_kwargs).build(x).as_networkx(force=True)
    G_nvg = NVG(**nvg_kwargs).build(x).as_networkx(force=True)

    layers = {"hvg": G_hvg, "nvg": G_nvg}
    return multiplex_graph(layers), layers
