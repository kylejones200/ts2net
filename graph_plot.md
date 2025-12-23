
Use one constructor pattern for all three families. Return a dataclass that holds the graph plus geometry plus build metadata.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import networkx as nx


EpsMode = Literal["fraction_max", "percentile"]
Metric = Literal["euclidean", "sqeuclidean", "manhattan", "chebyshev"]
KnnMode = Literal["none", "mutual", "directed"]


@dataclass(frozen=True)
class TSGraph:
    """Container for a time-series-derived graph plus geometry and build metadata.

    Attributes:
        graph: NetworkX graph with node and edge attributes.
        pos: Optional 2D or dD coordinates for nodes. Keys match graph nodes.
        meta: Build metadata such as method, parameters, and data shape.
    """
    graph: nx.Graph
    pos: Optional[Dict[Any, np.ndarray]]
    meta: Dict[str, Any]


def build_recurrence_graph(
    x: Union[np.ndarray, Iterable[float]],
    *,
    embed_dim: int = 3,
    delay: int = 1,
    eps: float = 0.2,
    eps_mode: EpsMode = "fraction_max",
    metric: Metric = "euclidean",
    exclude_diagonal: bool = True,
    theiler_window: int = 0,
    knn: int = 0,
    knn_mode: KnnMode = "none",
    weighted: bool = False,
    weight_mode: Literal["distance", "inverse_distance"] = "inverse_distance",
    return_pos: bool = True,
    node_id: Literal["time", "state"] = "time",
    dtype: np.dtype = np.float64,
) -> TSGraph:
    """Build an ε-recurrence network from a time series.

    You embed the series into state space, then connect nodes whose state vectors
    fall within an ε ball. This matches the style in recurrence-network figures
    where ε changes density.

    Args:
        x: 1D array-like of shape (n,) or array of shape (n, p) for multivariate.
        embed_dim: Embedding dimension m.
        delay: Delay τ in samples.
        eps: Threshold value. Interpreted by eps_mode.
        eps_mode: How to interpret eps.
            - "fraction_max": eps * max_pairwise_distance.
            - "percentile": eps is a percentile in [0, 100].
        metric: Distance metric in state space.
        exclude_diagonal: Remove self edges.
        theiler_window: Exclude edges for |i - j| <= theiler_window.
        knn: If > 0, also connect k nearest neighbors per node.
        knn_mode: How to apply knn edges.
            - "none": ignore knn parameter.
            - "mutual": keep only mutual kNN edges.
            - "directed": create directed kNN edges (returns DiGraph).
        weighted: Store weights on edges.
        weight_mode: Weight definition if weighted.
        return_pos: If True, return node positions as embedded vectors.
        node_id: Node labeling scheme.
            - "time": node id equals time index i.
            - "state": node id equals integer state index in embedding.
        dtype: Numeric dtype.

    Returns:
        TSGraph with:
            - graph nodes ordered by time index.
            - node attributes: "t" time index, "state" embedded vector.
            - edge attributes: "dist" and optionally "weight".
            - pos: embedded vectors (or None).
            - meta: method and parameters.
    """
    ...
```

This one function gets you the “ε = 10 percent, 20 percent, 30 percent” panels. You call it three times with different eps values. You draw with the same renderer.

Add two more constructors with the same return type.

```python
def build_ordinal_partition_graph(
    x: Union[np.ndarray, Iterable[float]],
    *,
    embed_dim: int = 4,
    delay: int = 1,
    directed: bool = True,
    weighted: bool = True,
    include_self_loops: bool = True,
    tie_break: Literal["stable", "jitter"] = "stable",
    return_pos: bool = False,
) -> TSGraph:
    """Build an ordinal partition network.

    Nodes represent permutation patterns. Directed edges represent observed
    transitions between patterns. Edge weight equals count or probability.

    Returns:
        TSGraph with node attribute "pattern" (tuple[int, ...]) and
        "count" (occurrence count). Edge attribute "weight".
    """
    ...


def build_visibility_graph(
    x: Union[np.ndarray, Iterable[float]],
    *,
    kind: Literal["hvg", "nvg"] = "hvg",
    directed: bool = False,
    weighted: bool = False,
    return_pos: bool = True,
) -> TSGraph:
    """Build a visibility graph (HVG or NVG).

    Returns:
        TSGraph with node attribute "t" and "x".
        pos defaults to 2D coordinates (t, x) so the plot matches the
        time series geometry.
    """
    ...
```

Then add one renderer that produces the look.

```python
def draw_tsgraph(
    tsgraph: TSGraph,
    *,
    ax=None,
    node_size: float = 10.0,
    edge_alpha: float = 0.15,
    node_alpha: float = 0.9,
    color_by: Literal["time", "community", "degree", "none"] = "time",
    cmap: str = "viridis",
    show: bool = True,
):
    """Draw graph with thin edges and colored nodes.

    Expects tsgraph.pos. Falls back to a layout if pos is None.
    """
    ...
```

That is the ideal shape. One build function per graph family. One draw function. A shared TSGraph container. Stable node and edge attributes across methods.

If you want a minimal first cut, implement only `build_recurrence_graph` plus `draw_tsgraph`. That alone lets you reproduce the epsilon sweep panels.


Your roadmap lines up with one clean public entrypoint and two private layers.

You want one user facing constructor for visibility graphs. You want one internal edge generator per method. You want one weight hook that runs after edge discovery.

That keeps backward compatibility. That keeps YAML simple. That keeps your factory simple.

Here is the ideal public function.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import networkx as nx


VisKind = Literal["hvg", "nvg", "bounded_nvg"]
WeightMode = Literal["none", "absdiff", "time_gap", "min_clearance", "slope"]


@dataclass(frozen=True)
class TSGraph:
    graph: Union[nx.Graph, nx.DiGraph]
    pos: Optional[Dict[int, np.ndarray]]
    meta: Dict[str, Any]


def build_visibility_graph(
    x: Union[np.ndarray, Iterable[float]],
    *,
    kind: VisKind = "hvg",
    directed: bool = False,
    weighted: Union[bool, WeightMode] = False,
    weight_mode: Optional[WeightMode] = None,
    limit: Optional[int] = None,
    max_edges: Optional[int] = None,
    max_edges_per_node: Optional[int] = None,
    max_memory_mb: Optional[int] = None,
    include_self_loops: bool = False,
    return_pos: bool = True,
    dtype: np.dtype = np.float64,
) -> TSGraph:
    """Construct HVG or NVG style graphs with optional direction and weights.

    Nodes map to time index i.
    Edge direction uses time forward orientation i -> j when i < j.
    Weights attach as edge attribute "weight".
    Distances and aux values attach as edge attributes when needed.

    Args:
        x: 1D series.
        kind: hvg, nvg, or bounded_nvg.
        directed: If True, emit a DiGraph and only time forward edges.
        weighted: False, True, or a string mode.
        weight_mode: Optional explicit mode. Overrides weighted when set.
        limit: Window limit for NVG variants.
        max_edges: Global cap for bounded_nvg.
        max_edges_per_node: Per node cap for bounded_nvg.
        max_memory_mb: Memory guard for bounded_nvg.
        include_self_loops: Rare. Default False.
        return_pos: If True, pos uses (t, x[t]) so plots match the series.
        dtype: Numeric dtype.

    Returns:
        TSGraph container with graph, pos, and meta.
    """
    x = np.asarray(x, dtype=dtype)
    n = int(x.shape[0])
    if n < 2:
        g = nx.DiGraph() if directed else nx.Graph()
        for i in range(n):
            g.add_node(i, t=i, x=float(x[i]))
        pos = {i: np.array([i, x[i]], dtype=dtype) for i in range(n)} if return_pos else None
        return TSGraph(graph=g, pos=pos, meta={"method": "visibility", "kind": kind})

    mode = _resolve_weight_mode(weighted=weighted, weight_mode=weight_mode)

    edges = _visibility_edges(
        x,
        kind=kind,
        limit=limit,
        max_edges=max_edges,
        max_edges_per_node=max_edges_per_node,
        max_memory_mb=max_memory_mb,
    )

    g: Union[nx.Graph, nx.DiGraph] = nx.DiGraph() if directed else nx.Graph()

    for i in range(n):
        g.add_node(i, t=i, x=float(x[i]))

    for i, j in edges:
        if not include_self_loops and i == j:
            continue
        u, v = (i, j) if not directed else ((i, j) if i < j else (j, i))
        if u == v and not include_self_loops:
            continue

        attrs: Dict[str, Any] = {}
        if mode != "none":
            attrs["weight"] = _vis_weight(x, i=u, j=v, mode=mode)
        g.add_edge(u, v, **attrs)

    pos = {i: np.array([i, x[i]], dtype=dtype) for i in range(n)} if return_pos else None
    meta = {
        "method": "visibility",
        "kind": kind,
        "directed": bool(directed),
        "weight_mode": mode,
        "limit": limit,
        "max_edges": max_edges,
        "max_edges_per_node": max_edges_per_node,
        "max_memory_mb": max_memory_mb,
        "n": n,
    }
    return TSGraph(graph=g, pos=pos, meta=meta)


def _resolve_weight_mode(*, weighted: Union[bool, WeightMode], weight_mode: Optional[WeightMode]) -> WeightMode:
    if weight_mode is not None:
        return weight_mode
    if weighted is True:
        return "absdiff"
    if weighted is False:
        return "none"
    return weighted


def _visibility_edges(
    x: np.ndarray,
    *,
    kind: VisKind,
    limit: Optional[int],
    max_edges: Optional[int],
    max_edges_per_node: Optional[int],
    max_memory_mb: Optional[int],
) -> np.ndarray:
    """Return array of undirected candidate edges as pairs (i, j)."""
    if kind == "hvg":
        return _hvg_edges(x)
    if kind == "nvg":
        return _nvg_edges(x, limit=limit)
    if kind == "bounded_nvg":
        return _bounded_nvg_edges(
            x,
            limit=limit,
            max_edges=max_edges,
            max_edges_per_node=max_edges_per_node,
            max_memory_mb=max_memory_mb,
        )
    raise ValueError(f"Unknown kind {kind}")


def _vis_weight(x: np.ndarray, *, i: int, j: int, mode: WeightMode) -> float:
    if i == j:
        return 0.0
    if mode == "absdiff":
        return float(abs(x[j] - x[i]))
    if mode == "time_gap":
        return float(abs(j - i))
    if mode == "slope":
        return float((x[j] - x[i]) / (j - i))
    if mode == "min_clearance":
        return float(_min_clearance(x, i=i, j=j))
    if mode == "none":
        return 1.0
    raise ValueError(f"Unknown weight mode {mode}")


def _min_clearance(x: np.ndarray, *, i: int, j: int) -> float:
    lo, hi = (i, j) if i < j else (j, i)
    if hi - lo <= 1:
        return float("inf")
    baseline = min(x[lo], x[hi])
    mid = x[lo + 1 : hi]
    return float(baseline - np.max(mid))
```

This function gives you your two biggest roadmap items in one place.

You add directed graphs with one switch. You add weight modes with one switch. You keep bounded NVG support as a kind.

You then place irreversibility in stats, not in the constructor. Put it on the graph wrapper class or a metrics module. Compute it from in degree and out degree when directed is True. Return 0 for undirected graphs.

You also keep your existing YAML pattern. Add two fields to your configs.

```python
@dataclass(frozen=True)
class HVGConfig:
    directed: bool = False
    weighted: Union[bool, WeightMode] = False

@dataclass(frozen=True)
class NVGConfig:
    directed: bool = False
    weighted: Union[bool, WeightMode] = False
    limit: Optional[int] = None
```

Factory dispatch stays the same. It passes config fields into build_visibility_graph.

Now your plots like the paper show up with the same renderer you already use for visibility graphs. The figure look comes from two choices. Use pos equals time and value. Use thin edges and small nodes. Use node color by time or by community.

Your tests map straight onto this API.

A sine wave with directed True yields near zero irreversibility. A ramp with directed True yields higher irreversibility. A constant signal yields zero. Weight mode checks stay simple. absdiff stays non negative. time_gap equals j minus i. slope flips sign on ramps. min_clearance spikes when a clean gap exists.

If you want, I can also write the metrics function that returns irreversibility_score plus weight summary and keeps the same shape as your current stats output.
