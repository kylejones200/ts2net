"""Authoritative definition of the role-feature schemas.

Column order for `ts2net.networks.roles.role_features_extended` previously
existed only implicitly, spread across `_role_features_basic` and the `hstack`
in `role_features_extended`. Nothing named the columns, so nothing could state
what column 9 meant. This module is the single place that does.

Two schemas are defined:

``ROLE_FEATURES_V1``
    The twelve columns `role_features_extended` returns today, in order. This
    is the shipped contract and is unchanged.

``ROLE_FEATURES_V2``
    A proposed non-redundant schema: one canonical representative of each
    independent signal. **Not yet produced by any public function.** It is
    recorded here so the redundancy is inspectable and testable before any
    default changes.

V1 carries twelve columns but only nine independent signals. Three columns are
exact aliases of three others, proved in ``docs/audits/role_features_audit.md``
and pinned by ``tests/test_role_features_characterization.py``:

===================  ==================  =========================================
alias                canonical           why they coincide
===================  ==================  =========================================
``ego_edges``        ``triangles``       an edge between two neighbours of *u*
                                         closes exactly one triangle through *u*
``ego_density``      ``clustering``      ``2*m_u/(k(k-1))`` with ``m_u = T(u)`` is
                                         the local clustering coefficient
``core_score``       ``core_number``     ``core/max(core)`` is a positive
                                         rescaling, and ``z(a*x) == z(x)``
===================  ==================  =========================================

Because the pipeline standardizes every column and then hands the matrix to a
Euclidean metric, each alias doubles its signal's weight in every distance.

A note on ``wedges``: it is *not* an alias but is fully determined by two
columns already present, as ``C(degree, 2) - triangles``. It is retained in v2
because it is quadratic in degree and therefore spans a direction neither of
its parents does.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple


class RoleFeature(NamedTuple):
    """One column of a role-feature matrix."""

    name: str
    """Canonical, stable identifier."""

    definition: str
    """Mathematical definition of the raw value, before standardization."""

    units: str
    """What the raw value is measured in, and its natural range."""

    provenance: str
    """Which implementation produces it."""


_DEGREE = RoleFeature(
    "degree",
    "k(u), the number of neighbours of u",
    "count, unbounded above; grows with density",
    "networkx Graph.degree",
)
_CLUSTERING = RoleFeature(
    "clustering",
    "2*T(u) / (k(u) * (k(u)-1)) for k >= 2, else 0",
    "ratio in [0, 1]",
    "networkx.clustering",
)
_PAGERANK = RoleFeature(
    "pagerank",
    "PageRank with damping 0.85",
    "probability; sums to 1 over nodes, so values shrink with graph size",
    "networkx.pagerank",
)
_EIGENVECTOR = RoleFeature(
    "eigenvector",
    "leading eigenvector of the adjacency matrix",
    "unit L2 norm over nodes, so values shrink with graph size; "
    "uniform on any vertex-transitive graph",
    "networkx.eigenvector_centrality_numpy",
)
_CORE_NUMBER = RoleFeature(
    "core_number",
    "largest k such that u survives in the k-core",
    "integer count; constant on Barabasi-Albert and Watts-Strogatz",
    "ts2net_rs.core_numbers, networkx.core_number fallback",
)
_BETWEENNESS = RoleFeature(
    "betweenness",
    "fraction of all-pairs shortest paths through u",
    "ratio in [0, 1] (normalized)",
    "networkx.betweenness_centrality",
)
_CLOSENESS = RoleFeature(
    "closeness",
    "reciprocal of mean shortest-path distance from u",
    "ratio in [0, 1]",
    "networkx.closeness_centrality",
)
_TRIANGLES = RoleFeature(
    "triangles",
    "T(u), the number of triangles containing u",
    "count, unbounded above",
    "ts2net_rs.triangles_per_node, networkx.triangles fallback",
)
_WEDGES = RoleFeature(
    "wedges",
    "C(k(u), 2) - T(u), the open two-paths centred on u",
    "count; determined by degree and triangles, but quadratic in degree",
    "roles._motif_features",
)
_EGO_EDGES = RoleFeature(
    "ego_edges",
    "edges among the neighbours of u",
    "count; equals triangles(u) exactly",
    "ts2net_rs.ego_edge_counts",
)
_EGO_DENSITY = RoleFeature(
    "ego_density",
    "2*m_u / (k(u) * (k(u)-1)) for k >= 2, else 0, m_u = edges among N(u)",
    "ratio in [0, 1]; equals clustering(u) exactly",
    "roles._egonet_density",
)
_CORE_SCORE = RoleFeature(
    "core_score",
    "core_number(u) / max(core_number)",
    "ratio in [0, 1]; a positive rescaling of core_number",
    "roles._core_periphery_scores",
)

#: The shipped twelve-column schema, in the order `role_features_extended`
#: returns. Unchanged; this records what exists.
ROLE_FEATURES_V1: tuple[RoleFeature, ...] = (
    _DEGREE,
    _CLUSTERING,
    _PAGERANK,
    _EIGENVECTOR,
    _CORE_NUMBER,
    _BETWEENNESS,
    _CLOSENESS,
    _TRIANGLES,
    _WEDGES,
    _EGO_EDGES,
    _EGO_DENSITY,
    _CORE_SCORE,
)

#: Proposed non-redundant schema: v1 minus the three aliases, order otherwise
#: preserved. Nine columns because there are nine independent signals -- the
#: width is a consequence, not a target. Not yet produced by any public
#: function.
ROLE_FEATURES_V2: tuple[RoleFeature, ...] = (
    _DEGREE,
    _CLUSTERING,
    _PAGERANK,
    _EIGENVECTOR,
    _CORE_NUMBER,
    _BETWEENNESS,
    _CLOSENESS,
    _TRIANGLES,
    _WEDGES,
)

#: Alias column -> the canonical column it duplicates. Every value is a member
#: of ``ROLE_FEATURES_V2``; every key is dropped from it.
REDUNDANT_V1_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "ego_edges": "triangles",
        "ego_density": "clustering",
        "core_score": "core_number",
    }
)

V1_NAMES: tuple[str, ...] = tuple(f.name for f in ROLE_FEATURES_V1)
V2_NAMES: tuple[str, ...] = tuple(f.name for f in ROLE_FEATURES_V2)


def index_of(name: str, schema: tuple[RoleFeature, ...] = ROLE_FEATURES_V1) -> int:
    """Column index of ``name`` in ``schema``.

    Lets callers address a column by meaning instead of by position.
    """
    for i, feature in enumerate(schema):
        if feature.name == name:
            return i
    raise KeyError(f"{name!r} is not in the schema; known: {[f.name for f in schema]}")
