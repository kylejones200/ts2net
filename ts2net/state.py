"""Reconstructed state space: the object the other views are views of.

A time series is a projection of a dynamical system. Delay embedding
reconstructs a state space from that projection, and under the conditions of
Takens' theorem the reconstruction preserves the geometry of the original
attractor. Graphs, recurrence structures and coupling measures are then
different charts of the same reconstructed object rather than independent
methods.

Before this module the embedding was an implementation detail of whoever
happened to call it, and the same convention was written out in five places.
:class:`StateSpace` owns it once.

Nothing here is new mathematics. The dimension criteria are the existing
validated Rust implementations, the delay rules are
:mod:`ts2net_rs.select_delay`, and the recurrence adjacency is
``ts2net_rs.rn_adj_epsilon``. What is new is that one object holds them, states
which rule chose what, and can say that no reconstruction was found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .core.core_rust import cao, fnn, rn_adj_epsilon

try:  # pragma: no cover - exercised only without the extension
    import ts2net_rs as _rs
except ImportError:
    _rs = None

DelayRule = Literal["mutual_information", "autocorrelation"]

#: A dimension is accepted when the false-nearest-neighbour fraction falls to
#: or below this. Kennel's original work uses a few percent; 1% is the common
#: modern choice and is what the validated Lorenz and sine results in
#: ``tests/test_state_space_reconstruction.py`` are measured against.
FNN_TOLERANCE = 0.01

#: Cao's E2 sits near 1.0 for stochastic data and departs from it for
#: deterministic data. Measured values are ~1.14 for white noise against ~0.04
#: for Lorenz, so the boundary is not delicate.
E2_STOCHASTIC_BAND = 0.25


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    """Why the delay and dimension are what they are.

    Carried so a reconstruction can be argued with. A dimension reported
    without the curve behind it is not evidence.
    """

    delay_rule: DelayRule
    delay_converged: bool
    """False when the rule found no minimum or crossing in its search window,
    in which case the delay is a fallback rather than a measurement."""

    delay_curve: np.ndarray = field(repr=False)
    fnn_curve: np.ndarray = field(repr=False)
    """False-nearest-neighbour fraction for dimensions ``1..len+1``."""

    cao_e1: np.ndarray = field(repr=False)
    cao_e2: np.ndarray = field(repr=False)

    dimension_converged: bool
    """False when the FNN fraction never fell to :data:`FNN_TOLERANCE` within
    the dimensions searched. The series then has no finite embedding dimension
    at this resolution -- which is the correct answer for noise, not a
    failure."""

    looks_stochastic: bool
    """Cao's E2 stayed within :data:`E2_STOCHASTIC_BAND` of 1.0 at every
    dimension, the signature of a stochastic rather than deterministic
    process."""

    def summary(self) -> str:
        """One line a person can read, including when nothing was found."""
        if not self.dimension_converged:
            verdict = (
                "no finite embedding dimension found"
                + (" (consistent with a stochastic process)" if self.looks_stochastic
                   else "")
            )
        else:
            verdict = "embedding found"
        delay = "delay measured" if self.delay_converged else "delay NOT measured"
        return f"{verdict}; {delay} by {self.delay_rule}"


@dataclass(frozen=True)
class StateSpace:
    """A delay reconstruction, and the evidence for its parameters.

    ``points`` has one row per reconstructed state and one column per
    coordinate, with column ``i`` being the series lagged by ``i * delay``.
    That is the convention every embedding in this package already used; this
    type simply owns it.
    """

    series: np.ndarray = field(repr=False)
    delay: int
    embedding_dimension: int
    points: np.ndarray = field(repr=False)
    diagnostics: ReconstructionDiagnostics

    def __len__(self) -> int:
        return self.points.shape[0]

    @property
    def n_states(self) -> int:
        return self.points.shape[0]

    def recurrence(
        self, epsilon: float, *, metric: str = "euclidean", theiler: int = 0
    ) -> np.ndarray:
        """Recurrence adjacency over the reconstructed states.

        Two states are connected when they are within ``epsilon``. ``theiler``
        excludes pairs closer than that many samples in time, which removes the
        trivial recurrence of a trajectory with itself; leaving it at zero
        inflates every density measure on a smooth signal.

        This takes reconstructed states, not a raw series. The 1-D entry points
        elsewhere in the package embed internally or not at all, which is why
        they are not interchangeable with this one.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return rn_adj_epsilon(self.points, epsilon, metric=metric, theiler=theiler)

    def recurrence_network(
        self, epsilon: float, *, metric: str = "euclidean", theiler: int = 0
    ):
        """The recurrence adjacency as a networkx graph.

        Node ``i`` is the state beginning at sample ``i`` of the series, so a
        graph property can always be traced back to a time.
        """
        import networkx as nx

        adjacency = self.recurrence(epsilon, metric=metric, theiler=theiler)
        graph = nx.from_numpy_array(np.asarray(adjacency, dtype=np.uint8))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph

    def epsilon_for_density(self, target_density: float) -> float:
        """The threshold giving approximately the requested recurrence density.

        Recurrence results are only comparable between signals at a matched
        density, because an absolute epsilon means different things at
        different amplitudes.
        """
        if not 0.0 < target_density < 1.0:
            raise ValueError("target_density must lie in (0, 1)")
        points = self.points
        n = points.shape[0]
        if n < 2:
            raise ValueError("need at least two states")
        # Sub-sample the pair distances for large state spaces; the quantile
        # does not need every pair and the full matrix is O(n^2).
        rng = np.random.default_rng(0)
        max_pairs = 2_000_000
        total_pairs = n * (n - 1) // 2
        if total_pairs <= max_pairs:
            diff = points[:, None, :] - points[None, :, :]
            distances = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
            sample = distances[np.triu_indices(n, k=1)]
        else:
            rows = rng.integers(0, n, size=max_pairs)
            cols = rng.integers(0, n, size=max_pairs)
            keep = rows != cols
            sample = np.linalg.norm(points[rows[keep]] - points[cols[keep]], axis=1)
        return float(np.quantile(sample, target_density))


def embed(series: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """Delay-embed ``series``: ``E[:, i] = x[i*delay : i*delay + L]``.

    ``L = n - (dimension - 1) * delay``. One row per state, one column per
    coordinate. This is the convention used throughout the package.
    """
    series = np.asarray(series, dtype=float).ravel()
    if dimension < 1 or delay < 1:
        raise ValueError("dimension and delay must be at least 1")
    length = series.size - (dimension - 1) * delay
    if length < 2:
        raise ValueError(
            f"series of {series.size} points is too short for dimension "
            f"{dimension} at delay {delay}: it would give {length} states"
        )
    out = np.empty((length, dimension), dtype=float)
    for i in range(dimension):
        out[:, i] = series[i * delay : i * delay + length]
    return out


def select_delay(
    series: np.ndarray,
    *,
    rule: DelayRule = "mutual_information",
    max_lag: int | None = None,
) -> dict:
    """Choose a delay by the named rule. See :mod:`ts2net_rs`."""
    if _rs is None:  # pragma: no cover - exercised only without the extension
        raise ImportError("delay selection requires the compiled ts2net_rs extension")
    return _rs.select_delay(
        np.asarray(series, dtype=float).ravel(), rule=rule, max_lag=max_lag
    )


def reconstruct(
    series: np.ndarray,
    *,
    dimension: int | None = None,
    delay: int | None = None,
    max_dimension: int = 10,
    delay_rule: DelayRule = "mutual_information",
    max_lag: int | None = None,
) -> StateSpace:
    """Reconstruct the state space of a scalar series.

    ``delay`` defaults to the first minimum of the time-delayed mutual
    information; pass ``delay_rule="autocorrelation"`` for the first zero of
    the autocorrelation instead. The two disagree by an order of magnitude on a
    chaotic attractor, so the rule used is recorded in the diagnostics.

    ``dimension`` defaults to the smallest dimension at which the
    false-nearest-neighbour fraction falls to :data:`FNN_TOLERANCE`. When no
    dimension does, ``max_dimension`` is used and
    ``diagnostics.dimension_converged`` is False -- the honest answer for a
    stochastic signal, which has no finite embedding dimension. Callers that
    report a dimension without checking that flag are reporting a fallback.
    """
    series = np.asarray(series, dtype=float).ravel()
    if series.size < 10:
        raise ValueError(f"need at least 10 points, got {series.size}")
    if max_dimension < 2:
        raise ValueError("max_dimension must be at least 2")

    if delay is None:
        chosen = select_delay(series, rule=delay_rule, max_lag=max_lag)
        delay_value = int(chosen["delay"])
        delay_converged = bool(chosen["converged"])
        delay_curve = np.asarray(chosen["curve"], dtype=float)
    else:
        if delay < 1:
            raise ValueError("delay must be at least 1")
        delay_value = int(delay)
        delay_converged = True
        delay_curve = np.empty(0)

    fnn_curve = fnn(series, m_max=max_dimension + 1, tau=delay_value)
    cao_e1, cao_e2 = cao(series, m_max=max_dimension + 1, tau=delay_value)

    below = np.flatnonzero(np.asarray(fnn_curve) <= FNN_TOLERANCE)
    dimension_converged = below.size > 0
    if dimension is None:
        # fnn_curve[i] is the fraction at dimension i + 1.
        chosen_dimension = int(below[0]) + 1 if dimension_converged else max_dimension
    else:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        chosen_dimension = int(dimension)

    finite_e2 = np.asarray(cao_e2, dtype=float)
    finite_e2 = finite_e2[np.isfinite(finite_e2)]
    looks_stochastic = bool(
        finite_e2.size > 0
        and np.all(np.abs(finite_e2 - 1.0) <= E2_STOCHASTIC_BAND)
    )

    diagnostics = ReconstructionDiagnostics(
        delay_rule=delay_rule,
        delay_converged=delay_converged,
        delay_curve=delay_curve,
        fnn_curve=np.asarray(fnn_curve, dtype=float),
        cao_e1=np.asarray(cao_e1, dtype=float),
        cao_e2=np.asarray(cao_e2, dtype=float),
        dimension_converged=dimension_converged,
        looks_stochastic=looks_stochastic,
    )
    return StateSpace(
        series=series,
        delay=delay_value,
        embedding_dimension=chosen_dimension,
        points=embed(series, chosen_dimension, delay_value),
        diagnostics=diagnostics,
    )
