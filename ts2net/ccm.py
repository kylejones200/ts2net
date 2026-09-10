"""Convergent cross mapping: does one series' manifold contain another?

Convergent cross mapping (Sugihara et al., *Science* 2012) asks a question
correlation cannot. If X drives Y, then the state of Y at time t carries a
trace of X at time t, so the reconstructed manifold of Y can be used to
estimate X. The test is not merely that the estimate is good, but that it
*improves* as more of the manifold is filled in -- convergence. A fixed
relationship that does not converge is not evidence of coupling.

Direction reads backwards from intuition and is the most common way to get CCM
wrong: **to test whether X drives Y, cross-map X from the manifold of Y.**
Every function here takes ``cause`` and ``effect`` by name for that reason.

What this module deliberately separates
---------------------------------------
Five statements are routinely conflated, and CCM is only evidence for some of
them. :class:`CouplingVerdict` reports each separately:

``correlated``
    Linear association. Says nothing about direction or mechanism.
``predictive``
    Cross-map skill above chance. A shared driver produces this.
``convergent``
    Skill rises with library size. This is the CCM-specific claim.
``asymmetric``
    Skill differs by direction. Symmetric skill suggests a common driver or
    synchrony rather than one-way forcing.
``significant``
    Skill exceeds a surrogate null. Without this, strong autocorrelation alone
    can manufacture apparent skill.

Known traps, and what is done about them
----------------------------------------
CCM is easy to over-read. Shared periodicity, synchrony, autocorrelation,
short samples and a badly chosen delay all produce misleading skill.

* **Autocorrelation.** Neighbours in a smooth trajectory are usually
  neighbours *in time*, so an estimate built from them is really an
  interpolation of the target's own past. A Theiler window excludes library
  points within a given number of samples of the prediction point, and it
  defaults to the embedding delay rather than to zero.
* **Finite samples.** Skill is averaged over repeated random library draws and
  the spread is reported, so a difference smaller than the sampling spread
  cannot be read as a result.
* **Surrogates.** :func:`ccm_test` compares observed skill against IAAFT
  surrogates, which preserve the amplitude distribution and power spectrum of
  the original and therefore its autocorrelation. Skill that survives that
  null is not explained by linear structure alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .state import DelayRule, embed, reconstruct

try:  # pragma: no cover - exercised only without the extension
    import ts2net_rs as _rs
except ImportError:
    _rs = None

NullModel = Literal["iaaft", "shuffle"]

#: Minimum rise in skill from the smallest to the largest library, as a
#: fraction of the achievable range, for a relationship to count as convergent.
#: Convergence is the CCM-specific claim, so the bar is deliberately not zero:
#: a flat curve at high skill is a fixed relationship, not evidence of
#: coupling.
CONVERGENCE_TOLERANCE = 0.05


@dataclass(frozen=True)
class CrossMapResult:
    """Cross-map skill as a function of library size, in one direction."""

    library_sizes: np.ndarray = field(repr=False)
    skill: np.ndarray = field(repr=False)
    """Mean Pearson correlation between actual and cross-mapped values."""

    skill_spread: np.ndarray = field(repr=False)
    """Standard deviation across random library draws at each size. A skill
    difference smaller than this is not a finding."""

    embedding_dimension: int
    delay: int
    theiler: int
    n_predictions: int

    @property
    def terminal_skill(self) -> float:
        """Skill at the largest library."""
        return float(self.skill[-1])

    @property
    def initial_skill(self) -> float:
        return float(self.skill[0])

    @property
    def convergence_delta(self) -> float:
        """Rise in skill from the smallest to the largest library."""
        return self.terminal_skill - self.initial_skill

    @property
    def converged(self) -> bool:
        """Whether skill rose by more than :data:`CONVERGENCE_TOLERANCE`.

        Measured against the range still available above the initial skill, so
        a relationship starting at 0.9 is not required to rise as far as one
        starting at 0.1.
        """
        headroom = max(1.0 - self.initial_skill, 1e-12)
        return self.convergence_delta / headroom > CONVERGENCE_TOLERANCE


def _pairwise_to_library(
    query: np.ndarray, library: np.ndarray
) -> np.ndarray:
    diff = query[:, None, :] - library[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def cross_map(
    target: np.ndarray,
    manifold_series: np.ndarray,
    *,
    dimension: int,
    delay: int,
    library_size: int,
    theiler: int,
    rng: np.random.Generator,
) -> float:
    """Estimate ``target`` from the manifold of ``manifold_series``, once.

    Returns the Pearson correlation between the actual and estimated target,
    or NaN when the library is too small to supply ``dimension + 1`` neighbours
    outside the Theiler window.
    """
    points = embed(manifold_series, dimension, delay)
    n_states = points.shape[0]
    # State i spans samples i .. i + (dimension-1)*delay; the target is taken
    # at the state's final sample, which is the present in a delay embedding.
    target_index = np.arange(n_states) + (dimension - 1) * delay
    actual = np.asarray(target, dtype=float)[target_index]

    k = dimension + 1
    if library_size < k + 1 or library_size > n_states:
        return float("nan")

    library = rng.choice(n_states, size=library_size, replace=False)
    distances = _pairwise_to_library(points, points[library])

    # Temporal exclusion: a library point too close in time to the prediction
    # point makes the estimate an interpolation of the target's own past.
    time_gap = np.abs(np.arange(n_states)[:, None] - library[None, :])
    distances = np.where(time_gap <= theiler, np.inf, distances)

    usable = np.sum(np.isfinite(distances), axis=1) >= k
    if not np.any(usable):
        return float("nan")

    order = np.argsort(distances, axis=1)[:, :k]
    rows = np.arange(n_states)[:, None]
    nearest = distances[rows, order]
    neighbour_states = library[order]

    # Sugihara's exponential weights, relative to the closest neighbour.
    closest = nearest[:, [0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.exp(-nearest / np.where(closest > 0, closest, 1.0))
    weights = np.where(np.isfinite(weights), weights, 0.0)
    total = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)

    estimated = np.sum(weights * actual[neighbour_states], axis=1)

    valid = usable & np.isfinite(estimated)
    if valid.sum() < 3:
        return float("nan")
    a, b = actual[valid], estimated[valid]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cross_map_curve(
    cause: np.ndarray,
    effect: np.ndarray,
    *,
    dimension: int,
    delay: int,
    library_sizes: np.ndarray | None = None,
    n_draws: int = 8,
    theiler: int | None = None,
    seed: int = 3363,
) -> CrossMapResult:
    """Skill of estimating ``cause`` from the manifold of ``effect``.

    Rising skill is evidence that ``cause`` drives ``effect``. Note the
    direction: the manifold used is the *effect*'s.
    """
    cause = np.asarray(cause, dtype=float).ravel()
    effect = np.asarray(effect, dtype=float).ravel()
    if cause.size != effect.size:
        raise ValueError(
            f"series must be the same length, got {cause.size} and {effect.size}"
        )
    if dimension < 1 or delay < 1:
        raise ValueError("dimension and delay must be at least 1")

    n_states = effect.size - (dimension - 1) * delay
    if n_states < dimension + 4:
        raise ValueError(
            f"series is too short: dimension {dimension} at delay {delay} "
            f"leaves {n_states} states"
        )
    exclusion = delay if theiler is None else theiler

    if library_sizes is None:
        smallest = dimension + 3
        library_sizes = np.unique(
            np.geomspace(smallest, n_states, num=10).astype(int)
        )
    library_sizes = np.asarray(library_sizes, dtype=int)
    library_sizes = library_sizes[
        (library_sizes >= dimension + 2) & (library_sizes <= n_states)
    ]
    if library_sizes.size < 2:
        raise ValueError("need at least two usable library sizes")

    rng = np.random.default_rng(seed)
    means, spreads = [], []
    for size in library_sizes:
        draws = [
            cross_map(
                cause,
                effect,
                dimension=dimension,
                delay=delay,
                library_size=int(size),
                theiler=exclusion,
                rng=rng,
            )
            for _ in range(n_draws)
        ]
        finite = np.asarray([d for d in draws if np.isfinite(d)])
        means.append(float(finite.mean()) if finite.size else float("nan"))
        spreads.append(float(finite.std(ddof=1)) if finite.size > 1 else 0.0)

    return CrossMapResult(
        library_sizes=library_sizes,
        skill=np.asarray(means),
        skill_spread=np.asarray(spreads),
        embedding_dimension=dimension,
        delay=delay,
        theiler=exclusion,
        n_predictions=n_states,
    )


@dataclass(frozen=True)
class CouplingVerdict:
    """Five separate claims about a pair of series, reported separately."""

    forward: CrossMapResult
    """Estimating ``x`` from ``y``'s manifold: evidence that x drives y."""

    backward: CrossMapResult
    """Estimating ``y`` from ``x``'s manifold: evidence that y drives x."""

    pearson_r: float
    forward_p: float | None = None
    backward_p: float | None = None
    n_surrogates: int = 0
    null_model: NullModel | None = None
    alpha: float = 0.05

    def claims(self) -> dict[str, bool | None]:
        """Each statement, evaluated independently. None means not tested."""
        skill = self.forward.terminal_skill
        spread = float(self.forward.skill_spread[-1])
        return {
            "correlated": abs(self.pearson_r) > 0.1,
            "predictive": np.isfinite(skill) and skill > 0.1,
            "convergent": self.forward.converged,
            "asymmetric": abs(skill - self.backward.terminal_skill)
            > max(2.0 * spread, 0.05),
            "significant": None
            if self.forward_p is None
            else self.forward_p < self.alpha,
        }

    def summary(self) -> str:
        """A sentence that does not claim more than was established."""
        claims = self.claims()
        skill = self.forward.terminal_skill
        if not claims["predictive"]:
            return (
                f"no cross-map skill (rho={skill:.3f}); no evidence that the "
                f"first series drives the second"
            )
        parts = [f"cross-map skill rho={skill:.3f}"]
        parts.append(
            f"converges (+{self.forward.convergence_delta:.3f} over the library)"
            if claims["convergent"]
            else "does NOT converge, so the skill is a fixed relationship "
            "rather than evidence of coupling"
        )
        parts.append(
            "asymmetric"
            if claims["asymmetric"]
            else f"symmetric with the reverse direction "
            f"(rho={self.backward.terminal_skill:.3f}), consistent with a "
            f"shared driver rather than one-way forcing"
        )
        if claims["significant"] is None:
            parts.append("not tested against a surrogate null")
        elif claims["significant"]:
            parts.append(
                f"survives {self.n_surrogates} {self.null_model} surrogates "
                f"(p={self.forward_p:.4f})"
            )
        else:
            parts.append(
                f"does NOT survive {self.n_surrogates} {self.null_model} "
                f"surrogates (p={self.forward_p:.4f}), so linear structure "
                f"alone can explain it"
            )
        return "; ".join(parts)


def _surrogate(series: np.ndarray, null: NullModel, seed: int) -> np.ndarray:
    if null == "shuffle":
        return np.random.default_rng(seed).permutation(series)
    if null == "iaaft":
        if _rs is None:  # pragma: no cover
            raise ImportError("the iaaft null requires the compiled extension")
        return np.asarray(_rs.iaaft(series, 100, seed), dtype=float)
    raise ValueError(f"unknown null model {null!r}; expected 'iaaft' or 'shuffle'")


def ccm(
    cause: np.ndarray,
    effect: np.ndarray,
    *,
    dimension: int | None = None,
    delay: int | None = None,
    delay_rule: DelayRule = "mutual_information",
    n_draws: int = 8,
    theiler: int | None = None,
    seed: int = 3363,
) -> CouplingVerdict:
    """Convergent cross mapping in both directions, without a significance test.

    ``dimension`` and ``delay`` default to a reconstruction of ``effect``, the
    series whose manifold carries the forward test.

    Use :func:`ccm_test` to add a surrogate null. Without one, the
    ``significant`` claim is reported as untested rather than assumed.
    """
    cause = np.asarray(cause, dtype=float).ravel()
    effect = np.asarray(effect, dtype=float).ravel()
    if cause.size != effect.size:
        raise ValueError(
            f"series must be the same length, got {cause.size} and {effect.size}"
        )

    if dimension is None or delay is None:
        state = reconstruct(effect, delay_rule=delay_rule)
        dimension = state.embedding_dimension if dimension is None else dimension
        delay = state.delay if delay is None else delay

    shared = dict(
        dimension=dimension,
        delay=delay,
        n_draws=n_draws,
        theiler=theiler,
        seed=seed,
    )
    return CouplingVerdict(
        forward=cross_map_curve(cause, effect, **shared),
        backward=cross_map_curve(effect, cause, **shared),
        pearson_r=float(np.corrcoef(cause, effect)[0, 1]),
    )


def ccm_test(
    cause: np.ndarray,
    effect: np.ndarray,
    *,
    n_surrogates: int = 200,
    null: NullModel = "iaaft",
    alpha: float = 0.05,
    dimension: int | None = None,
    delay: int | None = None,
    delay_rule: DelayRule = "mutual_information",
    n_draws: int = 4,
    theiler: int | None = None,
    seed: int = 3363,
) -> CouplingVerdict:
    """CCM with a surrogate null in both directions.

    The null replaces ``cause`` with a surrogate that keeps its amplitude
    distribution and power spectrum -- and therefore its autocorrelation -- but
    destroys any relationship to ``effect``. Skill that survives is not
    explained by linear structure alone, which is the trap CCM is most often
    read past.

    The p-value is add-one smoothed, so it is never exactly zero: with 200
    surrogates the smallest reportable value is 1/201.
    """
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be at least 1")

    observed = ccm(
        cause,
        effect,
        dimension=dimension,
        delay=delay,
        delay_rule=delay_rule,
        n_draws=n_draws,
        theiler=theiler,
        seed=seed,
    )
    dimension = observed.forward.embedding_dimension
    delay = observed.forward.delay

    forward_hits = 0
    backward_hits = 0
    for i in range(n_surrogates):
        surrogate_cause = _surrogate(np.asarray(cause, float).ravel(), null, seed + i)
        surrogate_effect = _surrogate(
            np.asarray(effect, float).ravel(), null, seed + 10_000 + i
        )
        forward = cross_map_curve(
            surrogate_cause,
            effect,
            dimension=dimension,
            delay=delay,
            n_draws=n_draws,
            theiler=theiler,
            seed=seed + i,
        )
        backward = cross_map_curve(
            surrogate_effect,
            cause,
            dimension=dimension,
            delay=delay,
            n_draws=n_draws,
            theiler=theiler,
            seed=seed + i,
        )
        if forward.terminal_skill >= observed.forward.terminal_skill:
            forward_hits += 1
        if backward.terminal_skill >= observed.backward.terminal_skill:
            backward_hits += 1

    return CouplingVerdict(
        forward=observed.forward,
        backward=observed.backward,
        pearson_r=observed.pearson_r,
        forward_p=(forward_hits + 1) / (n_surrogates + 1),
        backward_p=(backward_hits + 1) / (n_surrogates + 1),
        n_surrogates=n_surrogates,
        null_model=null,
        alpha=alpha,
    )
