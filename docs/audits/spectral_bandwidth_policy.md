# Decision: RBF bandwidth policy for `node_roles_spectral`

**Outcome: the median heuristic ships as an opt-in policy. The default does not
change.** The evidence did not support making it the default.

**Reproduce:** `uv run python scripts/experiment_spectral_bandwidth.py`
**Contracts:** `tests/test_spectral_bandwidth.py`
**Prior work:** `docs/audits/role_schema_v2_design.md`

---

## Why this was open

`node_roles_spectral` used `gamma = 1 / X.shape[1]`, which made the feature
count a hyperparameter of the clustering algorithm. That was replaced by the
named constant `DEFAULT_SPECTRAL_GAMMA = 1/12`, pinned to the historical value
so the v2 schema change altered one thing at a time. A data-driven bandwidth
was left as a separate decision needing its own evidence. This is it.

## The policy

`median_heuristic_gamma(X)` sets `sigma**2` to the median squared distance
between distinct rows and returns `gamma = 1 / (2 * sigma**2)`, the convention
in Gretton et al., *A Kernel Two-Sample Test* (JMLR 2012). Cost is O(n^2),
which spectral clustering already pays to build a precomputed affinity matrix,
so it adds no asymptotic overhead.

Select it with `node_roles_spectral(..., gamma="median")`.

## Measured against the fixed default

Twenty-graph corpus, `n_roles=4`, four seeds. Aggregates over the eleven
well-posed graphs -- the other nine have so many structurally identical nodes
that any partition of them is arbitrary.

| diagnostic | fixed 1/12 | median heuristic |
|---|---|---|
| mean affinity spread (higher = more discriminating) | **0.3181** | 0.3028 |
| mean off-diagonal affinity (0 or 1 = uninformative) | 0.4931 | 0.5620 |
| more discriminating on | 8 of 11 | 3 of 11 |
| seed stability, mean pairwise ARI | **0.824** | 0.816 |
| more stable on | 2 graphs | 3 graphs |
| tied | 13 graphs | |

The bandwidths themselves differ by 0.27x to 1.50x, so this is not a case of
the two policies agreeing numerically. They genuinely differ, and neither wins.

## The heuristic's failure mode, found by the experiment

On `wheel_20` the first implementation returned a gamma of **1.3e29**. When
more than half of all node pairs coincide, the median distance is zero and the
bandwidth diverges. A wheel's rim is 19 of its 20 nodes; a star's leaves are 19
of 20.

An exact `median == 0` guard is not sufficient, for the same reason `std == 0`
was not sufficient in the standardizer: coincident rows differ by
floating-point residue, so the median is rarely exactly zero. The guard is
scale aware, comparing the median squared distance to the largest one with the
same tolerances the standardizer uses, and falls back to
`DEFAULT_SPECTRAL_GAMMA`.

**This affects 5 of the 20 corpus graphs** -- `cycle_20`, `star_19`,
`wheel_20`, `complete_12`, `barbell_8_3`. On a quarter of the corpus the median
heuristic has no opinion and defers to the constant.

## Decision

Ship it, do not default to it.

The argument for a data-driven bandwidth is real: role-feature columns can be
numerically constant for whole graph families -- `core_number` is constant on
Barabasi-Albert and Watts-Strogatz -- so points occupy fewer dimensions than
the matrix has, and a fixed bandwidth cannot know that. The median heuristic
does respond to actual spread.

But the measurements do not show it producing better-conditioned kernels or
more stable partitions on this corpus, it is slightly worse on both aggregate
diagnostics, and it degenerates precisely on the symmetric graphs where a
principled bandwidth would be most welcome. Changing a default requires
evidence that the new behaviour is better, not merely that it is more
sophisticated.

It is available for callers whose data has the structure the heuristic suits:
large graphs with distinguishable nodes, or feature matrices at a scale the
historical 1/12 was never chosen for. The constant was, after all, only ever
scikit-learn's `rbf_kernel` default for a twelve-column matrix.

## Open

- The corpus is twenty graphs of 12 to 77 nodes. A conclusion about large
  graphs, where the fixed value's implicit scale assumption is least likely to
  hold, would need a corpus of them.
- Kernel spread and seed stability are proxies. A task with ground-truth roles
  would measure quality directly; none exists in this repository.
