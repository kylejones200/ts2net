# Design: a non-redundant role-feature schema, and an independent spectral bandwidth

**Status:** **applied.** All five recommendations in section 7 have been
implemented; see section 9 for what shipped and what deliberately did not.
`role_features_extended` now returns the nine-column v2 matrix, and
`node_roles_spectral` uses a named bandwidth constant that does not read the
matrix width. Sections 1-6 describe the investigation that led there and are
left as written.

**Reproduce:** `uv run python scripts/experiment_role_schema.py`
**Contracts:** `tests/test_feature_schema.py`
**Prior work:** `docs/audits/role_features_audit.md`

---

## 1. The schemas are now explicit

Column order previously existed only implicitly, split between
`_role_features_basic` and the `hstack` in `role_features_extended`. Nothing
named the columns, so nothing could say what column 9 meant. That is now
`ts2net.networks.feature_schema`, an immutable tuple of `RoleFeature`
named tuples carrying name, mathematical definition, units, and provenance.

**v1 — the shipped twelve columns, in order:**

```
degree, clustering, pagerank, eigenvector, core_number, betweenness,
closeness, triangles, wedges, ego_edges, ego_density, core_score
```

**v2 — proposed, nine columns:**

```
degree, clustering, pagerank, eigenvector, core_number, betweenness,
closeness, triangles, wedges
```

The width is a consequence, not a target: v2 is v1 minus the three aliases,
order otherwise preserved. No replacement feature was invented to restore a
count of twelve.

### The nine were verified, not assumed

`scripts/experiment_role_schema.py` derives the redundant set from the
implementations rather than reading it off the earlier audit. For every pair of
columns it asks whether they coincide on *every* graph in the corpus, and
reports the result against what `feature_schema` declares:

```
derived from 20 graphs : {ego_density: clustering, core_score: core_number, ego_edges: triangles}
declared in schema     : {ego_edges: triangles, ego_density: clustering, core_score: core_number}
agree: True
```

Corpus-wide agreement is the right criterion, and the distinction matters. On a
star, `degree`, `pagerank`, `eigenvector`, `betweenness` and `closeness` all
take only two distinct values and standardize to the same vector — but they
separate on other graphs, so they are coincidences, not aliases. A genuine
alias never separates. `tests/test_feature_schema.py` pins this: the set of
pairs identical across the whole corpus must equal `REDUNDANT_V1_COLUMNS`
exactly.

## 2. Well-posedness: which graphs can answer the question

Nine of the twenty corpus graphs have so many structurally indistinguishable
nodes that any partition of them is arbitrary. They are reported but excluded
from aggregates.

| class | graphs | unique feature rows |
|---|---|---|
| **well-posed** (11) | karate, les misérables, 3×ER, 3×WS, 3×BA | 100% except les mis at 82% |
| **symmetric** (9) | path, cycle, star, wheel, complete, bipartite, tree, grid, barbell | 5%–50% |

`cycle_20` has **1** unique row among 20 nodes; `complete_12` has 1 among 12.
Reading an ARI off those would be measuring floating-point tie-breaking.

## 3. KMeans: the schema effect alone

KMeans involves no kernel, so this isolates the geometry change. Well-posed
graphs only, `n_roles=4`, seeds 3363/7/42, 33 runs:

| measure | min | median | mean |
|---|---|---|---|
| distance correlation `r(D_v1, D_v2)` | 0.9806 | 0.9884 | — |
| relative L2 distance shift | 0.0774 | 0.0832 | — |
| **ARI v1 vs v2** | **0.424** | 0.910 | 0.844 |
| **NMI v1 vs v2** | **0.601** | 0.920 | 0.867 |

Only **9 of 33 runs** are unchanged. De-duplication is a semantic change, not
cleanup — which is the whole reason it must not ride along with a bandwidth
change.

## 4. The four-way spectral experiment

Switching v1 → v2 currently moves gamma from 1/12 to 1/9 at the same time,
because `node_roles_spectral` sets `gamma = 1 / X.shape[1]`. The full factorial
separates the two:

```
A = v1, gamma 1/12    production today
B = v2, gamma 1/12    schema changed, bandwidth held
C = v1, gamma 1/9     bandwidth changed, schema held
D = v2, gamma 1/9     naive migration: both change together
```

Median ARI per graph, well-posed graphs only:

| contrast | what it isolates | min | median | mean | unchanged |
|---|---|---|---|---|---|
| **A~B** | schema only | 0.118 | 0.863 | 0.768 | 5/11 |
| **A~C** | gamma only | 0.118 | **1.000** | 0.897 | 8/11 |
| **A~D** | naive migration | 0.118 | 0.863 | **0.732** | **4/11** |
| C~D | schema, at gamma 1/9 | 0.304 | 1.000 | 0.825 | 6/11 |
| B~D | gamma, within v2 | 0.304 | 1.000 | 0.854 | 6/11 |

Three conclusions:

1. **The schema effect is the larger one.** Holding bandwidth fixed, changing
   the schema still moves 6 of 11 graphs (A~B median 0.863).
2. **The bandwidth effect alone is smaller but real.** Holding the schema fixed,
   1/12 → 1/9 leaves 8 of 11 unchanged (A~C median 1.000) but is not a no-op.
3. **They interact.** The schema effect measured at gamma 1/12 (A~B, median
   0.863) differs from the same effect measured at gamma 1/9 (C~D, median
   1.000), with per-graph differences up to **0.882**. The two changes are not
   additive, so they cannot be reasoned about separately after the fact — and
   the naive migration, which makes both at once, is the *worst* cell of the
   four (mean 0.732, only 4/11 unchanged).

This is precisely the confound the experiment was built to expose: coupling
bandwidth to width means a schema edit silently retunes the clustering
algorithm, and the combined result is not the sum of its parts.

## 5. Where `gamma = 1 / X.shape[1]` came from

**Provenance: a scikit-learn convention, re-implemented inline. Not a published
design choice for role features.** The evidence:

- The line is present **verbatim in the initial commit** (`1093dc2`) and has
  never been modified. `git log -S` over the whole history returns only the
  three commits that moved the file.
- There is no comment, docstring, test, reference or documentation anywhere in
  the repository that mentions it. The only other `gamma` in the package is
  soft-DTW's, unrelated, and defaulted to `1.0`.
- It reproduces scikit-learn's documented `rbf_kernel(gamma=None)` default of
  `1/n_features` **exactly**: `rbf_kernel(X)` and
  `rbf_kernel(X, gamma=1/X.shape[1])` are numerically identical. The explicit
  line therefore adds nothing over simply passing `gamma=None`.

Note that "the library default" is ambiguous here, and materially so:
`rbf_kernel` defaults to `1/n_features`, while `SpectralClustering`'s own
`gamma` parameter defaults to **1.0**. The code follows the former.

### Gamma policies evaluated for v2

Median ARI against production (A = v1 @ 1/12), well-posed graphs:

| policy | gamma | min | median | mean | unchanged |
|---|---|---|---|---|---|
| historical | 1/12 | 0.118 | **0.863** | **0.768** | **5/11** |
| dimension-derived | 1/9 | 0.118 | 0.863 | 0.732 | 4/11 |
| scikit-learn `SpectralClustering` default | 1.0 | 0.128 | 0.491 | 0.539 | 1/11 |

`1/9` is **not** better merely because v2 has nine columns — it agrees with
production slightly *less* than holding 1/12. And the `SpectralClustering`
default of 1.0 is materially different from both, agreeing with production on
only 1 of 11 graphs.

Closeness to production is not by itself an argument for correctness —
production carries the accidental 2× weighting. It is an argument for changing
one thing at a time.

## 6. Compatibility

Changing the default schema **does not require a major version bump** under the
project's own conventions. Positive evidence, not merely absent tests:

- `docs/API_STABILITY.md` grants guarantees to the **Stable** tier only.
  `ts2net.api_tiers` lists 29 STABLE and 12 EXPERIMENTAL entries;
  `role_features_extended`, `node_roles_kmeans` and `node_roles_spectral` are
  in **none** of them. The one role-shaped STABLE entry, `NodeRoleSummary`, is
  a reports type and does not consume this matrix.
- The package is at **0.9.0**, before the 1.0 freeze the policy describes.
- `ts2net.networks.roles` raised `ModuleNotFoundError` on import in every commit
  from the initial commit until `7fea264`, so no released version could execute
  it and no user output can depend on its layout.
- No consumer assumes a width or a positional meaning. The only references are
  the module itself, the package `__init__` export, this project's own tests
  and analysis scripts, and these documents.
- No serialized role-feature arrays and no persisted clustering artifacts exist
  in the repository. The only `.npy`/`.pkl` files are joblib's own test data
  inside `.venv`.

## 7. Recommended transition

1. **Ship v2 as the default feature basis.** Drop `ego_edges`, `ego_density`
   and `core_score`. Keep `wedges`, labelled derived: it is determined by
   `degree` and `triangles` as `C(degree,2) - triangles`, but is quadratic in
   degree and so spans a direction neither parent does. Do not invent
   replacement features to restore a width of twelve.

2. **Do not preserve v1 as a `legacy` alias.** There is no compatibility reason:
   no released version could run this code, nothing is serialized, and it sits
   in no stability tier. `ROLE_FEATURES_V1` remains in `feature_schema` as the
   documented history of what the columns were, which is enough to interpret
   any matrix someone still holds. A code path would have to be maintained and
   tested for no identified consumer.

3. **Decide weighting explicitly.** If a factor of 2 on the triangle,
   clustering or coreness signals is actually wanted, express it as a named
   weight vector applied after standardization — not as duplicated columns. The
   default should be uniform weights, since the duplication was never a
   deliberate choice. The evidence that the current geometry is
   triangle-saturated is in the prior audit: median KMeans ARI against a
   *triplicated* matrix is 1.000, so adding more triangle weight changes
   nothing while removing the duplication changes much.

4. **Make bandwidth an explicit policy, decoupled from width.** Replace
   `gamma = 1 / X.shape[1]` with a named default that does not read
   `X.shape[1]`. On this evidence, pinning the historical `1/12` is the
   conservative choice for the transition, because it changes one thing at a
   time and agrees with production most closely. A principled alternative worth
   evaluating separately is a data-driven bandwidth such as the median
   heuristic, which depends on the distance distribution rather than the column
   count. Whatever is chosen, it must be a stated policy with a name, not a
   consequence of matrix shape.

5. **Sequence the change.** Land the schema change and the bandwidth policy as
   two commits with the bandwidth pinned first, so the spectral change is
   attributable. The factorial shows the effects interact, so a combined change
   cannot be decomposed afterwards.

## 8. The principle

**Feature width must never silently control model semantics.** A matrix may
gain, lose or reorder columns for good reasons. Kernel bandwidth, feature
weighting, normalization and clustering policy should each carry their own
named contract, so that changing one is a decision rather than a side effect.
`gamma = 1 / X.shape[1]` violated this: it made the number of columns a
hyperparameter of the clustering algorithm.


---

## 9. Transition applied

The five recommendations landed as three commits on one branch, in the order
the factorial required.

**1. Bandwidth first, as a deliberate no-op.** `gamma = 1 / X.shape[1]` became
`roles.DEFAULT_SPECTRAL_GAMMA = 1/12`. Because the schema still had twelve
columns at that point, the old rule already evaluated to 1/12 and spectral
labels were verified identical to passing `gamma=1/12` explicitly. Landing it
first means the schema change that follows is attributable to feature geometry
alone -- which matters, since section 4 showed the two effects interact by up
to 0.882 ARI per graph. A test asserts no non-comment line of `roles.py` reads
`X.shape[1]`.

**2. v2 as the default basis.** `ego_edges`, `ego_density` and `core_score` are
gone. `wedges` is kept and documented as derived. Nothing was invented to
restore a width of twelve. The private helpers that produced the removed
columns were deleted with them; `ts2net_rs.ego_edge_counts` and
`ts2net_rs.core_numbers` remain part of the Rust graph API and are still tested
against networkx, since they are general graph primitives independent of this
schema.

**3. No legacy path.** v1 is not preserved as a code path, for the reasons in
section 6: no stability tier, pre-1.0, unimportable in every released version,
nothing serialized. `ROLE_FEATURES_V1` stays in `feature_schema` as the
documented record of what the columns were.

**4. Weighting is now explicit.** `role_features_extended`, `node_roles_kmeans`
and `node_roles_spectral` accept `weights`, a mapping from feature name to
multiplier applied after standardization. **The default is uniform.** Unknown
names and negative weights are rejected.

The old duplication turns out to be *exactly* a weight vector, which closes the
investigation. Duplicating a standardized column adds `(dx)^2` twice to a
squared distance; a weight `w` on one column adds `w^2 (dx)^2`. So duplication
is precisely `w = sqrt(2)`. Verified: v1 pairwise distances equal v2 pairwise
distances under `sqrt(2)` weights on `triangles`, `clustering` and
`core_number` to **1.8e-15** on karate, les misérables, Erdős-Rényi and
Barabási-Albert. That vector is published as
`feature_schema.V1_EQUIVALENT_WEIGHTS`, documented as a record of what the old
schema did and explicitly *not* a recommended setting.

This is the concrete form of the principle: the package did encode a weighting
scheme, and it is now a named parameter with a uniform default rather than an
accident of column count.

**5. Not done, deliberately.** A data-driven bandwidth such as the median
heuristic remains a separate decision needing its own evidence.
`DEFAULT_SPECTRAL_GAMMA` is pinned at the historical `1/12` so that exactly one
thing changed in this transition.
