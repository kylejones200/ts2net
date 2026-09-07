# Audit: the `role_features_extended` feature matrix

**Status:** characterization complete. The **correctness** defect in section 4
has since been fixed (see section 4a). The **schema** findings in sections 2, 5
and 8 are unchanged and remain proposals: no column has been removed, no
ordering changed, and no clustering default touched.

**Scope:** `ts2net.networks.roles.role_features_extended`, the twelve-column
node-feature matrix consumed by `node_roles_kmeans` and `node_roles_spectral`.

**Reproduce:** `uv run python scripts/audit_role_features.py`
**Contracts pinned by tests:** `tests/test_role_features_characterization.py`

---

## Question

Do the twelve columns represent twelve distinct signals?

**No.** They span at most nine directions, and fewer on several common graph
classes. Three columns are exact aliases of three others. Because the matrix is
standardized column-wise and then fed to Euclidean-distance methods, the
aliasing is not cosmetic: it doubles the weight of three signals in every
distance computation. The package encodes a feature-weighting scheme that
nothing documents and that appears to be unintentional.

A second, independent defect surfaced during the audit: standardization is
applied twice, which converts numerically constant columns into full-amplitude
noise and makes the function non-deterministic on symmetric graphs.

---

## 1. Feature definitions

Columns in the order returned. The first seven come from
`ts2net.networks.communities._role_features_basic`, which **already
standardizes them**; `role_features_extended` standardizes the concatenation
again.

| # | Name | Definition | Source | Scale | Depends on |
|---|------|-----------|--------|-------|-----------|
| 0 | `deg` | degree `k(u)` | `G.degree` | raw count, grows with density | — |
| 1 | `cc` | local clustering `2T(u)/(k(k−1))`, 0 if `k≤1` | `nx.clustering` | normalized to [0,1] | degree, triangles |
| 2 | `pr` | PageRank, damping 0.85 | `nx.pagerank` | normalized, sums to 1 → **size dependent** | whole graph |
| 3 | `ev` | eigenvector centrality | `nx.eigenvector_centrality_numpy` | unit L2 norm → **size dependent** | whole graph |
| 4 | `core` | k-core number | `nx.core_number` | raw integer | degeneracy ordering |
| 5 | `btw` | betweenness, normalized | `nx.betweenness_centrality` | [0,1] | all shortest paths |
| 6 | `clo` | closeness | `nx.closeness_centrality` | [0,1] | all shortest paths |
| 7 | `tri` | triangles at `u` | Rust `triangles_per_node` | raw count | degree, clustering |
| 8 | `wedges` | `C(k,2) − T(u)` | `_motif_features` | raw count | **deterministic in `deg` and `tri`** |
| 9 | `ego_edges` | edges among `N(u)` | Rust `ego_edge_counts` | raw count | **≡ `tri`** |
| 10 | `ego_density` | `2m_u/(k(k−1))`, 0 if `k≤1` | `_egonet_density` | [0,1] | **≡ `cc`** |
| 11 | `core_score` | `core / max(core)` | `_core_periphery_scores` | [0,1] | **rescaling of `core`** |

## 2. The three equivalences, proved

**`ego_edges ≡ tri`.** An edge between two neighbours of `u` closes exactly one
triangle through `u`, and every triangle through `u` supplies exactly one such
edge. The correspondence is a bijection, so the counts are equal at every node.
Verified as exact integer equality on 15 hand-built graphs and 6 randomized
ones; degree-0 and degree-1 nodes give 0 under both.

**`ego_density ≡ cc`.** `_egonet_density(u) = 2·m_u/(k(k−1))` for `k ≥ 2` and 0
otherwise, where `m_u` is the edge count among `N(u)`. Substituting `m_u = T(u)`
from the identity above gives the textbook local clustering coefficient, which
is what `nx.clustering` computes, including the same `k ≤ 1 → 0` convention.
Verified to 1e-12 on all 20 graphs against both the closed form and
`nx.clustering`.

**`core_score ≡ core` after standardization.** `core_score = core/max(core)` is
multiplication by a positive constant, and `z(a·x) = z(x)` for `a > 0`. Raw,
the columns differ by a scale factor; downstream they are identical. Verified
to 1e-9 on all 20 graphs.

`wedges` is *not* an alias but is fully determined by two columns already
present: `wedges = C(deg,2) − tri` exactly, with the `np.maximum(…, 0)` clamp
never binding because `T(u) ≤ C(k(u),2)` always. It survives as an independent
*linear* direction only because it is quadratic in degree.

**Measured rank:** 9 of 12 on graphs with varied structure (karate, les
misérables, Erdős–Rényi); 8 on Barabási–Albert and Watts–Strogatz; as low as 1
on vertex-transitive graphs.

## 3. Degeneracy under common graph classes

| Graph class | Degenerate columns | Effective rank |
|---|---|---|
| complete, cycle | all 12 constant | 1 |
| star, complete bipartite | `cc`,`core`,`tri`,`ego_edges`,`ego_density`,`core_score` | 2 |
| path, grid, balanced tree | same six | 5–6 |
| Barabási–Albert (fixed `m`), Watts–Strogatz (fixed `k`) | `core`, `core_score` | 8 |
| Erdős–Rényi, karate, les misérables | none | 9 |

On triangle-free graphs (path, star, tree, grid, bipartite) **six of twelve
columns are identically constant** and contribute nothing to any distance.

## 4. Second defect: double standardization amplifies float noise

`_role_features_basic` returns standardized columns; `role_features_extended`
standardizes again. For a genuinely varying column this is a no-op
(`z(z(x)) = z(x)`). For a numerically constant one it is not:

```
raw eigenvector centrality on a 20-cycle : std 1.9e-15   (uniform by symmetry)
after the first z-score (+1e-12 guard)   : std 1.9e-03   (guard dominates)
after the second z-score                 : std 1.0       (guard irrelevant)
```

The guard `+ 1e-12` protects the first pass but not the second, so pure
floating-point residue is rescaled to a unit-variance feature.

Consequence: `nx.eigenvector_centrality_numpy` runs ARPACK from a random start
vector, so the residue differs between calls. **`role_features_extended` is
non-deterministic on vertex-transitive graphs**: five identical calls returned
matrices differing by up to **3.26** (`cycle_20`) and **3.48** (`complete_12`).
On karate the spread is 1.1e-14.

This is independent of the aliasing and, unlike it, produces results that are
not reproducible at all.

## 4a. Correctness fix applied

Section 4's defect is fixed. The schema findings are untouched.

**Ownership.** Standardization occurred in two places:
`communities._role_features_basic` (over its seven columns) and
`roles.role_features_extended` (over the concatenation). Both are removed in
favour of a single owner, `ts2net.networks._standardize.standardize`. Feature
builders now return raw columns; whoever assembles the final matrix
standardizes it once. `communities.node_roles`, the other consumer of
`_role_features_basic`, standardizes its own matrix through the same function.

**Degeneracy criterion.** `std == 0` is not used, because a mathematically
constant graph feature computed in floating point has a spread near 1e-15. A
column is degenerate when

```
std <= max(DEGENERACY_ATOL, DEGENERACY_RTOL * scale)
scale = max(|mean|, max|x|)
```

with both tolerances at `1e-12`. The relative term catches a constant at any
magnitude; the absolute floor catches residue straddling zero, where there is
no magnitude for a relative test to work against. A degenerate column
standardizes to **exactly 0.0**. Non-degenerate columns are divided directly:
the tolerance has already established the divisor is safe, so no `+ epsilon`
remains. An epsilon should protect arithmetic, not manufacture structure.

**Results after the fix.**

| Measure | Before | After |
|---|---|---|
| Repeated-call spread, `cycle_20` | 3.26 | **0.0** |
| Repeated-call spread, `complete_12` | 3.48 | **0.0** |
| Repeated-call spread, `karate` | 1.1e-14 | 7.8e-15 |
| Graphs reproducible to 1e-9 | 18/20 | **20/20** |
| Columns amplified from constant to unit variance | 2 | **0** |
| Reconstruction check (independent implementation) | n/a | 2.2e-11 |

Degenerate columns now standardize to exact zero, matching the section 3 table:
12 of 12 zero on cycle and complete; 6 of 12 on path, star and complete
bipartite; 2 of 12 (`core`, `core_score`) on Barabasi-Albert and
Watts-Strogatz.

**Impact on results.** On the 18 graphs where the old pipeline was
deterministic, node distances are **unchanged**: Pearson r between old and new
distance vectors is 1.0000 and the relative L2 shift is 0.0000 on every one.
The matrices differ only at the 1e-11 level, which is the removal of the
`+ 1e-12` epsilon. Only `ev` was ever amplified, because it alone is computed
by an iterative solver; integer-valued columns such as `tri` and `core` are
*exactly* constant, so `std` was exactly 0 and the old code already produced
exactly 0.

Cluster assignments are identical (ARI 1.000) on every graph whose nodes are
structurally distinguishable, including karate, les misérables and all nine
random graphs. They differ only where the clustering problem is ill-posed to
begin with: `wheel_20` has 19 of its 20 nodes sharing an identical feature row,
`star_19` 19 of 20, `complete_bipartite(5,7)` 12 of 12. Partitioning identical
points into four clusters is decided by floating-point tie-breaking, and a
1e-11 perturbation flips it. That is not a behavioural regression. On
`cycle_20` and `complete_12` the old output was not reproducible at all, so
"changed" is not well defined there; the new output is a stable all-zero matrix.

The alias identities in section 2 all still hold, and the rank table in section
3 is unchanged.

## 5. Measured effect of the duplicate weighting

Comparison matrices were built in analysis code only: `full12` as shipped,
`canonical9` keeping one copy of each signal, and `triplicated13` adding a
third copy of `tri`. Standardization is applied after column selection, exactly
as production does.

**Geometry** (12 graphs where the duplicated columns vary; the other 8 are
unaffected because those columns are constant there):

| | min | median | max |
|---|---|---|---|
| Pearson `r` between `D12` and `D9` distances | 0.9806 | 0.9889 | 0.9972 |
| Relative L2 difference (norm-matched) | 0.0585 | 0.0830 | 0.1104 |

Node distances shift by **6–11%** before any clustering algorithm runs.

**Clustering** (`n_roles=4`, seeds 3363/7/42, 36 runs). Agreement between the
shipped 12-column result and the de-duplicated 9-column result:

| | min | median | mean |
|---|---|---|---|
| KMeans ARI | **0.424** | 0.929 | 0.857 |
| Spectral ARI (γ held at 1/12) | **0.118** | 0.851 | 0.739 |

Worst cases: spectral on les misérables ARI 0.118; spectral on barbell
0.296–0.493; KMeans on Erdős–Rényi 0.424; KMeans on karate 0.783 across all
three seeds.

**The weighting interpretation.** Median KMeans ARI between the shipped matrix
and the *triplicated* one is **1.000** — adding yet more triangle weight
usually changes nothing, while removing the duplication changes a great deal.
The shipped geometry already sits in a triangle-dominated regime. In squared
Euclidean terms the implicit weights are:

| Signal | Columns | Implicit weight |
|---|---|---|
| triangle count | `tri`, `ego_edges` | **2** |
| clustering coefficient | `cc`, `ego_density` | **2** |
| coreness | `core`, `core_score` | **2** |
| degree, PageRank, eigenvector, betweenness, closeness, wedges | one each | 1 |

Four of the twelve columns carry triangle-derived information (`tri`,
`ego_edges` raw; `cc`, `ego_density` normalized), and `wedges` is a fifth
triangle-dependent column.

Note `node_roles_spectral` defaults `gamma = 1/X.shape[1]`, so the column count
*itself* is a hyperparameter. Removing three columns changes γ from 1/12 to 1/9
even if nothing else changes. The table above holds γ fixed to isolate the
duplication; the script reports both.

## 6. Is the twelve-column layout a promised API?

**No**, and this rests on positive evidence rather than absence of tests.

- `ts2net/networks/roles.py` raised `ModuleNotFoundError` on import in **every
  commit from the initial commit until `7fea264`** (2026-09-07), because it
  imported `ts2net.networks.utils`, which has never existed on any branch. No
  released version could execute this code, so no user output can depend on it.
- Not referenced in `docs/` at all; `grep -rln roles docs/` is empty.
- Absent from `docs/API_STABILITY.md` and `ts2net/api_tiers.py`.
- No serialization, model-persistence or column-name mapping anywhere consumes
  it. The only callers are the two clustering functions in the same module.
- The sole external references are `ts2net/networks/__init__.py` (exported in
  `7fea264`) and `tests/test_roles_parity.py`, which compares the Rust and
  networkx paths and asserts nothing about width or column order.

## 7. Classification

| Feature | Classification |
|---|---|
| `deg` | independent signal |
| `cc` | independent signal (normalized triangle density; degenerate on triangle-free graphs) |
| `pr` | independent signal; graph-size dependent (normalized to sum 1) |
| `ev` | **constant or degenerate under common graph classes** — uniform on every vertex-transitive graph, where it becomes amplified float noise |
| `core` | independent signal; **constant on BA and WS** |
| `btw` | independent signal |
| `clo` | independent signal |
| `tri` | independent signal (raw triangle count) |
| `wedges` | **deterministic transform** of `deg` and `tri`: `C(deg,2) − tri`; linearly independent only because it is quadratic in degree |
| `ego_edges` | **deterministic transform — exact duplicate of `tri`** |
| `ego_density` | **deterministic transform — exact duplicate of `cc`** |
| `core_score` | **normalization of `core`**; exact duplicate after standardization |

## 8. Recommendations

Not applied. Each requires its own change with its own characterization.

| Feature | Recommendation | Rationale |
|---|---|---|
| `ego_edges` | **Remove** in the next compatibility break | Exactly `tri`. Carries no information; only doubles that signal's weight. |
| `ego_density` | **Remove** in the next compatibility break | Exactly `cc`. Same reasoning. |
| `core_score` | **Remove** in the next compatibility break | A rescaling the pipeline erases. If a bounded coreness is wanted, it should replace `core`, not accompany it. |
| `wedges` | **Retain** | Not a duplicate. Quadratic in degree, so it adds a real direction. Document it as derived. |
| `ev` | **Replace with a genuinely independent structural feature** | Degenerate on every regular graph and, through double standardization, actively harmful there. If retained, it needs a determinism fix and a degeneracy guard. |
| Triangle weighting | **Decide explicitly** | If a factor of 2 on triangle-derived signals is wanted, it should be a documented, named weight, not three duplicated columns. |

Two changes should precede any schema change, because they are correctness
issues rather than design ones:

1. **Standardize once.** Either `_role_features_basic` returns raw columns and
   the caller standardizes, or `role_features_extended` does not re-standardize.
   This removes the noise amplification and the non-determinism.
2. **Guard degenerate columns.** A column whose raw standard deviation is
   negligible relative to its mean should standardize to exactly zero, not to
   rescaled float residue.

Because `role_features_extended` has never been reachable in a released
version, all of the above can be done without a deprecation cycle.

## 9. Wider principle

Every feature vector in `ts2net` should carry a documented mathematical basis,
a units and normalization contract, and an explicit reason for inclusion. This
matrix mixes raw counts (`deg`, `tri`, `wedges`), bounded ratios (`cc`, `btw`,
`clo`, `ego_density`, `core_score`), size-dependent normalized quantities
(`pr`, `ev`) and integers (`core`) with no stated contract, then standardizes
and hands the result to a Euclidean metric. Aliasing was easy to introduce
because nothing recorded what each column was for.
