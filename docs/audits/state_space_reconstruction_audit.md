# Audit: state-space reconstruction, and a design for `reconstruct()`

**Status:** **complete.** All six steps in section 9 have been implemented.
The two defects in section 4 are fixed (4a), the delay gap in section 8 is
closed, the neighbour ceiling in section 5 is raised, `reconstruct()` exists,
the recurrence entry points are reconciled, and CCM is built on top. Sections
1-8 are left as the investigation that led there; section 10 records what
shipped and what the work found along the way.

**Reproduce:** `uv run pytest tests/test_state_space_reconstruction.py`

---

## Why

`ts2net` treats delay embedding as one utility among many. The proposal is to
make the reconstructed state the object everything hangs off:

```
                     ┌─ graph
                     │
time series ─→ state ├─ manifold
                     │
                     ├─ recurrence
                     │
                     └─ causal coupling (CCM)
```

Before designing that, this establishes what already exists, what is correct,
and what is missing.

## 1. What exists

| Capability | Where | State |
|---|---|---|
| Delay embedding | 5 implementations (below) | present, all agree |
| Embedding dimension: FNN | `ts2net_rs::embedding::false_nearest_neighbors` | **validated correct** |
| Embedding dimension: Cao E1/E2 | `ts2net_rs::embedding::cao_e1_e2` | **validated correct** |
| FNN / Cao NumPy fallbacks | `ts2net/stats/stats.py` | **fixed** (section 4a) |
| Neighbour queries | `ts2net_rs.knn`, `.radius` (kiddo k-d tree) | present, capped at 6 dimensions |
| Recurrence adjacency | `ts2net_rs.rn_adj_epsilon` + 5 Python modules | present, contracts differ |
| **Delay (tau) selection** | — | **absent** |
| **Manifold geometry** | — | absent |
| **CCM** | — | absent |

## 2. Delay embedding: five implementations, one convention

`ts2net/core/__init__.py:embed`, `ts2net/stats/threshold_sensitivity.py:_delay_embed`,
two closures inside `ts2net/stats/stats.py`, and Rust `embedding::fnn::delay_embed`.

All build the same thing:

```
E[:, i] = x[i*tau : i*tau + L]      L = n - (m-1)*tau
```

so a row is a state vector and column `i` is the series lagged by `i*tau`.
Verified identical across `m` in {2,3,4,5,6} and `tau` in {1,3,4,5,10}.

This is duplication *without* divergence -- unlike the role features, where
duplication had drifted into aliasing. A `reconstruct()` type should own this
convention once, but adopting it breaks nothing.

One nuance: the Rust `delay_embed` takes the length `L` as an argument, and
`false_nearest_neighbors` passes `n - m*tau` rather than `n - (m-1)*tau`, so
the m- and (m+1)-dimensional embeddings have equal row counts. That is
deliberate and correct -- and it is precisely what the NumPy fallbacks get
wrong.

## 3. The Rust dimension selection is scientifically correct

Measured against systems whose embedding dimension is known analytically.

FNN fraction by dimension:

| signal | m=1 | m=2 | m=3 | m=4 | m=5 | expected |
|---|---|---|---|---|---|---|
| sine | 0.045 | **0.000** | 0.000 | 0.000 | 0.000 | unfolds at 2 |
| Lorenz *x* | 0.995 | 0.063 | **0.000** | 0.000 | 0.000 | unfolds at 3 |
| white noise | 0.995 | 0.779 | 0.326 | 0.179 | 0.161 | never unfolds |

Cao's E2, which should sit near 1 for stochastic data and away from it for
deterministic data:

| signal | mean E2 | reading |
|---|---|---|
| white noise | ~1.14 | stochastic |
| Lorenz *x* | ~0.04 | deterministic |
| sine | 0.00 | deterministic |

This is the bedrock a manifold layer needs, and it holds.

## 4. Both NumPy fallbacks are non-functional

`ts2net.stats.stats.false_nearest_neighbors` and `cao_e1_e2` fall back to NumPy
when the compiled extension is absent. Neither works.

**Root cause, shared.** Neighbour indices are computed on the m-dimensional
embedding, which has `n-(m-1)*tau` rows, then used to index the
(m+1)-dimensional embedding, which has `tau` fewer rows.

**FNN** additionally collapses the criterion. Line:

```python
num = np.linalg.norm(Xmp[np.arange(Xmp.shape[0]), -1] - Xmp[j, -1], ord=2)
```

`np.linalg.norm` without `axis=` returns a **scalar**, but FNN needs one ratio
per point. The next line then compares that single number against every
neighbour distance, so `fnn` collapses to 0.0 or 1.0 for the whole series,
independent of the data. Two independent defects in one function.

**Cao** fails on the same mismatch, surfacing as `IndexError` or as
`ValueError: operands could not be broadcast together with shapes (397,2)
(400,2)` depending on where the nearest neighbour falls.

Neither is reachable while the extension is installed -- the same shape as
`roles.py`'s phantom Rust fast path: a fallback that has never run.

## 4a. Both fallbacks fixed

Both now embed at a common length of `n - m*tau`, as the Rust implementation
does, so the m- and (m+1)-dimensional reconstructions span the same points and
a neighbour index is valid in both. `false_nearest_neighbors` applies its
criterion per point rather than collapsing it to a scalar.

Two shared private helpers were added: `_embed_fixed_length`, which takes the
row count rather than deriving it, and `_nearest_neighbour`, a blocked
brute-force search that computes distances from coordinate differences directly
rather than through a Gram matrix, and resolves ties to the lowest index. Both
choices are what make the result match the Rust path exactly rather than
approximately.

| | agreement with Rust |
|---|---|
| `false_nearest_neighbors` | **bit-identical** over 3 signals x 3 `(m_max, tau)` settings |
| `cao_e1_e2` | E1 to 3.5e-11, E2 to 2.9e-15 -- summation order only |

The fallback is also verified correct in its own right, not merely equal to
Rust: FNN falls below 0.01 at m=2 for a sine and m=3 for Lorenz, and stays
above 0.05 through m=5 for white noise. A regression test asserts the fractions
take intermediate values, which the scalar-criterion defect made impossible.

`m_max < 2` now raises in the NumPy path as it already did in Rust, and an
embedding longer than the series degrades to 1.0 (FNN) and NaN (Cao) instead of
failing.

## 5. The neighbour-query dimension ceiling

`ts2net_rs.knn` and `.radius` are monomorphised for dimensions 1 to 6 and raise
above that; `kiddo` takes the dimension as a const generic, so each width needs
explicit instantiation. This is an implementation limit, not a mathematical one.

It does **not** bind on dimension selection or recurrence adjacency, both of
which use brute-force O(L^2) neighbours: `false_nearest_neighbors` runs at
`m_max=10`, and `rn_adj_epsilon` works at dimension 8.

It binds exactly where a manifold layer would want fast neighbour queries.
Lorenz needs 3, but real signals frequently reconstruct above 6, and CCM
cross-mapping is k-NN in the shadow manifold.

## 6. Recurrence has incompatible entry points

Construction is spread across `core/recurrence.py`, `core/recurrence_backend.py`,
`core/recurrence_optimized.py`, `graphs/recurrence.py`,
`multivariate/joint_cross.py` and Rust `rn_adj_epsilon`.

They do not share an input contract. `graphs.recurrence.recurrence_matrix`
rejects an embedded point cloud:

```
ValidationError: recurrence_matrix: Input must be a 1-D array, got shape (250, 3)
```

while `rn_adj_epsilon` requires exactly that `(n, d)` matrix. So one takes a raw
series and embeds internally (or not at all), the other takes reconstructed
states.

**Whether these paths agree numerically is not characterized here.** Comparing
them needs its own slice; asserting agreement without measuring it would be the
error this project has spent several batches removing.

## 7. Design: `reconstruct()`

The smallest type that makes the state the primary object, built only from what
exists plus the one missing piece:

```python
state = ts2net.reconstruct(x, dim=None, delay=None)

state.delay                  # tau, chosen or supplied
state.embedding_dimension    # m, chosen or supplied
state.points                 # (L, m) delay coordinates
state.diagnostics            # fnn curve, Cao E1/E2, how m and tau were chosen
state.recurrence(eps=...)    # adjacency over state.points
state.network(...)           # graph view of that adjacency
```

Design commitments, each following from a finding above:

1. **One embedding convention**, owned here, matching the five that exist, so
   adoption is behaviour-preserving (section 2).
2. **`dim=None` selects via Cao E1/E2 with FNN as corroboration**, using the
   validated Rust path (section 3). The selection rule must be named and
   documented, not implicit -- the same discipline as `DEFAULT_SPECTRAL_GAMMA`.
3. **`delay=None` requires new code** (section 8). Until it exists,
   `reconstruct()` cannot honestly claim to choose a reconstruction.
4. **`recurrence()` takes reconstructed states**, matching `rn_adj_epsilon`,
   and the 1-D entry points are reconciled to it rather than the reverse
   (section 6).
5. **Report degeneracy rather than narrate it.** `diagnostics` must be able to
   say *no finite embedding dimension was found* -- white noise gives exactly
   that, and Cao E2 near 1 is the signal. This project has now found three
   cases where a statistic was degenerate rather than zero; the fourth should
   not be discovered by a user.

## 8. Blocking gap: no delay selection

There is no mutual information, no autocorrelation, no first-zero or
first-minimum rule anywhere in the package. Every function takes `tau` as an
argument and defaults it to 1.

`reconstruct(x)` with no arguments is not implementable today. This is the one
piece that must be built rather than re-homed, and it should be its own slice
with its own validation: mutual-information first-minimum and ACF first-zero
both have known answers on a sine and on Lorenz.

## 9. Recommended order

1. **Fix the two NumPy fallbacks** (section 4). Unambiguous defects, testable
   against the validated Rust results, no design content.
2. **Add delay selection** (section 8), validated on known systems.
3. **Introduce `reconstruct()`** (section 7), re-homing existing code behind
   one contract.
4. **Reconcile the recurrence entry points** (section 6), after characterizing
   whether they agree.
5. **Raise or remove the neighbour-query ceiling** (section 5), which CCM needs.
6. **CCM**, on the manifold the above provides, with surrogate-backed
   significance distinguishing *correlated* from *coupled* from *convergent*
   from *significant against a null*.


---

## 10. What shipped

Six steps, four commits, each validated against systems with a known answer
rather than against recorded output.

**Delay selection** (section 8's blocking gap). Mutual information first
minimum and autocorrelation first zero, with the rule carried in the result.
Mutual information puts Lorenz at 16 samples, the literature value.

Three things the measurements changed. The naive "first strict local minimum"
picks up estimator noise -- on a sine at period 40 it returns lag 7 for a dip
of 0.001 on a curve spanning 0.95 -- so a minimum must now rise again
afterwards by 5% of the curve's range, measured forward rather than against
immediate neighbours. An all-zero autocorrelation "crosses zero" at lag 1, so
a constant series was reporting a delay; the crossing test now requires the
curve to start positive. And the two rules **do not agree**: on Lorenz mutual
information gives 16 while the autocorrelation first zero is an order of
magnitude further out. An earlier version of that test asserted they agree.

**Neighbour ceiling** (section 5). 6 to 16, dispatched through a macro so
`knn` and `radius` cannot drift apart, with a test walking every advertised
width.

**`reconstruct()`** (section 7). Lorenz reconstructs in 3 dimensions, a sine
in 2, and white noise reports *no finite embedding dimension found (consistent
with a stochastic process)* rather than returning the fallback as a result --
the commitment section 7 point 5 asked for.

**Recurrence reconciliation** (section 6). The paths agree at dimension 1, as
suspected but not previously measured. Measuring it found a defect:
`recurrence_quantification` accepted `m` and `tau`, passed them to the builder
and the epsilon search, then rebuilt the matrix from the raw series. RQA at
m=8, tau=5 was bit-identical to m=1, tau=1, with no warning. Fixed; the
embedded path now agrees exactly with `StateSpace.recurrence`, and recurrence
rate falls from 0.322 at dimension 1 to 0.055 at dimension 8 as it should.

**CCM**. Validated on the coupled logistic system of Sugihara et al. (2012):
one-way coupling gives forward skill 0.981 against backward 0.590 while the
Pearson correlation is 0.147. Five claims are reported separately --
correlated, predictive, convergent, asymmetric, significant -- and
`significant` is None when untested rather than False. A Theiler window
defaulting to the delay suppresses the autocorrelation trap; `ccm_test`
compares against IAAFT surrogates, which is only trustworthy because the IAAFT
in this package was repaired earlier in this work.

**Totals.** pytest 1199 passed; Rust 76.

### Still open

- Scenario-style anchoring of the delay: `reconstruct` picks one delay for the
  whole series, so a signal whose timescale changes is reconstructed at a
  compromise.
- CCM cross-maps at lag zero. Distinguishing time-delayed causal interactions
  needs a lag sweep over the cross map, which is a separate slice.
- The `recurrence_quantification` fix is a behaviour change for anyone who
  passed `m` or `tau`. The previous behaviour was wrong rather than different,
  but published numbers computed with those arguments will not reproduce.
