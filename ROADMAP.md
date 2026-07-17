# ts2net Roadmap

Single source of truth for product direction, release milestones, and implementation status.

## Product Vision

ts2net converts time series into network graphs — and makes the result **usable**.

The goal is not only to build graphs from signals. The goal is to make temporal structure **computable, explainable, and actionable**.

A user should start with sensor data, market data, logs, production data, patient data, or any multivariate time series, then produce network representations that support exploration, causality, anomaly detection, forecasting, graph ML, and **decision support**.

ts2net sits at the intersection of time series analysis, network science, causal inference, and machine learning.

## Phase Shift: Library Construction → Product Proof

The method library is mostly done. v0.4 through v0.9 are complete. v0.3 API polish is complete for public modules under CI.

**The next move is not “add more graph methods.”**

The next move is **make ts2net useful enough that people can trust it, copy it, cite it, and apply it.**

| Question | What “done” looks like |
| -------- | ---------------------- |
| Can someone use this in 15 minutes? | Quick start + recipes + method chooser |
| Can they explain the result to a boss? | Graph reports, edge explanations, summaries |
| Can they trust the evidence? | Benchmarks, confidence, assumptions surfaced |
| Can they cite the method? | References, reproducible validation artifacts |
| Can they apply it to a real domain? | Domain recipes, not toy examples |
| Can it produce a decision? | Decision packages with evidence and next action |

---

## Active Roadmap (Post–v0.9)

### v1.0 — Freeze the Core

Stop method churn. Ship a stable release people can depend on.

| Work | Outcome |
| ---- | ------- |
| API tiers | Define **stable**, **experimental**, and **deprecated** public APIs |
| Builder patterns | Lock main builder patterns (`fit` / `transform` / configs / provenance) |
| Release notes | Per-version changelog with breaking changes called out |
| Migration guide | v0.x → v1.0 upgrade path for common workflows |
| Governance | Contribution guide, release process, maintainer decision rules |
| Finish v0.3 polish | Complete typing, docstrings, and CI gates on remaining public modules |

**Status:** ✅ SHIPPED — API tiers, CHANGELOG, MIGRATION, governance, builder contract tests; typing CI on reports, api_tiers, causal/dynamic summaries, datasets/ucr

### Reports — Make Graphs Explain Themselves

This is the biggest product gap. Every graph should answer:

- What made this edge exist?
- What changed?
- What is anomalous?
- What can I do with this?

| Component | Purpose | Status |
| --------- | ------- | ------ |
| `GraphReport` | Human-readable summary of topology, hubs, communities, instability | ✅ `build_graph_report()` |
| `EdgeExplanation` | Method, parameters, lag, threshold, strength, confidence per edge | ✅ `explain_edge_from_graph()`, `explain_edges_from_causal()` |
| `NodeRoleSummary` | Role, centrality context, drift, anomaly flags per node | ✅ via `node_roles()` in graph reports |
| `DynamicChangeReport` | Regime breaks, edge birth/death, attribution across windows | ✅ `build_dynamic_change_report()` |
| `DecisionPackage` | Evidence, confidence, assumptions, what changed, suggested next action | ✅ `build_decision_package()` |

Build on existing: `CausalAnalysisResult.summary()`, `DynamicAnalysisResult.summary()`, `graph_summary()`, `FeatureMetadata`.

**Horizon:** 7 (Interpretability and Reporting) · **Status:** ✅ SHIPPED (v1.0-rc)

### Recipes — Turn Methods into Use Cases

Horizon 8 becomes the main backlog. Not broad examples — **real recipes**.

| Domain | Recipe focus | Status |
| ------ | ------------ | ------ |
| Industrial sensors | Drift, causal driver, failure precursor | ✅ `examples/recipes/industrial_sensors.py` |
| Energy production | Analog wells, interference, abnormal decline, forecast risk | ✅ synthetic + `energy_spain_real.py` |
| Finance | Contagion, regime change, unstable correlations | ✅ synthetic + `finance_fred_real.py` |
| Observability | Service dependency, incident precursors, noisy subsystem | ✅ `examples/recipes/observability_services.py` |
| Healthcare | Patient trajectory states, risk shifts | ✅ `examples/recipes/healthcare_trajectory.py` |

Each recipe: data → graph → report → decision hook, runnable in &lt;15 minutes.

**Horizon:** 8 (Domain Recipes) · **Status:** ✅ SHIPPED (synthetic + real-data variants)

### Benchmarks — Prove When Graphs Beat Tables

The package needs **proof artifacts**. Each benchmark should compare ts2net graph features against ordinary time-series features (statistical, tsfresh, matrix profile, etc.).

The result should say **where graphs help, where they do not, and why**.

| Work | Status |
| ---- | ------ |
| UCR classification harness | ✅ Bundled `.npz`, baselines, `run_ucr_benchmark.py` |
| Literature validation fixtures | ✅ HVG, TE, RN, RQA, PCMCI — `run_validation.py` |
| Feature-set comparison API | ✅ `compare_feature_sets()`, statistical baselines |
| Published “when graphs win” reports | ✅ multi-dataset `when_graphs_win.py` → `benchmarks/results/` |
| Threshold sensitivity + null models | ✅ `threshold_sensitivity_sweep()`, causal permutation/bootstrap |

**Horizon:** 9 (validation infra + proof narratives) · **Status:** ✅ SHIPPED

### Decision Workflows — Connect to Decision Systems

ts2net should be part of **Decision Systems Institute**, not a standalone package.

```
time series → network → decision package
```

A decision package includes:

- Evidence (edges, metrics, causal paths)
- Confidence (p-values, bootstrap, stability)
- Assumptions (method, window, threshold, lag)
- What changed (regime, edge churn, role shift)
- Next action (investigate, monitor, intervene, forecast)

**Status:** ✅ SHIPPED — `build_decision_package()` ties graph, causal, and dynamic evidence

### Adoption — Make It Easy to Steal the Pattern

| Work | Status |
| ---- | ------ |
| Examples gallery | ✅ `examples/GALLERY.md` |
| Five core notebooks | ✅ `examples/recipes/*.ipynb` (generated via `scripts/generate_recipe_notebooks.py`) + Spain case study |
| “Choose your method” guide | ✅ `docs/method_chooser.md` |
| Comparison pages | ✅ `docs/comparisons.md` |
| PyData-style talks / downstream examples | PLANNED |

**Horizon:** 10 (Package Maturity and Community) · **Status:** ✅ SHIPPED (docs + notebooks wired; PyData talks remain optional)

---

## Release Milestones

| Version | Theme | Outcome | Status |
| ------- | ----- | ------- | ------ |
| 0.3 | API hardening | Stable public API, typing, docs, tests | ✅ **Completed** |
| 0.4 | Core graph expansion | Visibility, recurrence, transition, similarity builders | ✅ **Completed** |
| 0.5 | Causal networks | Lag search, confidence, confounders, causal summaries | ✅ **Completed** |
| 0.6 | Scale | Streaming, sparse, parallel, GPU, Dask/Ray | ✅ **Completed** |
| 0.7 | ML integration | sklearn, PyG, DGL, feature selection, baselines | ✅ **Completed** |
| 0.8 | Dynamic analytics | Rolling graphs, regimes, edge persistence, anomalies | ✅ **Completed** |
| 0.9 | Research validation | UCR, literature fixtures, baselines, Sphinx refs | ✅ **Completed** |
| **1.0** | **Stable release** | **Freeze core API, reports, recipes, proof benchmarks, adoption** | **Ready** — tag/push pending maintainer sign-off |

### Remaining before v1.0 tag

| Item | Status |
| ---- | ------ |
| Recipe notebooks (5 domains) | ✅ `examples/recipes/*.ipynb` |
| v0.3 typing / docstring polish on public modules | ✅ CI gates on reports, summaries, ucr |
| Real-data recipe variants (Spain, FRED) | ✅ `energy_spain_real.py`, `finance_fred_real.py` |
| Extended benchmark narratives (multi-dataset) | ✅ `when_graphs_win.py` rollup across UCR panels |
| DecisionPackage walkthrough | ✅ `examples/decision_package_walkthrough.py` |
| PyPI release with `1.0.0` version bump | ⏳ pending tag/push (not done yet) |

---

## Strategic Direction

| Theme | Ambition |
| ----- | -------- |
| Core graph construction | ✅ Major method families shipped — maintain, do not expand blindly |
| Causality | First-class workflow — extend with reports and recipes |
| Scale | Large, streaming, distributed — done for core paths |
| ML integration | sklearn / PyG / DGL — done; focus on recipe outcomes |
| **Interpretability** | **Reports shipped — extend edge cases and causal/dynamic tie-ins** |
| **Benchmarks** | **Multi-dataset proof narratives shipped — maintain harness** |
| **Documentation** | **Gallery, notebooks, real-data recipes shipped** |
| **Decision support** | **Decision packages shipped — connect to external decision systems** |

## Guiding Principles

| Principle | Meaning |
| --------- | ------- |
| Make graphs useful | Every builder supports downstream analysis, not only graph creation |
| Keep simple things simple | Common workflows need one or two function calls |
| Preserve provenance | Every graph knows how it was built |
| Explain the result | Users understand edges, nodes, metrics, and confidence |
| Respect uncertainty | Causal and statistical claims expose assumptions and sensitivity |
| Stay interoperable | PyData, sklearn, NetworkX, graph ML ecosystems |
| **Stop method churn** | **v1.0 freezes the core; new work is proof and product** |

## North Star

ts2net is the library people reach for when the **shape** of a time series contains structure a feature table cannot show — and they need to **trust, explain, and act on** that structure.

Four questions every release should strengthen:

1. What network does this time series imply?
2. How does that network change over time?
3. What relationships appear causal, unstable, anomalous, or predictive?
4. **What should someone do next — and why?**

---

## Completed Work (Reference)

<details>
<summary>Horizon 1 — Foundation Hardening (v0.3 partial)</summary>

| Area | Status |
| ---- | ------ |
| API consistency, typing, validation, CI | ✅ |
| Docstrings, coverage targets | ✅ (public modules under CI) |
| Fuzz testing (Hypothesis) | ✅ PARTIAL |

</details>

<details>
<summary>Horizon 2 — Core Method Expansion (v0.4 complete)</summary>

Visibility, recurrence, transition, correlation, similarity, causal, event, dynamic, multiplex graphs; Rust SINDy backend. See git history and `ts2net.graphs` for full inventory.

</details>

<details>
<summary>Horizon 3 — Causal Intelligence (v0.5 complete)</summary>

Transfer entropy, Granger, lag search, confounders, `run_causal_analysis()`, PC/FCI adapters. Intervention simulation remains PLANNED.

</details>

<details>
<summary>Horizon 4 — Scale (v0.6 complete)</summary>

Rust/Numba/Python backends, streaming, Dask/Ray, GPU correlation, sparse graphs, approximate KNN, incremental HVG, benchmarks.

</details>

<details>
<summary>Horizon 5 — ML Integration (v0.7 complete)</summary>

sklearn extractors, PyG/DGL converters, baseline comparisons, feature metadata.

</details>

<details>
<summary>Horizon 6 — Dynamic Analytics (v0.8 complete)</summary>

`RollingGraphSequence`, regime detection, edge persistence, anomaly scores. See `examples/dynamic_analytics_example.py`.

</details>

<details>
<summary>Horizon 9 — Research Validation (v0.9 complete)</summary>

Bundled UCR, literature fixtures, `run_validation.py`, `run_ucr_benchmark.py`, Sphinx method refs, threshold sensitivity.

</details>

---

## Runnable Examples

| Example | Area |
| ------- | ---- |
| `examples/quick_start.py` | Core builders |
| `examples/unified_graphs_example.py` | Core graph families |
| `examples/causal_workflow_example.py` | Causal workflow |
| `examples/dynamic_analytics_example.py` | Dynamic analytics |
| `examples/ml_integration_example.py` | PyG/DGL/baselines |
| `examples/validation_example.py` | Research validation |
| `examples/sindy_example.py` | Dynamics discovery |
| `examples/scale_streaming_example.py` | Streaming scale |
| `examples/spain_meter_case_study.ipynb` | Energy domain (recipe seed) |
| `benchmarks/run_validation.py` | Literature validation CLI |
| `benchmarks/run_ucr_benchmark.py` | UCR benchmark CLI |
| `examples/recipes/*_real.py` | Real-data domain recipes (Spain, FRED) |
| `examples/decision_package_walkthrough.py` | DecisionPackage demo |
| `benchmarks/when_graphs_win.py` | Multi-dataset “when graphs win” report |

**Next examples to add:** intervention simulation demo (Horizon 3 backlog); PyData talk notebooks (optional).
