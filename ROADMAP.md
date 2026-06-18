# ts2net Roadmap

Single source of truth for product direction, release milestones, and implementation status.
Consolidates the former `horizons.md` and `ROADMAP.md` documents.

## Product Vision

ts2net should become the standard Python toolkit for converting time series into network graphs.

The goal is not only to build graphs from signals. The goal is to make temporal structure computable.

A user should be able to start with sensor data, market data, logs, production data, patient data, climate data, or any multivariate time series, then produce network representations that support exploration, causality, anomaly detection, clustering, forecasting, graph machine learning, and decision support.

ts2net should sit at the intersection of time series analysis, network science, causal inference, and machine learning.

## Strategic Direction

| Theme                   | Ambition                                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Core graph construction | Support the major families of time-series-to-network methods with clean APIs and strong defaults.                  |
| Causality               | Make causal network inference a first-class capability, not an add-on.                                             |
| Scale                   | Handle large, streaming, and high-dimensional time series without forcing all data into memory.                    |
| ML integration          | Make graph features and graph objects easy to use in sklearn, PyTorch Geometric, DGL, and downstream ML workflows. |
| Interpretability        | Explain why edges exist, how they change, and what network structure means.                                        |
| Benchmarks              | Prove value against baseline time series methods on public datasets.                                               |
| Documentation           | Make ts2net easy for scientists, engineers, and data scientists to adopt.                                          |

## Release Milestones

| Version | Theme                | Outcome                                                                                     | Status |
| ------- | -------------------- | ------------------------------------------------------------------------------------------- | ------ |
| 0.3     | API hardening        | Stable public API, complete typing, consistent docs, stronger tests.                        | **In progress** — `py.typed`, `NetworkBuilder` protocol, validation layer, CI gates |
| 0.4     | Core graph expansion | Broader visibility, recurrence, transition, similarity, and dynamic graph builders.         | ✅ **Completed** — `ts2net.graphs` module |
| 0.5     | Causal networks      | Full causal workflow with lag search, confidence, confounders, and causal summaries.        | ✅ **Completed** — `run_causal_analysis()` |
| 0.6     | Scale                | Streaming, sparse, parallel, and optional GPU-backed builders.                              | **In progress** — all four builders have streaming stats fast paths; chunked DTW; GPU/Dask open |
| 0.7     | ML integration       | sklearn, PyG, DGL, feature selection, and benchmark comparisons.                            | ✅ **Completed** — see `examples/ml_integration_example.py` |
| 0.8     | Dynamic analytics    | Rolling graph sequences, regime detection, edge persistence, and network anomaly detection. | ✅ **Completed** — see `examples/dynamic_analytics_example.py` |
| 0.9     | Research validation  | Public benchmarks, reproduced papers, statistical testing, and formal method references.    | **In progress** — PC/FCI discovery, directional visibility asymmetry; reference datasets pending |
| 1.0     | Stable release       | Stable API, mature docs, examples gallery, governance, and production-ready workflows.      | **Planned** |

## Guiding Principles

| Principle                 | Meaning                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------- |
| Make graphs useful        | Every graph builder should support downstream analysis, not only graph creation.      |
| Keep simple things simple | Common workflows should need only one or two function calls.                          |
| Preserve provenance       | Every graph should know how it was built.                                             |
| Prefer composability      | Builders, metrics, features, and exporters should work together.                      |
| Scale by design           | Large datasets should not require a rewrite.                                          |
| Explain the result        | Users should understand what edges, nodes, and metrics mean.                          |
| Respect uncertainty       | Causal and statistical claims should expose confidence, sensitivity, and assumptions. |
| Stay interoperable        | ts2net should fit the PyData, sklearn, NetworkX, and graph ML ecosystems.             |

## Development Principles

- **Backward compatibility**: New parameters have sensible defaults.
- **Consistent patterns**: Dataclass configs, factory dispatch, type safety.
- **Documentation**: Each feature gets docstrings, examples, and config docs.
- **Benchmarking**: Validate performance and memory on real datasets.
- **Testing**: Comprehensive test coverage for all new features.

## North Star

ts2net should become the library people reach for when they believe the shape of a time series contains structure that a normal feature table cannot show.

The package should answer four questions:

1. What network does this time series imply?
2. How does that network change over time?
3. What relationships appear causal, unstable, anomalous, or predictive?
4. How can I use that structure in a real machine learning or decision workflow?

---

## Horizon 1: Foundation Hardening

**Milestone: v0.3** · **Status: IN PROGRESS**

| Area               | Work                                                                                                                                                | Status |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| API consistency    | Standardize fit, transform, fit_transform, and network builder patterns across all modules.                                                         | ✅ PARTIAL — `SklearnBuildMixin` on all four builders; `NetworkBuilder` protocol |
| Type system        | Complete type hints across all public APIs. Add py.typed support. Add optional stub files where needed.                                             | ✅ PARTIAL — `py.typed`, `NetworkBuilder` protocol, mypy on core API modules |
| Config model       | Move complex builder options into dataclass configs. Keep simple function APIs for common use.                                                      | PARTIAL — pipeline dataclass configs; all four builders expose `backend=` |
| Docstrings         | Standardize purpose, inputs, outputs, assumptions, examples, and references for every public function.                                              | PARTIAL |
| Test coverage      | Add coverage targets. Cover edge cases, empty inputs, constant series, missing values, short windows, unequal lengths, and high-dimensional inputs. | ✅ PARTIAL — `tests/test_api_hardening.py`, CI `--cov-fail-under=15` |
| CI quality gates   | Enforce formatting, linting, type checks, unit tests, coverage, and benchmark smoke tests.                                                          | ✅ PARTIAL — ruff + mypy + coverage + `TS2NET_CI_SMOKE` benchmark in `.github/workflows/ci.yml` |
| Error handling     | Replace silent failures and opaque errors with clear validation messages.                                                                           | ✅ PARTIAL — `ValidationError`, `NotBuiltError`, centralized `validate_series()` |
| Versioned examples | Add runnable examples for every major graph construction family.                                                                                    | ✅ PARTIAL — see Examples table below |
| Fuzz testing       | Random time series to catch numerical errors.                                                                                                       | ✅ PARTIAL — Hypothesis property tests in `tests/unit/test_properties.py` |

## Horizon 2: Core Method Expansion

**Milestone: v0.4** · **Status: COMPLETED**

| Method Family        | Capability                                                                                                                                                                      | Status |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| Visibility graphs    | Natural visibility graphs, horizontal visibility graphs, directed visibility graphs, weighted visibility graphs, multiplex visibility graphs, and multiscale visibility graphs. | ✅ PARTIAL — HVG/NVG/directed/weighted/multiscale exist; `multiplex_visibility_graph()`; Rust/Numba degree-only stats via `visibility_degree_stats()` |
| Recurrence networks  | Fixed-threshold recurrence, adaptive recurrence, k-nearest recurrence, cross-recurrence, joint recurrence, and recurrence quantification features.                              | ✅ COMPLETED — `recurrence_quantification()`, `adaptive_recurrence_network()`, `cross_recurrence_network()`; Rust degree-only stats via `recurrence_degree_stats()` |
| Transition networks  | Symbolic transition networks, ordinal pattern networks, entropy-maximizing symbolization, SAX-based transitions, and Markov transition graphs.                                  | ✅ PARTIAL — `sax_symbolize()`, `entropy_max_symbolize()`, `sax_transition_network()`; stats-only path via `transition_degree_stats()` |
| Correlation networks | Pearson, Spearman, Kendall, distance correlation, partial correlation, rolling correlation, and thresholded correlation graphs.                                                 | ✅ COMPLETED — `correlation_network()`, `partial_correlation_network()`, `rolling_correlation_network()` |
| Similarity networks  | DTW, soft-DTW, Euclidean, shape-based distance, matrix profile distance, and learned embedding distance.                                                                        | ✅ COMPLETED — `similarity_network()` with `soft_dtw`, `matrix_profile`, `dtw`, etc. |
| Causal networks      | Transfer entropy, conditional transfer entropy, Granger causality, nonlinear Granger, PCMCI-style lagged discovery, PC, FCI, and time-aware constraint methods.                 | ✅ PARTIAL — TE/Granger workflow + PC/FCI (`ts2net.causal`); PCMCI-style via `time_lagged_causality_network()` |
| Event networks       | Convert events, spikes, regime changes, and detected motifs into temporal graphs.                                                                                               | ✅ COMPLETED — `event_sequence_network()`, `event_sync_network()` |
| Dynamic networks     | Build graph sequences from rolling windows. Track edge birth, edge death, node role changes, and regime shifts.                                                                 | ✅ COMPLETED — see Horizon 6 |
| Multiplex networks   | Represent multiple edge types at once, such as correlation, causality, recurrence, and transition edges.                                                                        | ✅ COMPLETED — `MultiplexGraph`, `multiplex_graph()`, `multiplex_visibility_graph()` |

## Horizon 3: Causal Intelligence

**Milestone: v0.5** · **Status: COMPLETED** (advanced discovery algorithms deferred to v0.9)

| Capability                      | Description                                                                                           | Status |
| ------------------------------- | ----------------------------------------------------------------------------------------------------- | ------ |
| Transfer entropy                | Pairwise and network-level information-theoretic causality.                                           | ✅ `transfer_entropy()`, `transfer_entropy_network()` — `tests/test_transfer_entropy.py` |
| Conditional transfer entropy    | Multi-variable causal inference with confounders.                                                     | ✅ `conditional_transfer_entropy()` |
| Granger causality               | Linear (statsmodels) and nonlinear (MLP permutation) pairwise tests and networks.                     | ✅ `granger_causality()`, `granger_causality_network()` — `tests/test_granger_causality.py` |
| Lag selection                   | Automatic lag search with information criteria and permutation tests.                                 | ✅ `search_granger_lag()`, `search_te_lag()` |
| Confounder handling             | Partial Granger and conditional TE in multi-variable settings.                                        | ✅ `partial_granger_causality()`, `conditional_te_network()` |
| Causal edge confidence          | P-values, permutation scores, and bootstrapped confidence intervals.                                  | ✅ `te_permutation_test()`, `te_bootstrap_ci()` |
| Causal network metrics          | Path-based and node-level causal summaries.                                                           | ✅ `causal_strength()`, `directionality_index()`, `causal_network_metrics()` |
| Time-lagged analysis            | Transfer entropy or Granger causality across multiple lags.                                           | ✅ `time_lagged_causality_network()` |
| End-to-end workflow             | Lag search, confidence, confounders, network construction, and reports.                               | ✅ `run_causal_analysis()`, `CausalAnalysisResult.summary()` — `examples/causal_workflow_example.py` |
| Causal discovery adapters       | PC, FCI, PCMCI-like workflows, and constraint-based discovery for lagged time series.                 | ✅ PARTIAL — `pc_algorithm()`, `fci_algorithm()`, `pc_timeseries_network()`, `fci_timeseries_network()` — `tests/test_causal_discovery.py` |
| Directional visibility analysis | Use directed visibility graphs to detect irreversibility and temporal asymmetry.                      | ✅ `directed_visibility_analysis()`, `visibility_irreversibility()` — `tests/test_visibility_causal.py` |
| Intervention simulation         | Estimate downstream effects when a source node changes or disappears.                                 | PLANNED |
| Network-based causal inference  | Leverage topology to infer causal relationships (e.g. directed visibility irreversibility).           | ✅ PARTIAL — visibility asymmetry panel and temporal asymmetry index |

**Use cases:** multi-sensor causal driver identification, information-flow analysis in complex systems, network-based causal discovery for time series.

## Horizon 4: Scale and Performance

**Milestone: v0.6** · **Status: IN PROGRESS**

| Area                   | Work                                                                                                              | Status |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- | ------ |
| Compute backends       | Unified `rust` → `numba` → `python` selection for graph builders and distance kernels.                            | ✅ PARTIAL — `resolve_compute_backend()`, `TS2NET_BACKEND`, `backend=` on all builder configs; visibility + recurrence Rust stats — `tests/test_backend.py` |
| Out-of-core processing | Streaming builders for chunked arrays, Parquet files, Arrow tables, and memory-mapped data.                       | ✅ PARTIAL — chunk iterators + `stream_chunk_stats()` with fast stats for all four core builders (`visibility_degree_stats`, `recurrence_degree_stats`, `transition_degree_stats`) — `examples/scale_streaming_example.py` |
| Distributed execution  | Dask and Ray-compatible execution for pairwise distance, causal tests, and rolling-window graph construction.     | ✅ PARTIAL — experimental `ts2net.distributed` module; chunking in distance jobs |
| Parallelization        | Controls for embarrassingly parallel workloads.                                                                   | ✅ PARTIAL — `n_jobs` on `build_windows()`, causal network builders; `cdist_dtw_chunked()` and `ts_dist(panel_chunk_threshold=…)` for large DTW panels |
| GPU acceleration       | Optional CuPy and PyTorch backends for distance matrices, window operations, and selected network builders.       | PLANNED — `[gpu]` extra reserved; install CuPy manually |
| Sparse graph support   | Build sparse adjacency structures directly. Avoid dense matrices where possible.                                  | ✅ PARTIAL — `Graph.adjacency_matrix(format='sparse')`, `to_sparse_csr()`, `edges_to_csr()` |
| Approximate algorithms | Approximate nearest neighbors, sketching, and pruning for high-dimensional graph construction.                    | ✅ PARTIAL — `approximate_knn_network()`, `similarity_network(approximate=True)` via `[approx]` / pynndescent |
| Incremental updates    | Update graphs as new time points arrive without full rebuilds.                                                     | ✅ PARTIAL — `IncrementalHVG.append()` for streaming HVG extension |
| Benchmark suite        | Track runtime, memory, graph size, and accuracy across method families and dataset sizes.                         | ✅ PARTIAL — `benchmarks/run_benchmarks.py`; CI smoke via `TS2NET_CI_SMOKE=1` (~11s) |
| Performance contracts  | Publish expected scaling behavior for each builder.                                                                 | ✅ PARTIAL — `get_performance_contract()`, `list_performance_contracts()` incl. `cdist_dtw` |
| Rust fast paths        | Degree-only visibility stats and rectangular DTW blocks without full edge materialisation.                          | ✅ PARTIAL — visibility + recurrence Rust stats; `transition_degree_stats()` Python fast path; `cdist_dtw_chunked()` |

### v0.6 remaining

- GPU backends (CuPy/PyTorch) for distance matrices and selected builders
- Dask/Ray execution beyond experimental `ts2net.distributed`

## Horizon 5: Machine Learning and Graph ML

**Milestone: v0.7** · **Status: COMPLETED**

| Integration          | Capability                                                                                                            | Status |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- | ------ |
| sklearn              | Network features, rolling features, and feature selection helpers.                                                    | ✅ `NetworkFeatureExtractor`, `RollingNetworkFeatureExtractor`, `NetworkFeatureSelector` — `examples/network_features_sklearn.py` |
| PyTorch Geometric    | Converters from ts2net graph outputs to PyG Data objects.                                                             | ✅ `to_pyg_data()`, `panel_to_pyg_list()` (`[pyg]` extra) |
| DGL                  | DGL graph adapters.                                                                                                   | ✅ `to_dgl_graph()`, `panel_to_dgl_list()` (`[dgl]` extra) |
| NetworkX             | Keep NetworkX compatibility without making it the only graph representation.                                         | ✅ existing |
| numpy and scipy      | Sparse matrices, dense arrays, and feature matrices as first-class outputs.                                           | ✅ `features_to_dataframe()` |
| pandas and Polars    | Tabular time series inputs with entity, timestamp, variable, and value columns.                                       | PARTIAL — `from_pandas()`, `from_polars()`, pipeline I/O |
| Feature stores       | Export graph features with stable names and metadata.                                                                 | ✅ `FeatureMetadata`, `features_to_dataframe()` |
| Baseline comparisons | Compare network features against tsfresh, catch22, sktime, matrix profile, and statistical features.                  | ✅ PARTIAL — `statistical_baseline_features()`, optional `tsfresh_baseline_features()`, `compare_feature_sets()` |

## Horizon 6: Dynamic Network Analytics

**Milestone: v0.8** · **Status: COMPLETED**

| Capability                | Description                                                                                            | Status |
| ------------------------- | ------------------------------------------------------------------------------------------------------ | ------ |
| Rolling graph builder     | Graph sequences over sliding, expanding, or event-based windows.                                       | ✅ `RollingGraphSequence`, `build_windows()` |
| Regime detection          | Structural breaks from graph metrics, edge churn, community changes, and centrality shifts.              | ✅ `detect_regime_changes()`, `run_dynamic_analysis()` |
| Temporal communities      | Track communities across graph windows.                                                                  | ✅ `track_communities()` |
| Node role evolution       | Identify nodes that become hubs, bridges, sinks, sources, or isolates over time.                       | ✅ `node_role_evolution()` |
| Edge persistence          | Measure which relationships persist, fade, reverse, or spike.                                          | ✅ `edge_persistence()`, `graph_churn()`, `edge_birth_death()` |
| Network anomaly detection | Detect abnormal graphs, nodes, edges, and transitions.                                                 | ✅ `window_anomaly_scores()`, `edge_transition_anomalies()` |
| Change attribution        | Explain which metrics, edges, or windows caused a network-level change.                                | ✅ PARTIAL — metric shift attribution at regime breaks |

See `examples/dynamic_analytics_example.py`.

## Horizon 7: Interpretability and Reporting

**Status: PLANNED**

| Feature                | Description                                                                                                         | Status |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- | ------ |
| Edge explanations      | Explain why an edge exists, which method produced it, which lag mattered, and how strong the evidence was.          | PARTIAL — causal reports cover lag and p-value |
| Node summaries         | Describe each node's role in the network.                                                                           | PARTIAL — `node_roles()`, causal emitter/receiver metrics |
| Graph summaries        | Summarize topology, density, hubs, communities, feedback loops, and instability.                                    | PARTIAL — `graph_summary()`, causal network metrics |
| Method provenance      | Attach builder name, parameters, data window, lag settings, threshold rules, and random seed to every graph.      | PARTIAL — `FeatureMetadata`, builder stats |
| Confidence annotations | Statistical confidence, bootstrap stability, and sensitivity to threshold choices.                                  | PARTIAL — causal permutation/bootstrap; TE confidence |
| Report export          | Export markdown, JSON, HTML, and notebook-friendly summaries.                                                       | PARTIAL — `CausalAnalysisResult.to_markdown()`, `DynamicAnalysisResult.summary()` |
| Visualization helpers  | Plotting for adjacency matrices, rolling metrics, causal networks, and dynamic graph changes.                       | PARTIAL — `ts2net.viz` gallery plots |

## Horizon 8: Domain Recipes

**Status: PLANNED**

| Domain                | Example Workflow                                                                               | Status |
| --------------------- | ---------------------------------------------------------------------------------------------- | ------ |
| Industrial sensors    | Detect causal drivers, sensor drift, control-loop instability, and equipment state changes.    | ✅ PARTIAL — `run_causal_analysis()`, `run_dynamic_analysis()`, `directed_visibility_analysis()` |
| Energy production     | Convert well production histories into similarity graphs, anomaly graphs, and analog networks. | PARTIAL — Spain meter case study, FRED example |
| Finance               | Build correlation, causality, and contagion networks from asset prices or returns.             | PARTIAL — `examples/example_fred_data.py` |
| Climate               | Analyze teleconnection networks and dynamic climate relationships.                             | PLANNED |
| Healthcare            | Build patient trajectory graphs, physiological signal networks, and event transition graphs.   | PLANNED |
| Web and observability | Convert logs, traces, and service metrics into dependency and anomaly networks.                | PLANNED |
| Neuroscience          | Build connectivity networks from EEG, fMRI, or spike train data.                               | PLANNED |
| Mobility              | Convert trajectories and flow time series into spatial-temporal networks.                      | PLANNED |

## Horizon 9: Research-Grade Validation

**Milestone: v0.9** · **Status: IN PROGRESS**

| Area                     | Work                                                                                                            | Status |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- | ------ |
| Reference datasets       | Curated public datasets for classification, clustering, causality, anomaly detection, and regime detection.   | PLANNED |
| Method validation        | Reproduce known results from visibility graph, recurrence network, and transfer entropy papers.                 | PLANNED |
| Benchmark papers         | Comparative benchmarks showing when network features outperform standard time series features.                    | ✅ PARTIAL — `compare_feature_sets()`; `benchmarks/run_benchmarks.py` |
| Statistical tests        | Permutation tests, bootstrap tests, surrogate data tests, and threshold sensitivity analysis.                   | ✅ PARTIAL — causal TE permutation/bootstrap; surrogate tests in `ts2net.stats.null_models` |
| Reproducibility          | Version benchmark datasets, parameters, random seeds, and output artifacts.                                     | PLANNED |
| Documentation references | Cite source papers for each method and explain assumptions.                                                     | PARTIAL |
| Advanced causal discovery| PC algorithm, FCI, and PCMCI-style methods adapted for time series networks.                                    | ✅ PARTIAL — `pc_algorithm()`, `fci_algorithm()`, `pc_timeseries_network()`, `fci_timeseries_network()` — `examples/causal_discovery_example.py` |
| Directional asymmetry    | Irreversibility and time-arrow statistics from directed visibility graphs.                                      | ✅ PARTIAL — `directed_visibility_analysis()`, `visibility_irreversibility()` — `tests/test_visibility_causal.py` |

## Horizon 10: Package Maturity and Community

**Milestone: v1.0** · **Status: PLANNED**

| Area               | Work                                                                                            | Status |
| ------------------ | ----------------------------------------------------------------------------------------------- | ------ |
| Governance         | Code of conduct, contribution guide, maintainer guide, release process, and decision rules.     | PLANNED |
| Issue templates    | Templates for bugs, features, methods, benchmarks, and documentation.                           | PLANNED |
| Examples gallery   | Gallery with copy-paste examples and expected outputs.                                            | PARTIAL — `examples/viz_gallery.py`, Binder notebook |
| Tutorials          | Quick starts for Colab, Binder, local Python, and notebook workflows.                           | PARTIAL — `binder/`, `examples/spain_meter_case_study.ipynb` |
| API stability      | Define experimental, stable, and deprecated API areas.                                          | PLANNED |
| Community adoption | PyData-style talks, comparison guides, and curated downstream examples.                           | PLANNED |
| Comparison guides  | Guides to alternative time-series and network libraries.                                        | PLANNED |

---

## Runnable Examples

| Example | Milestone / area |
| ------- | ---------------- |
| `examples/quick_start.py` | Core builders |
| `examples/unified_graphs_example.py` | Core graph families |
| `examples/viz_gallery.py` | Visualization |
| `examples/network_features_sklearn.py` | v0.7 sklearn |
| `examples/ml_integration_example.py` | v0.7 PyG/DGL/baselines |
| `examples/dynamic_analytics_example.py` | v0.8 dynamic analytics |
| `examples/causal_workflow_example.py` | v0.5 causal workflow |
| `examples/causal_discovery_example.py` | v0.9 PC/FCI discovery |
| `examples/scale_streaming_example.py` | v0.6 streaming scale |
| `examples/polars_spain_windows.py` | Large-scale meter data |
| `examples/spain_meter_case_study.ipynb` | Domain recipe (energy) |
