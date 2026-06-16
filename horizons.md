# ts2net Roadmap

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

## Roadmap Horizon 1: Foundation Hardening

**Status: IN PROGRESS (v0.3 API hardening)**

| Area               | Work                                                                                                                                                | Status |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| API consistency    | Standardize fit, transform, fit_transform, and network builder patterns across all modules.                                                         | ✅ PARTIAL — `SklearnBuildMixin` on all four builders; `NetworkBuilder` protocol |
| Type system        | Complete type hints across all public APIs. Add py.typed support. Add optional stub files where needed.                                             | ✅ PARTIAL — `py.typed`, `NetworkBuilder` protocol, mypy on core API modules |
| Config model       | Move complex builder options into dataclass configs. Keep simple function APIs for common use.                                                      | PARTIAL — dataclass configs exist for pipeline; builders use kwargs |
| Docstrings         | Standardize purpose, inputs, outputs, assumptions, examples, and references for every public function.                                              | PARTIAL |
| Test coverage      | Add coverage targets. Cover edge cases, empty inputs, constant series, missing values, short windows, unequal lengths, and high-dimensional inputs. | ✅ PARTIAL — `tests/test_api_hardening.py`, CI `--cov-fail-under=15` |
| CI quality gates   | Enforce formatting, linting, type checks, unit tests, coverage, and benchmark smoke tests.                                                          | ✅ PARTIAL — ruff + mypy + coverage in CI (scoped to API modules) |
| Error handling     | Replace silent failures and opaque errors with clear validation messages.                                                                           | ✅ PARTIAL — `ValidationError`, `NotBuiltError`, centralized `validate_series` |
| Versioned examples | Add runnable examples for every major graph construction family.                                                                                    | PARTIAL — examples exist; gallery pending |

## Roadmap Horizon 2: Core Method Expansion

**Status: IN PROGRESS (v0.4 core graph expansion)**

| Method Family        | Planned Capability                                                                                                                                                              | Status |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| Visibility graphs    | Natural visibility graphs, horizontal visibility graphs, directed visibility graphs, weighted visibility graphs, multiplex visibility graphs, and multiscale visibility graphs. | ✅ PARTIAL — HVG/NVG/directed/weighted/multiscale exist; `multiplex_visibility_graph()` added |
| Recurrence networks  | Fixed-threshold recurrence, adaptive recurrence, k-nearest recurrence, cross-recurrence, joint recurrence, and recurrence quantification features.                              | ✅ COMPLETED — `recurrence_quantification()`, adaptive/cross; joint in `multivariate` |
| Transition networks  | Symbolic transition networks, ordinal pattern networks, entropy-maximizing symbolization, SAX-based transitions, and Markov transition graphs.                                  | ✅ PARTIAL — `sax_symbolize()`, `entropy_max_symbolize()`, `sax_transition_network()` |
| Correlation networks | Pearson, Spearman, Kendall, distance correlation, partial correlation, rolling correlation, and thresholded correlation graphs.                                                 | ✅ COMPLETED — `correlation_network()`, `partial_correlation_network()`, `rolling_correlation_network()` |
| Similarity networks  | DTW, soft-DTW, Euclidean, shape-based distance, matrix profile distance, and learned embedding distance.                                                                        | ✅ COMPLETED — `similarity_network()` with `soft_dtw`, `matrix_profile`, `dtw`, etc. |
| Causal networks      | Transfer entropy, conditional transfer entropy, Granger causality, nonlinear Granger, PCMCI-style lagged discovery, PC, FCI, and time-aware constraint methods.                 | ✅ PARTIAL — see Horizon 3 |
| Event networks       | Convert events, spikes, regime changes, and detected motifs into temporal graphs.                                                                                               | ✅ COMPLETED — `event_sequence_network()`, `event_sync_network()` |
| Dynamic networks     | Build graph sequences from rolling windows. Track edge birth, edge death, node role changes, and regime shifts.                                                                 | ✅ PARTIAL — `RollingGraphSequence`, `graph_churn()`, `edge_persistence()` |
| Multiplex networks   | Represent multiple edge types at once, such as correlation, causality, recurrence, and transition edges.                                                                        | ✅ PARTIAL — `MultiplexGraph`, `multiplex_graph()` |

## Roadmap Horizon 3: Causal Intelligence

This phase makes causal analysis one of the package’s strongest differentiators.

| Capability                      | Description                                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Causal discovery adapters       | Add PC, FCI, PCMCI-like workflows, and constraint-based discovery for lagged time series.             |
| Lag selection                   | Add automatic lag search with information criteria, permutation tests, and stability checks.          |
| Confounder handling             | Extend conditional transfer entropy and partial Granger workflows to multi-variable settings.         |
| Causal edge confidence          | Report p-values, permutation scores, bootstrapped confidence intervals, and stability across windows. |
| Directional visibility analysis | Use directed visibility graphs to detect irreversibility and temporal asymmetry.                      |
| Network-based causal scoring    | Use topology to identify causal hubs, sinks, mediators, bottlenecks, and feedback loops.              |
| Intervention simulation         | Estimate what happens to downstream nodes when a source node changes or disappears.                   |
| Causal explanation reports      | Generate plain-English summaries of likely drivers, lag effects, and evidence strength.               |

## Roadmap Horizon 4: Scale and Performance

This phase makes ts2net practical for industrial workloads.

| Area                   | Work                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Out-of-core processing | Add streaming builders that process chunked arrays, Parquet files, Arrow tables, and memory-mapped data.          |
| Distributed execution  | Add Dask and Ray-compatible execution for pairwise distance, causal tests, and rolling-window graph construction. |
| GPU acceleration       | Add optional CuPy and PyTorch backends for distance matrices, window operations, and selected network builders.   |
| Sparse graph support   | Build sparse adjacency structures directly. Avoid dense matrices where possible.                                  |
| Approximate algorithms | Add approximate nearest neighbors, sketching, and pruning for high-dimensional graph construction.                |
| Incremental updates    | Update graphs as new time points arrive. Avoid full rebuilds for streaming signals.                               |
| Benchmark suite        | Track runtime, memory, graph size, and accuracy across method families and dataset sizes.                         |
| Performance contracts  | Publish expected scaling behavior for each builder.                                                               |

## Roadmap Horizon 5: Machine Learning and Graph ML

**Status: COMPLETED (v0.7 ML integration)**

| Integration          | Capability                                                                                                            | Status |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- | ------ |
| sklearn              | Expand NetworkFeatureExtractor with richer graph features, rolling features, and feature selection helpers.           | ✅ `RollingNetworkFeatureExtractor`, `NetworkFeatureSelector` |
| PyTorch Geometric    | Add converters from ts2net graph outputs to PyG Data objects.                                                         | ✅ `to_pyg_data()`, `panel_to_pyg_list()` (`[pyg]` extra) |
| DGL                  | Add DGL graph adapters.                                                                                               | ✅ `to_dgl_graph()`, `panel_to_dgl_list()` (`[dgl]` extra) |
| NetworkX             | Keep NetworkX compatibility, but avoid making it the only graph representation.                                       | ✅ existing |
| numpy and scipy      | Support sparse matrices, dense arrays, and feature matrices as first-class outputs.                                   | ✅ `features_to_dataframe()` |
| pandas and Polars    | Accept tabular time series inputs with entity, timestamp, variable, and value columns.                                | PARTIAL |
| Feature stores       | Export graph features with stable names and metadata.                                                                 | ✅ `FeatureMetadata`, `features_to_dataframe()` |
| Baseline comparisons | Compare network-derived features against tsfresh, catch22, sktime, matrix profile, and standard statistical features. | ✅ PARTIAL — `statistical_baseline_features()`, optional `tsfresh_baseline_features()`, `compare_feature_sets()` |

## Roadmap Horizon 6: Dynamic Network Analytics

This phase moves ts2net beyond static graph construction.

| Capability                | Description                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------ |
| Rolling graph builder     | Build graph sequences over sliding, expanding, or event-based windows.                                 |
| Regime detection          | Detect structural breaks based on graph metrics, edge churn, community changes, and centrality shifts. |
| Temporal communities      | Track communities across graph windows.                                                                |
| Node role evolution       | Identify nodes that become hubs, bridges, sinks, sources, or isolates over time.                       |
| Edge persistence          | Measure which relationships persist, fade, reverse, or spike.                                          |
| Network anomaly detection | Detect abnormal graphs, abnormal nodes, abnormal edges, and abnormal transitions.                      |
| Change attribution        | Explain which time series, edges, or windows caused a network-level change.                            |

## Roadmap Horizon 7: Interpretability and Reporting

This phase makes outputs understandable.

| Feature                | Description                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Edge explanations      | Explain why an edge exists, which method produced it, which lag mattered, and how strong the evidence was.          |
| Node summaries         | Describe each node’s role in the network.                                                                           |
| Graph summaries        | Summarize topology, density, hubs, communities, feedback loops, and instability.                                    |
| Method provenance      | Attach builder name, parameters, data window, lag settings, threshold rules, and random seed to every graph.        |
| Confidence annotations | Attach statistical confidence, bootstrap stability, and sensitivity to threshold choices.                           |
| Report export          | Export markdown, JSON, HTML, and notebook-friendly summaries.                                                       |
| Visualization helpers  | Add minimal plotting utilities for adjacency matrices, rolling metrics, causal networks, and dynamic graph changes. |

## Roadmap Horizon 8: Domain Recipes

This phase shows real value through complete examples.

| Domain                | Example Workflow                                                                               |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| Industrial sensors    | Detect causal drivers, sensor drift, control-loop instability, and equipment state changes.    |
| Energy production     | Convert well production histories into similarity graphs, anomaly graphs, and analog networks. |
| Finance               | Build correlation, causality, and contagion networks from asset prices or returns.             |
| Climate               | Analyze teleconnection networks and dynamic climate relationships.                             |
| Healthcare            | Build patient trajectory graphs, physiological signal networks, and event transition graphs.   |
| Web and observability | Convert logs, traces, and service metrics into dependency and anomaly networks.                |
| Neuroscience          | Build connectivity networks from EEG, fMRI, or spike train data.                               |
| Mobility              | Convert trajectories and flow time series into spatial-temporal networks.                      |

## Roadmap Horizon 9: Research-Grade Validation

This phase makes ts2net credible for academic and enterprise use.

| Area                     | Work                                                                                                            |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Reference datasets       | Add curated public datasets for classification, clustering, causality, anomaly detection, and regime detection. |
| Method validation        | Reproduce known results from visibility graph, recurrence network, and transfer entropy papers.                 |
| Benchmark papers         | Publish comparative benchmarks that show when network features outperform standard time series features.        |
| Statistical tests        | Add permutation tests, bootstrap tests, surrogate data tests, and threshold sensitivity analysis.               |
| Reproducibility          | Version benchmark datasets, parameters, random seeds, and output artifacts.                                     |
| Documentation references | Cite source papers for each method and explain assumptions.                                                     |

## Roadmap Horizon 10: Package Maturity and Community

This phase makes ts2net a sustainable open-source project.

| Area               | Work                                                                                            |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| Governance         | Add code of conduct, contribution guide, maintainer guide, release process, and decision rules. |
| Issues             | Add issue templates for bugs, features, methods, benchmarks, and documentation.                 |
| Examples gallery   | Publish a gallery with copy-paste examples and expected outputs.                                |
| Tutorials          | Add quick starts for Colab, Binder, local Python, and notebook workflows.                       |
| Release milestones | Publish versioned milestones with clear scope.                                                  |
| API stability      | Define experimental, stable, and deprecated API areas.                                          |
| Community adoption | Create PyData-style talks, comparison guides, and curated downstream examples.                  |

## Proposed Milestones

| Version | Theme                | Outcome                                                                                     |
| ------- | -------------------- | ------------------------------------------------------------------------------------------- |
| 0.3     | API hardening        | Stable public API, complete typing, consistent docs, stronger tests.                        |
| 0.4     | Core graph expansion | Broader visibility, recurrence, transition, similarity, and dynamic graph builders.         | ✅ **COMPLETED** — `ts2net.graphs` module |
| 0.5     | Causal networks      | Full causal workflow with lag search, confidence, confounders, and causal summaries.        | **In progress** — `run_causal_analysis()` workflow shipped; PCMCI/PC/FCI planned for 0.9. |
| 0.6     | Scale                | Streaming, sparse, parallel, and optional GPU-backed builders.                              |
| 0.7     | ML integration       | sklearn, PyG, DGL, feature selection, and benchmark comparisons.                            | ✅ **Completed** — rolling features, selectors, PyG/DGL adapters, baseline benchmarks. See `examples/ml_integration_example.py`. |
| 0.8     | Dynamic analytics    | Rolling graph sequences, regime detection, edge persistence, and network anomaly detection. |
| 0.9     | Research validation  | Public benchmarks, reproduced papers, statistical testing, and formal method references.    |
| 1.0     | Stable release       | Stable API, mature docs, examples gallery, governance, and production-ready workflows.      |

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

## North Star

ts2net should become the library people reach for when they believe the shape of a time series contains structure that a normal feature table cannot show.

The package should answer four questions.

What network does this time series imply?

How does that network change over time?

What relationships appear causal, unstable, anomalous, or predictive?

How can I use that structure in a real machine learning or decision workflow?
