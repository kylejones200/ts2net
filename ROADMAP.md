

## Remaining Work

### Medium Priority Features

### Performance & Scalability
- **Out-of-core streaming builders**: Process data in chunks rather than requiring entire series in memory
- **GPU acceleration**: Via CuPy/PyTorch for high-volume workloads
- **Parallelization controls**: ✅ **PARTIAL** - `n_jobs` on causal network builders and multivariate `ts_dist()`; chunking in distributed distance jobs
- **Benchmark suite**: ✅ **PARTIAL** - `benchmarks/run_benchmarks.py` tracks performance curves (CI integration pending)

### API & Usability Improvements
- **Type hints and stubs**: ✅ **PARTIAL** — `py.typed` marker, `NetworkBuilder` protocol, mypy on core API modules (`exceptions`, `_validation`, `_builder_api`, `protocols`)
- **Consistent docstrings**: Standardized sections (purpose, inputs, outputs, examples) (partial - standardized for fit/transform methods, but not complete across all modules)
- **API consistency**: ✅ **PARTIAL** — All four builders (`HVG`, `NVG`, `RecurrenceNetwork`, `TransitionNetwork`) share `build`/`fit`/`transform`/`fit_transform` via `SklearnBuildMixin`
- **Error handling**: ✅ **PARTIAL** — `ValidationError`, `NotBuiltError`, centralized `validate_series()` with clear messages
- **CI quality gates**: ✅ **PARTIAL** — ruff, mypy, coverage threshold in `.github/workflows/ci.yml`


### Causal Inference & Network-Based Causality
- **Transfer entropy networks**: ✅ **COMPLETED** - `transfer_entropy()` computes information-theoretic causality between time series pairs, `transfer_entropy_network()` constructs directed networks based on transfer entropy values. Supports discrete binning and k-NN methods. Exported in `ts2net.causal` module. Comprehensive tests added (`tests/test_transfer_entropy.py`).
- **Conditional transfer entropy**: ✅ **COMPLETED** - `conditional_transfer_entropy()` accounts for confounding variables in multi-variable causal inference. Exported in `ts2net.causal` module.
- **Granger causality networks**: ✅ **COMPLETED** - `granger_causality()` (linear via statsmodels, nonlinear via MLP permutation test) and `granger_causality_network()` build directed networks from pairwise tests. Exported in `ts2net.causal`.
- **Causal network metrics**: ✅ **COMPLETED** - `causal_strength()`, `directionality_index()`, and `causal_network_metrics()` for path-based and node-level causal summaries.
- **Time-lagged network analysis**: ✅ **COMPLETED** - `time_lagged_causality_network()` evaluates transfer entropy or Granger causality across multiple lags.
- **Causal workflow (0.5)**: ✅ **COMPLETED** - `run_causal_analysis()` with lag search, permutation confidence, confounder adjustment, and `CausalAnalysisResult.summary()`. See `examples/causal_workflow_example.py`.
- **Causal discovery algorithms**: PC algorithm, FCI, and constraint-based methods adapted for time series networks
- **Network-based causal inference**: Leverage network topology to infer causal relationships (e.g., using directed visibility graphs for irreversibility analysis)

**Use Cases:**
- Identifying causal relationships in multi-sensor systems (e.g., which sensor influences which)
- Understanding information flow in complex systems
- Detecting causal drivers in time series data
- Network-based causal discovery for time series

---

## Lower Priority / Future Enhancements

### Statistical & ML Integrations
- **Feature pipeline**: ✅ **COMPLETED** - `NetworkFeatureExtractor` in `ts2net.sklearn` for sklearn pipelines. See `examples/network_features_sklearn.py`.
- **ML integration (0.7)**: ✅ **COMPLETED** - `RollingNetworkFeatureExtractor`, `NetworkFeatureSelector`, `features_to_dataframe`, `compare_feature_sets`, PyG/DGL adapters (`to_pyg_data`, `to_dgl_graph`). See `examples/ml_integration_example.py`.
- **Benchmark comparisons**: ✅ **COMPLETED** - `statistical_baseline_features()`, optional `tsfresh_baseline_features()`, `compare_feature_sets()`.
- **Feature selection routines**: ✅ **COMPLETED** - `NetworkFeatureSelector` with mutual information / F-test scoring.

### Core Method Enhancements (v0.4)
- **Correlation & similarity networks**: ✅ **COMPLETED** — `correlation_network()`, `partial_correlation_network()`, `similarity_network()` (euclidean, spearman, dtw, soft_dtw, matrix_profile)
- **Dynamic graph sequences**: ✅ **COMPLETED** — `RollingGraphSequence`, `graph_churn()`, `edge_persistence()`
- **Multiplex networks**: ✅ **COMPLETED** — `MultiplexGraph`, `multiplex_visibility_graph()`
- **Adaptive recurrence networks**: ✅ **COMPLETED** — `adaptive_recurrence_network()` with target-density epsilon
- **Cross-recurrence**: ✅ **COMPLETED** — `cross_recurrence_network()` (joint recurrence in `multivariate`)
- **Recurrence quantification (RQA)**: ✅ **COMPLETED** — `recurrence_quantification()` with RR, DET, L, ENTR, LAM, TT
- **Alternative symbolization**: ✅ **COMPLETED** — `sax_symbolize()`, `entropy_max_symbolize()`, `sax_transition_network()`
- **Event networks**: ✅ **COMPLETED** — `event_sequence_network()`, `event_sync_network()`

### Testing, CI, and Quality
- **Expand test coverage**: Cover all methods and edge cases
- **Code coverage targets**: Enforce via CI
- **Fuzz tests**: Random time series to catch numerical errors

### Documentation & Tutorials
- **Gallery of examples**: Show how ts2net solves real tasks (anomaly detection, clustering, comparison)
- **Quick start notebooks**: On common platforms (Binder, Colab)
- **Comparison guides**: To alternative libraries

### Ecosystem & Interoperability
- **Scikit-learn wrappers**: ✅ **COMPLETED** - `NetworkFeatureExtractor` in `ts2net.sklearn` for sklearn pipelines. See `examples/network_features_sklearn.py`.
- **Graph ML adapters**: ✅ **COMPLETED** - `to_pyg_data()`, `to_dgl_graph()` in `ts2net.ml` (optional `[pyg]` / `[dgl]` extras).
- **PyData materials**: Talks and curated lists of downstream users

### Community & Governance
- **Code of conduct**: Clear contribution process
- **Issue templates**: For feature requests, enhancements, and bugs
- **Roadmap milestones**: Publish timelines in the repo

---

## Development Principles

- **Backward compatibility**: New parameters have sensible defaults
- **Consistent patterns**: Dataclass configs, factory dispatch, type safety
- **Documentation**: Each feature gets docstrings, examples, config docs
- **Benchmarking**: Validate performance and memory on real datasets
- **Testing**: Comprehensive test coverage for all new features
