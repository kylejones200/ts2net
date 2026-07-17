# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Reports (`ts2net.reports`)**: `GraphReport`, `EdgeExplanation`, `NodeRoleSummary`, `DynamicChangeReport`, `DecisionPackage`, `build_graph_report()`, `build_decision_package()`.
- **Domain recipes**: `examples/recipes/` — industrial, energy, finance, observability, healthcare (synthetic + real-data variants).
- **Real-data recipes**: `energy_spain_real.py` (Spain meter panel + ItalyPowerDemand UCR), `finance_fred_real.py` (bundled FRED-style macro panel).
- **DecisionPackage walkthrough**: `examples/decision_package_walkthrough.py`.
- **Proof artifacts**: multi-dataset `benchmarks/when_graphs_win.py` narrative benchmark report.
- **Bundled datasets**: `fred_macro_panel` in `ts2net.datasets.registry`, `ts2net/datasets/data/fred_panel.csv`.
- **Adoption docs**: `docs/method_chooser.md`, `docs/comparisons.md`, `examples/GALLERY.md`, `docs/API_STABILITY.md`, `MIGRATION.md`, `docs/adoption.rst`.
- **API tiers**: `ts2net.api_tiers` stable/experimental/deprecated lists; `tests/test_api_tiers.py`.
- **Recipe notebooks**: `examples/recipes/*.ipynb` for all five domain workflows.

### Changed
- Roadmap refocused from method expansion to product proof (reports, recipes, adoption).
- v1.0 target: freeze core API; see migration guide.
- v0.3 typing CI extended to `causal/summary`, `dynamic/summary`, `datasets/ucr`.
- `when_graphs_win.py` now compares all bundled UCR datasets with rollup narrative.

## [0.9.0] - 2026-06-14

### Added
- **Core graph expansion (`ts2net.graphs`)**: correlation, similarity, recurrence RQA, SAX/entropy symbolization, event networks, multiplex visibility, adaptive/cross recurrence.
- **Causal workflow (`ts2net.causal`)**: `run_causal_analysis()` with lag search, permutation/bootstrap confidence, confounder adjustment, and plain-language reports.
- **Causal discovery**: PC and FCI algorithms with lag-expanded time-series adapters (`pc_timeseries_network`, `fci_timeseries_network`).
- **Directed visibility asymmetry**: irreversibility and temporal-asymmetry metrics from directed HVG.
- **ML integration (`ts2net.sklearn`, `ts2net.ml`)**: `NetworkFeatureExtractor`, rolling features, feature selection, baseline comparisons, PyG/DGL adapters.
- **Dynamic analytics (`ts2net.dynamic`)**: `run_dynamic_analysis()` with regime detection, anomaly scores, community tracking, and node role evolution.
- **Scale & performance (`ts2net.scale`)**: streaming window iterators, chunk/Parquet readers, `IncrementalHVG`, approximate kNN, performance contracts, sparse CSR helpers.
- **Install extras**: `[ml]`, `[pyg]`, `[dgl]`, `[tsfresh]`, `[pipeline]`, `[approx]`.
- Examples: causal workflow, ML integration, dynamic analytics, causal discovery, scale streaming; Binder notebook for Spain meter case study.
- CI benchmark smoke test (`TS2NET_CI_SMOKE=1`).

### Changed
- Consolidated `horizons.md` and `ROADMAP.md` into a single `ROADMAP.md`.
- `build_windows()` supports `n_jobs` and `streaming=True`.
- `similarity_network()` supports `approximate=True` with auto-threshold for large panels.
- Slimmed core dependencies; optional features moved to extras.

### Fixed
- `soft_dtw_distance` returns 0 for identical series and clamps negative numerical noise.
- Lazy YAML import in `PipelineConfig.from_yaml()` when `[pipeline]` extra is not installed.

## [0.8.0] - 2026-05-15

### Fixed
- `cdist_dtw`: `usize::MAX` integer overflow caused all off-diagonal distances to return
  `inf`. When `band=None`, `i + usize::MAX` wrapped to 0 in release mode, making the
  inner loop body unreachable. Fix: explicit `match band` sets `jmax = m` directly for
  the unbounded case, with no arithmetic on `usize::MAX`.
- `cdist_dtw`: keyword argument `band=2` caused a segfault on Python 3.14 under
  PyO3 0.19. Root cause: `Option<usize>` keyword extraction was broken in that version.

### Changed
- Upgraded PyO3 0.19 → 0.28 (adds Python 3.14 support, fixes keyword-arg segfault).
- Upgraded numpy crate 0.19 → 0.28. Migrated all array return paths to Bound API:
  `into_pyarray(py).unbind()`, `from_vec(py, v).unbind()`.
- Removed direct `ndarray` dependency from `ts2net_rs/Cargo.toml`; switched import to
  `use numpy::ndarray` so lib.rs always uses the same ndarray version as numpy,
  eliminating the 0.15/0.17 type-split that broke CI on Python 3.12/3.13.
- `cdist_dtw` `band` parameter type changed from `Option<usize>` to `Option<u64>` at
  the Python boundary for reliable FFI across all Python versions.

### Added
- Strengthened `test_dtw_distance`: now asserts off-diagonal values are finite and
  positive. Previous test only checked shape and symmetry, which passed even when all
  off-diagonal values were `inf`.

## [0.6.0] - 2024-12-20

### Added
- BSTS (Bayesian Structural Time Series) decomposition and residual topology analysis
  - `ts2net.bsts` module with `decompose()`, `features()`, and `BSTSSpec`
  - Structural decomposition (level, trend, seasonal components)
  - Residual network analysis (HVG, NVG, transition on residuals)
  - Windowed analysis support for long series
  - YAML pipeline integration for BSTS
- Comprehensive test suite for correctness and invariants
  - Hard correctness tests comparing fast vs naive O(n²) implementations
  - Property-based tests for pathological ties (repeated values)
  - Cross-platform determinism tests
  - End-to-end pipeline tests with known fixtures
  - Performance regression tests
  - Data hygiene tests
- PyPI publishing workflow improvements
  - Triggers on version tags in addition to GitHub releases
  - Setup documentation (`PYPI_SETUP.md`)

### Changed
- Explicit tie-breaking rules documented for HVG and NVG
- Improved test coverage and organization

### Fixed
- Fixed indentation error in recurrence.py
- Fixed NVG test unpacking (4-value return)
- Fixed pipeline determinism test (sort results for comparison)

## [0.5.0] - 2024-12-19

### Added
- Real-world example using FRED economic data (`examples/example_fred_data.py`)
  - Fetches GDP, Unemployment Rate, and CPI from FRED
  - Demonstrates proximity networks from sliding windows
  - Network visualizations with signalplot
- Pre-push git hook for automated testing before pushing to main
- Examples documentation page (`docs/examples.rst`)
- `examples/images/` directory for generated visualizations
- Pre-push testing setup documentation

### Changed
- Restricted Python version support to 3.12+ (removed 3.9-3.11)
- Updated CI workflows to test only Python 3.12 and 3.13
- Simplified test suite (removed parity tests, reduced unit test verbosity)
- Updated documentation to use new API (`build()` instead of `fit_transform()`)
- Updated ReadTheDocs configuration to use Python 3.12
- Improved network visualizations with better styling and statistics

### Fixed
- Fixed RecurrenceNetwork and TransitionNetwork parameter mapping in API wrapper
- Fixed floating-point precision issues in distance tests
- Fixed approximate k-NN tests to handle feature matrices correctly
- Fixed MIC tests to properly check for minepy availability
- Fixed z-score normalization test expectations

### Removed
- Parity testing framework (R dependency removed)
- Redundant test files (`tests_visibility.py`, `tests_recurrance.py`, etc.)
- Duplicate `wheels.yml` workflow (using trusted publishing instead)

## [0.4.0] - 2024-12-19

### Added
- Initial release with core functionality
- Time series to network conversion using various methods
- Visibility graph algorithms (HVG, NVG)
- Recurrence network support
- Transition networks
- Multivariate time series support
- Rust bindings for performance-critical operations
- CLI interface
- Comprehensive test suite

### Changed
- Project structure optimized for distribution
- Documentation setup with Sphinx

[Unreleased]: https://github.com/kylejones200/ts2net/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/kylejones200/ts2net/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/kylejones200/ts2net/releases/tag/v0.8.0

