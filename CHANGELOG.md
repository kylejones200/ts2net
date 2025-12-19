# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.4.0] - 2024-12-18

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

[Unreleased]: https://github.com/kylejones200/ts2net/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/kylejones200/ts2net/releases/tag/v0.4.0

