

## Remaining Work

### Medium Priority Features

### Performance & Scalability
- **Out-of-core streaming builders**: Process data in chunks rather than requiring entire series in memory
- **GPU acceleration**: Via CuPy/PyTorch for high-volume workloads
- **Parallelization controls**: Expose job count, chunking in API
- **Benchmark suite**: Track performance curves across methods and data sizes

### API & Usability Improvements
- **Type hints and stubs**: Complete IDE support (partial - type hints added to fit/transform methods, but not complete across all modules)
- **Consistent docstrings**: Standardized sections (purpose, inputs, outputs, examples) (partial - standardized for fit/transform methods, but not complete across all modules)


### Causal Inference & Network-Based Causality
- **Transfer entropy networks**: Construct directed networks based on information-theoretic causality (transfer entropy between time series pairs)
- **Granger causality networks**: Build networks from Granger causality tests (linear and nonlinear variants)
- **Conditional transfer entropy**: Multi-variable causal inference accounting for confounding variables
- **Causal network metrics**: Path-based causality measures, causal strength, and directionality indices
- **Time-lagged network analysis**: Networks with temporal delays to capture causal relationships
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
- **Feature pipeline**: Extract network statistics and feed into ML workflows (classification/regression)
- **Benchmark comparisons**: Compare ts2net features to baseline methods on standard datasets
- **Feature selection routines**: Tailored to network statistics

### Core Method Enhancements
- **Adaptive recurrence networks**: Choose thresholds using statistical criteria
- **Alternative symbolization**: Entropy-maximizing, SAX variants for transition networks

### Testing, CI, and Quality
- **Expand test coverage**: Cover all methods and edge cases
- **Code coverage targets**: Enforce via CI
- **Fuzz tests**: Random time series to catch numerical errors

### Documentation & Tutorials
- **Gallery of examples**: Show how ts2net solves real tasks (anomaly detection, clustering, comparison)
- **Quick start notebooks**: On common platforms (Binder, Colab)
- **Comparison guides**: To alternative libraries

### Ecosystem & Interoperability
- **Scikit-learn wrappers**: Use ts2net features inside sklearn pipelines
- **Graph ML adapters**: PyTorch Geometric, DGL integration
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
