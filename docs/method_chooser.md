# Choose Your Method

Quick guide for picking a ts2net workflow. Start with the question, not the algorithm.

## Decision tree

```
Multivariate panel?
├─ Yes → Need causal drivers?
│        ├─ Yes → run_causal_analysis() (Granger or transfer entropy)
│        └─ No  → correlation_network() or similarity_network()
└─ No  → Need how structure changes over time?
         ├─ Yes → run_dynamic_analysis() + build_decision_package()
         └─ No  → Single snapshot graph:
                  ├─ Local shape / peaks → HVG or NVG
                  ├─ Recurrence / cycles → RecurrenceNetwork
                  └─ Symbolic dynamics   → TransitionNetwork
```

## Method cheat sheet

| Goal | Method | Entry point |
| ---- | ------ | ----------- |
| Explain peaks & visibility | HVG, NVG | `HVG().build(x)` |
| Recurrence / periodicity | Recurrence network | `RecurrenceNetwork()` |
| Symbolic dynamics | Transition / SAX | `TransitionNetwork()`, `sax_transition_network()` |
| Pairwise coupling | Correlation | `correlation_network(X)` |
| Analog finding | Similarity (DTW) | `similarity_network(X, method="dtw")` |
| Causal drivers | Granger / TE | `run_causal_analysis(X)` |
| Regime / drift | Rolling graphs | `run_dynamic_analysis(x)` |
| Governing equations | SINDy | `fit_sindy(X, t)` |
| Boss-ready summary | Reports | `build_graph_report(G)`, `build_decision_package(x)` |

## Backend selection

| Situation | Backend |
| --------- | ------- |
| Default | `backend="auto"` (Rust → Numba → Python) |
| No Rust build | `pip install` then `maturin develop` or accept Python fallback |
| Large panel distances | `ts_dist(..., executor="dask")` or `"ray"` |
| GPU correlation | `device="gpu"` where supported |

## Outputs

| Need | Use |
| ---- | --- |
| NetworkX graph | `builder.as_networkx()` |
| Sparse edges | `Graph` object / CSR helpers |
| ML features | `NetworkFeatureExtractor` |
| Human report | `build_graph_report()` |
| Decision evidence | `build_decision_package()` |

## Domain recipes

See [examples/recipes/README.md](../examples/recipes/README.md) for copy-paste industrial, energy, finance, observability, and healthcare workflows.

## Stability

Stable APIs for v1.0 are listed in [API_STABILITY.md](API_STABILITY.md).
