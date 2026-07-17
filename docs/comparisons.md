# Comparison Guide

How ts2net relates to common tools — and when to combine them.

## ts2net vs feature libraries

| Tool | What it does | ts2net overlap |
| ---- | ------------ | -------------- |
| **tsfresh** | Massive automatic feature table | `compare_feature_sets()`, statistical baselines in sklearn module |
| **catch22** | 22 canonical time-series features | Use as baseline; graphs when shape/topology matters |
| **sktime** | Forecasting & classification pipelines | Export graph features into sktime-compatible arrays |
| **matrix profile** | Motif/discord discovery | `similarity_network(..., method="matrix_profile")` |

**When graphs win:** classification where temporal *shape* or recurrence structure differs by class (see `benchmarks/when_graphs_win.md`).

**When tables win:** smooth, low-dimensional series where mean/variance/spectral bands suffice.

## ts2net vs NetworkX

| | NetworkX | ts2net |
| - | -------- | ------ |
| Role | General graph library | Time series → graph *construction* |
| Input | Graph | Raw series |
| Use together | Always | `builder.as_networkx()` then NetworkX algorithms |

ts2net builds the graph; NetworkX analyzes it.

## ts2net vs PyTorch Geometric / DGL

| | PyG / DGL | ts2net |
| - | --------- | ------ |
| Role | Graph neural networks | Graph construction + classical features |
| Entry | `Data` / DGLGraph | `to_pyg_data()`, `to_dgl_graph()` |

Pipeline: ts2net graph → PyG/DGL → GNN training.

## ts2net vs PySINDy

| | PySINDy | ts2net |
| - | ------- | ------ |
| Role | Discover ODEs from data | Graph + network views of dynamics |
| ts2net | `fit_sindy()` with Rust or PySINDy backend; `sindy_coupling_network()` |

Use SINDy for equations; use graph methods for topology, causality, and reporting.

## ts2net vs causal discovery libraries

PCMCI, Tigramite, etc. provide rigorous causal discovery. ts2net wraps Granger/TE workflows with **reports** and **decision packages** for applied teams.

Use `run_causal_analysis()` for integrated lag search + network + plain-language summary.

## Adoption path

1. `examples/quick_start.py` — 5 minutes
2. `docs/method_chooser.md` — pick a method
3. `examples/recipes/` — domain workflow
4. `build_decision_package()` — evidence for stakeholders
