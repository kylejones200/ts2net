# Migration Guide

How to upgrade ts2net installations across minor and major releases.

## Upgrading to v1.0 (from v0.9.x)

v1.0 **freezes** the builder and workflow APIs. Most v0.9 code runs unchanged.

### What stays the same

```python
from ts2net import HVG, graph_summary

builder = HVG()
builder.build(x)
G = builder.as_networkx()
print(graph_summary(G))
```

Causal and dynamic workflows:

```python
from ts2net.causal import run_causal_analysis, CausalWorkflowSpec
from ts2net.dynamic import run_dynamic_analysis, DynamicWorkflowSpec

causal = run_causal_analysis(X, spec=CausalWorkflowSpec(method="granger"))
print(causal.summary())

dynamic = run_dynamic_analysis(x, spec=DynamicWorkflowSpec(method="hvg", window=50))
print(dynamic.summary())
```

### New in v1.0 (recommended)

Replace ad-hoc `print(graph_summary(...))` with structured reports:

```python
from ts2net.reports import build_graph_report, build_decision_package

report = build_graph_report(G, method="hvg", parameters={"weighted": False})
print(report.summary())

package = build_decision_package(x, method="hvg", window=50)
print(package.to_markdown())
```

### Backend selection

Graph builders accept `backend="auto"|"rust"|"numba"|"python"`. Set `TS2NET_BACKEND`
or pass explicitly. If Rust is unavailable, `auto` falls back with a warning.

### Optional extras

Install feature groups explicitly:

```bash
pip install 'ts2net[sindy]'      # PySINDy dynamics
pip install 'ts2net[ml]'         # sklearn + PyG helpers
pip install 'ts2net[pipeline]'   # YAML CLI
```

### Breaking changes (v0.9 → v1.0)

None planned for core builders. Experimental modules (pipeline YAML, neural inference)
may rename parameters; check CHANGELOG before upgrading minors.

## Upgrading from v0.8.x or earlier

1. **DTW / Rust extension** — Rebuild: `maturin develop --manifest-path ts2net_rs/Cargo.toml`
2. **Causal API** — Prefer `run_causal_analysis()` over calling Granger/TE primitives separately
3. **Graph modules** — High-level functions live in `ts2net.graphs`; core builders remain in `ts2net.api`
4. **Dynamic analytics** — Use `ts2net.dynamic.run_dynamic_analysis()` instead of rolling windows alone

## Deprecations

Symbols in `ts2net.api_tiers.DEPRECATED` emit warnings. None are scheduled for removal
before v2.0 as of v1.0.0-rc.

## Getting help

- [ROADMAP.md](ROADMAP.md) — product direction
- [docs/API_STABILITY.md](docs/API_STABILITY.md) — tier definitions
- [CHANGELOG.md](CHANGELOG.md) — per-version notes
