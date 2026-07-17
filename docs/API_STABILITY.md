# API Stability Policy

ts2net v1.0 freezes the core. New graph methods are not the default backlog; reports,
recipes, and proof artifacts are.

## Tiers

| Tier | Meaning | Semver |
| ---- | ------- | ------ |
| **Stable** | Supported for production; breaking changes only in major releases | Guaranteed in 1.x |
| **Experimental** | API may change in minor releases | Use with pin or vend |
| **Deprecated** | Emits `DeprecationWarning`; removed in next major | Migrate before 2.0 |
| **Internal** | Underscore modules, CLI, benchmarks | No guarantee |

Programmatic lists live in `ts2net.api_tiers` (`STABLE`, `EXPERIMENTAL`, `DEPRECATED`).

## Stable surface (v1.0 target)

### Builders

- `HVG`, `NVG`, `RecurrenceNetwork`, `TransitionNetwork`
- `build_network()`, `build_windows()`
- `Graph`, `NetworkBuilder` protocol
- Pattern: `build(x)` or `fit(x).transform()` → graph or stats; configs as dataclasses

### Workflows

- `run_causal_analysis()` → `CausalAnalysisResult`
- `run_dynamic_analysis()` → `DynamicAnalysisResult`

### Reports & decisions

- `build_graph_report()` → `GraphReport`
- `build_decision_package()` → `DecisionPackage`
- `EdgeExplanation`, `NodeRoleSummary`, `DynamicChangeReport`

### Utilities

- `graph_summary(G)`
- `ValidationError`, `NotBuiltError`, `Ts2NetError`

## Experimental (may evolve)

- YAML pipeline (`PipelineConfig`, factory helpers)
- SINDy dynamics (`fit_sindy`, Rust backend)
- Neural inference / temporal CNN
- PC/FCI discovery adapters
- Multiplex graph helpers

## Builder contract

All first-class builders should:

1. Accept validated 1-D (or documented N-D) series input
2. Expose configuration via a dataclass or explicit kwargs with defaults
3. Record provenance on graph objects where applicable (`G.graph["method"]`, params)
4. Support `stats()` or equivalent without materialising full edge lists when possible

## Release process

See [CONTRIBUTING.md](../CONTRIBUTING.md#release-process). Each release updates
[CHANGELOG.md](../CHANGELOG.md) and, for breaking changes, [MIGRATION.md](../MIGRATION.md).
