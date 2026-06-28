# Method references (Horizon 9 / v0.9)

Short citations and assumptions for validation fixtures and core builders.
Sphinx pages: ``docs/references.rst``.

## Visibility graphs

- **HVG / NVG**: Lacasa et al. (2008) *PNAS* 105, 4972–4975.
- **Directed irreversibility**: Lacasa et al. (2012) *Sci. Rep.* 2, 378.

## Recurrence networks and RQA

- **Epsilon recurrence**: Marwan et al. (2007) *Phys. Rep.* 438, 237–379.
- **RQA determinism**: periodic series → high DET (`rqa_sine_determinism` fixture).

## Transfer entropy

- **TE**: Schreiber (2000) *Phys. Rev. Lett.* 85, 461.

## PCMCI-style lagged discovery

- **Lagged TE panel**: Runge et al. (2019) *Nat. Commun.* 10, 2553 (`pcmci_lagged_driver` fixture).

## Empirical validation

- **Spain meters**: bundled CSV; HVG degree ≈ 4.
- **UCR**: GunPoint, ItalyPowerDemand, Coffee bundled under `ts2net/datasets/data/ucr/`.

## Baseline comparisons

- **Statistical / catch22 / sktime**: `compare_feature_sets()` and `run_ucr_benchmark()`.
- **Recorded baselines**: `ts2net/datasets/data/ucr_baselines.json`.
