# Method references (Horizon 9 / v0.9)

Short citations and assumptions for validation fixtures and core builders.

## Visibility graphs

- **HVG / NVG**: Lacasa et al. (2008) *PNAS* 105, 4972–4975. Natural visibility between samples; mean HVG degree ≈ 4 for uncorrelated series.
- **Directed irreversibility**: Lacasa et al. (2012) *Sci. Rep.* 2, 378. Compare forward vs reversed visibility statistics.

## Recurrence networks

- **Epsilon recurrence**: Marwan et al. (2007) *Phys. Rep.* 438, 237–379. Larger distance thresholds yield denser recurrence graphs (monotone edge count).

## Transfer entropy

- **TE**: Schreiber (2000) *Phys. Rev. Lett.* 85, 461. Asymmetric information flow; coupled logistic maps in `synthetic_causal` should show TE(X→Y) > TE(Y→X).

## Empirical validation

- **Spain meters**: Bundled `experiments/spain-multi-meter/spain_meter_network_results.csv`; per-meter HVG average degree clusters near 4.

## Baseline comparisons

- **Statistical**: mean, std, skew, kurtosis, lag-1 autocorrelation, trend, zero-crossing rate.
- **tsfresh / catch22 / sktime**: optional dependencies; used in `compare_feature_sets` UCR harness when installed.
