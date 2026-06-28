Research validation (Horizon 9)
=================================

ts2net ships literature fixtures, bundled UCR datasets, and reproducibility manifests.

Datasets
--------

- :mod:`ts2net.datasets` — ``synthetic_causal``, ``synthetic_classification``,
  ``synthetic_anomaly``, ``synthetic_regime``, ``spain_meters_summary``
- Bundled UCR archives under ``ts2net/datasets/data/ucr/``
- :func:`ts2net.datasets.load_ucr`, :func:`ts2net.datasets.run_ucr_benchmark`

Fixtures
--------

Run all checks::

    python benchmarks/run_validation.py
    TS2NET_CI_SMOKE=1 python benchmarks/run_validation.py

Recorded UCR baselines live in ``ts2net/datasets/data/ucr_baselines.json``.

Outputs
-------

- ``benchmarks/results/validation_manifest.json``
- ``benchmarks/results/ucr_benchmark.json``
