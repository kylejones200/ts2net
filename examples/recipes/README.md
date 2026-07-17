# Domain Recipes

End-to-end workflows: **data → graph → report → decision**. Each script runs in under a minute on synthetic data.

| Recipe | Domain | Script |
| ------ | ------ | ------ |
| Industrial sensors | Drift, causal driver, failure precursor | `industrial_sensors.py` |
| Energy production | Well analogs, abnormal decline | `energy_production.py` |
| Finance | Regime change, rolling correlation | `finance_regime.py` |
| Observability | Service dependency, incident precursor | `observability_services.py` |
| Healthcare | Patient trajectory risk shift | `healthcare_trajectory.py` |
| Energy (real data) | Spain meter panel + UCR power demand | `energy_spain_real.py` |
| Finance (real data) | Bundled macro panel (FRED-style) | `finance_fred_real.py` |

```bash
uv run python examples/recipes/industrial_sensors.py
uv run python examples/recipes/energy_spain_real.py
uv run python examples/recipes/finance_fred_real.py
```

Replace synthetic arrays with your Parquet/CSV panels via `ts2net.io_adapters.from_pandas` or the pipeline CLI.

Each recipe uses `build_decision_package()` from `ts2net.reports` — the hook into Decision Systems workflows.

## Notebooks

Each domain has a matching `.ipynb` in this folder. Regenerate from scripts with:

```bash
uv run python scripts/generate_recipe_notebooks.py
```
