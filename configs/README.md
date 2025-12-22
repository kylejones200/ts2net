# Configuration Files

YAML-based configuration for ts2net pipeline execution.

## Usage

**CLI (Recommended):**
```bash
ts2net run configs/spain_smart_meters.yaml
ts2net run configs/morocco_zones.yaml --validate-only  # Validate only
```

**Direct script:**
```bash
python scripts/run_from_config.py configs/spain_smart_meters.yaml
```

## Configuration Schema

### Dataset Section

```yaml
dataset:
  name: string              # Dataset identifier
  path: string              # Path to Parquet file
  id_col: string            # Column name for series ID (null for single series)
  time_col: string          # Column name for timestamps
  value_col: string         # Column name for values
  start: string | null      # Start date filter (ISO format, optional)
  end: string | null        # End date filter (ISO format, optional)
  tz: string | null         # Timezone (optional)
```

### Sampling Section

```yaml
sampling:
  frequency: string | null   # Aggregation frequency (e.g., 'hourly', 'daily', 'monthly')
  agg: string                # Aggregation function ('mean', 'sum', 'max', 'min')
  resample: bool            # Whether to resample (future use)
```

### Graphs Section

Each graph type (hvg, nvg, recurrence, transition) can be configured:

```yaml
graphs:
  hvg:
    enabled: bool
    output: string           # "edges", "degrees", or "stats"
    weighted: bool
    limit: int | null        # Horizon limit (NVG only)
  
  nvg:
    enabled: bool
    output: string
    limit: int | null        # Maximum temporal distance
    max_edges: int | null    # Maximum total edges
    max_edges_per_node: int | null
    max_memory_mb: float | null
  
  recurrence:
    enabled: bool
    output: string
    rule: string             # "knn" or "epsilon"
    k: int                   # For k-NN rule
    m: int                   # Embedding dimension
    tau: int                 # Time delay
    epsilon: float           # For epsilon rule
    metric: string           # Distance metric
  
  transition:
    enabled: bool
    output: string
    symbolizer: string       # "ordinal" or other
    order: int               # Pattern order
    n_states: int | null     # Number of states (optional)
```

### Windows Section

```yaml
windows:
  enabled: bool
  size: int | null          # Window size (null = process full series)
  step: int                 # Step size between windows
```

### Output Section

```yaml
output:
  format: string            # "parquet" or "json"
  path: string              # Output file path
  overwrite: bool
```

### Logging Section

```yaml
logging:
  log_errors: bool
  error_path: string | null # Path to error log JSON file
```

## Safety Checks

The pipeline includes built-in safety checks (in code, not YAML):

1. **Dense adjacency refusal**: Refuses to build dense matrices for n > 50,000 unless `force_dense: true` is set
2. **Recurrence exact refusal**: Refuses exact all-pairs recurrence for n > 50,000 (must use k-NN)
3. **Memory protection**: NVG limits (max_edges, max_memory_mb) are enforced

## Example Configurations

- `spain_smart_meters.yaml`: Large-scale smart meter data with hourly aggregation
- `morocco_zones.yaml`: Zone-level analysis with all graph types
- `north_dakota_wells.yaml`: Well production data with windowed analysis

## Creating New Configurations

1. Copy an existing config file
2. Update dataset paths and column names
3. Enable/disable graph types as needed
4. Adjust limits and parameters for your scale
5. Run: `python scripts/run_from_config.py configs/your_config.yaml`
