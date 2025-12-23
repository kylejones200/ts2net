# Spain Smart Meter Multi-Series Analysis

## Experiment Configuration

**Dataset:** `spain_smart_meter_data.parquet`
- **Total rows:** 633,130,317 (633 million)
- **Unique meters:** 25,559
- **Meters analyzed:** Top 50 by reading count
- **Readings per meter:** 43,959 to 116,409 (mean: 59,422)
- **Date range:** 2014-11-02 to 2022-06-05 (8 years)
- **Temporal resolution:** Half-hourly → resampled to hourly

**Methods Applied:**
- Horizontal Visibility Graph (HVG) - full resolution
- Natural Visibility Graph (NVG) - horizon limit 2000
- Transition Network (TN) - ordinal patterns (order=3)
- Recurrence Network (RN) - **EXCLUDED** (too slow for this scale)

**Strategy:**
- Process each meter independently
- Resample from 30-minute to 1-hour intervals for tractability
- Memory-efficient: no dense matrices, degrees computed from edge lists
- Save progress every 1000 meters

**Hardware:**
- Memory: Low constant usage per meter
- Runtime: ~42 seconds for 50 meters
- Processing: ~1.2 meters/second

## Summary Results

See: `spain_meter_network_results.csv`

**Aggregate statistics across 50 meters:**

| Metric | Mean | Std | Min | 25% | Median | 75% | Max |
|--------|------|-----|-----|-----|--------|-----|-----|
| n_points | 46,759 | 14,232 | 15,408 | 43,968 | 47,340 | 59,989 | 62,019 |
| hvg_edges | 93,293 | 28,559 | 30,781 | 87,620 | 94,656 | 119,819 | 123,907 |
| hvg_avg_degree | 3.99 | 0.01 | 3.96 | 3.98 | 3.99 | 4.00 | 4.00 |
| nvg_edges | 282,421 | 139,751 | 65,419 | 181,724 | 252,634 | 368,844 | 623,181 |
| nvg_avg_degree | 11.90 | 4.03 | 4.95 | 9.29 | 10.29 | 13.62 | 22.17 |
| tn_nodes | 53 | 4 | 38 | 51 | 53 | 55 | 61 |
| tn_edges | 315 | 42 | 38 | 324 | 324 | 324 | 324 |

**Key observations:**
- Total hours analyzed: 2,337,965 (across 50 meters)
- HVG degree consistently at ~4.0 (validates theory at scale)
- NVG shows high variability between meters (4.95 to 22.17 avg degree)
- Most meters have 324 transition network edges (maximal ordinal pattern connectivity)

## Key Findings

1. **Scale achievement**: Successfully processed 2.3M hourly readings in under a minute without RNN

2. **HVG consistency**: Average degree of 3.99 holds across diverse consumption patterns and meter types

3. **NVG variability**: Wide range (4.95 to 22.17) suggests different consumption behaviors:
   - Low degree: more predictable, regular patterns
   - High degree: complex, irregular consumption

4. **Linear scaling validated**: HVG edges scale perfectly with series length (see plot)

5. **Meter heterogeneity**: Despite similar time spans, meters show distinct network signatures

## Visualization

See: `spain_meter_network_analysis.png`

![Spain Multi-Meter Analysis](../spain_meter_network_analysis.png)

The figure shows:
- **Consumption time series** (top left): Daily and seasonal patterns visible across sample meters
- **Edge distributions** (top middle/right): HVG tightly distributed, NVG shows spread
- **Perfect HVG scaling** (middle left): Linear relationship confirms O(n) algorithm
- **Variability vs complexity** (middle right): Low correlation between consumption std and network degree
- **Degree boxplots** (bottom right): Clear separation between HVG (~4), NVG (~12), TN (~4)

## Files Generated

- `spain_meter_network_results.csv` - Per-meter metrics (50 rows)
- `spain_meter_network_analysis.png` - Multi-panel visualization
- `analyze_spain.py` - Analysis script with configuration

## Reproducibility

**Note:** This experiment was run on a specific experimental setup. The script references data files and paths from that environment. To reproduce:

1. Ensure `ts2net` is installed with all dependencies
2. Provide the Spain smart meter dataset (633M rows)
3. Update paths in the script as needed
4. Run the analysis:

```bash
cd experiments/spain-multi-meter
python analyze_spain.py
```

**Runtime:** ~42 seconds for 50 meters

**Original dataset:** Spain smart meter data (633M rows) - dataset path needs to be configured in script

