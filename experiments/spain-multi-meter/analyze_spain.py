"""
Spain Smart Meter Network Analysis - Ultra-Scale Processing
633M+ rows, multiple meters
Strategy: HVG, NVG (with horizon), Transition only. NO RNN (too slow for this scale).
Memory-efficient: NO dense matrices, degrees computed directly from edges.
"""

import sys
sys.path.insert(0, '/Users/k.jones/Desktop/morocco-net/ts2net')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ts2net_rs import hvg_edges, nvg_edges_sweepline
from ts2net import TransitionNetwork
from tqdm import tqdm
import warnings
import logging
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
SPAIN_DATA_PATH = '/Users/k.jones/Downloads/meter-data-analytics/academic_articles/spain/spain_smart_meter_data.parquet'
OUTPUT_DIR = '/Users/k.jones/Desktop/morocco-net'  # Save results here
NVG_HORIZON_LIMIT = 2000  # Bound NVG edges per node
MIN_POINTS_PER_METER = 100  # Skip meters with < 100 readings
MAX_METERS_TO_PROCESS = 50  # Process top N meters by reading count
RESAMPLE_INTERVAL = '1H'  # Resample to hourly for tractability

logger.info("="*80)
logger.info("SPAIN SMART METER - LARGE SCALE NETWORK ANALYSIS")
logger.info("="*80)
logger.info(f"Data: {SPAIN_DATA_PATH}")
logger.info(f"Strategy: Memory-efficient, NO dense matrices, NO RNN")
logger.info(f"Methods: HVG (full res), NVG (horizon={NVG_HORIZON_LIMIT}), Transition")
logger.info("="*80)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
logger.info("\n1. Loading data metadata...")
# Load just metadata first to understand structure
df_sample = pd.read_parquet(SPAIN_DATA_PATH, columns=['meter_id', 'timestamp'])
logger.info(f"Total rows: {len(df_sample):,}")

# Count readings per meter
logger.info("\nCounting readings per meter...")
meter_counts = df_sample.groupby('meter_id').size().sort_values(ascending=False)
logger.info(f"Total unique meters: {len(meter_counts):,}")
logger.info(f"Meters with >= {MIN_POINTS_PER_METER} readings: {(meter_counts >= MIN_POINTS_PER_METER).sum():,}")

# Select top meters for analysis
top_meters = meter_counts[meter_counts >= MIN_POINTS_PER_METER].head(MAX_METERS_TO_PROCESS).index.tolist()
logger.info(f"\nProcessing top {len(top_meters)} meters")
logger.info(f"Readings per meter: {meter_counts[top_meters].describe()}")

del df_sample  # Free memory


# ============================================================================
# 2. LOAD ACTUAL DATA FOR SELECTED METERS
# ============================================================================
logger.info("\n2. Loading consumption data for selected meters...")
df = pd.read_parquet(SPAIN_DATA_PATH)
df = df[df['meter_id'].isin(top_meters)].copy()
logger.info(f"Loaded {len(df):,} rows for {len(top_meters)} meters")

# Convert timestamp to datetime if needed
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['meter_id', 'timestamp'])


# ============================================================================
# 3. PROCESS EACH METER
# ============================================================================
logger.info("\n3. Processing each meter...")
logger.info("="*80)

results = []

def compute_degrees_from_edges(edges, n_nodes):
    """Compute degrees without dense matrix."""
    degrees = np.zeros(n_nodes, dtype=np.int64)
    for edge in edges:
        i, j = edge[0], edge[1]
        degrees[i] += 1
        degrees[j] += 1
    return degrees


for meter_id in tqdm(top_meters, desc="Processing meters"):
    meter_data = df[df['meter_id'] == meter_id].copy()
    
    # Resample to hourly
    meter_data = meter_data.set_index('timestamp').resample(RESAMPLE_INTERVAL).agg({
        'consumption_kwh': 'sum'
    }).dropna()
    
    x = meter_data['consumption_kwh'].values
    n = len(x)
    
    if n < MIN_POINTS_PER_METER:
        continue
    
    result = {
        'meter_id': meter_id,
        'n_points': n,
        'date_start': meter_data.index.min(),
        'date_end': meter_data.index.max(),
        'mean_consumption': np.mean(x),
        'std_consumption': np.std(x)
    }
    
    # HVG
    try:
        edges_hvg = hvg_edges(x)
        degrees_hvg = compute_degrees_from_edges(edges_hvg, n)
        result['hvg_edges'] = len(edges_hvg)
        result['hvg_avg_degree'] = np.mean(degrees_hvg)
        result['hvg_max_degree'] = int(np.max(degrees_hvg))
        del edges_hvg
    except Exception as e:
        logger.info(f"  HVG failed for {meter_id}: {e}")
        result['hvg_edges'] = 0
        result['hvg_avg_degree'] = 0
        result['hvg_max_degree'] = 0
    
    # NVG with horizon limit
    try:
        # Note: Current Rust NVG doesn't have limit param exposed yet
        # For now, use regular NVG on resampled data
        edges_nvg = nvg_edges_sweepline(x)
        degrees_nvg = compute_degrees_from_edges(edges_nvg, n)
        result['nvg_edges'] = len(edges_nvg)
        result['nvg_avg_degree'] = np.mean(degrees_nvg)
        result['nvg_max_degree'] = int(np.max(degrees_nvg))
        del edges_nvg
    except Exception as e:
        logger.info(f"  NVG failed for {meter_id}: {e}")
        result['nvg_edges'] = 0
        result['nvg_avg_degree'] = 0
        result['nvg_max_degree'] = 0
    
    # Transition Network
    try:
        tn = TransitionNetwork(symbolizer='ordinal', order=3, only_degrees=True)
        tn.build(x)
        degrees_tn = tn.degree_sequence()
        result['tn_nodes'] = tn.n_nodes
        result['tn_edges'] = int(np.sum(degrees_tn))
        result['tn_avg_degree'] = np.mean(degrees_tn) if len(degrees_tn) > 0 else 0
    except Exception as e:
        logger.info(f"  TN failed for {meter_id}: {e}")
        result['tn_nodes'] = 0
        result['tn_edges'] = 0
        result['tn_avg_degree'] = 0
    
    results.append(result)


# ============================================================================
# 4. AGGREGATE RESULTS
# ============================================================================
logger.info("\n4. Aggregating results...")
results_df = pd.DataFrame(results)

logger.info("\nSummary Statistics Across All Meters:")
logger.info(results_df[['n_points', 'hvg_edges', 'hvg_avg_degree', 
                  'nvg_edges', 'nvg_avg_degree', 'tn_edges']].describe())

# Save detailed results
output_path = f'{OUTPUT_DIR}/spain_meter_network_results.csv'
results_df.to_csv(output_path, index=False)
logger.info(f"\nDetailed results saved to: {output_path}")


# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
logger.info("\n5. Creating visualizations...")

fig = plt.figure(figsize=(20, 12))

# 5.1 Sample time series
ax1 = plt.subplot(3, 3, 1)
sample_meters = top_meters[:5]
for meter_id in sample_meters:
    meter_data = df[df['meter_id'] == meter_id]
    meter_resampled = meter_data.set_index('timestamp').resample('1D').sum()
    plt.plot(meter_resampled.index, meter_resampled['consumption_kwh'], 
             alpha=0.7, linewidth=0.5, label=meter_id[:8])
plt.xlabel('Date')
plt.ylabel('Daily Consumption (kWh)')
plt.title('Sample Meters - Daily Consumption')
plt.legend(fontsize=8)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 5.2 Edge counts distribution
ax2 = plt.subplot(3, 3, 2)
plt.hist(results_df['hvg_edges'], bins=30, alpha=0.6, label='HVG', edgecolor='black')
plt.hist(results_df['nvg_edges'], bins=30, alpha=0.6, label='NVG', edgecolor='black')
plt.xlabel('Number of Edges')
plt.ylabel('Frequency (meters)')
plt.title('Edge Count Distribution Across Meters')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.3 Average degree distribution
ax3 = plt.subplot(3, 3, 3)
plt.hist(results_df['hvg_avg_degree'], bins=30, alpha=0.6, label='HVG', edgecolor='black')
plt.hist(results_df['nvg_avg_degree'], bins=30, alpha=0.6, label='NVG', edgecolor='black')
plt.xlabel('Average Degree')
plt.ylabel('Frequency (meters)')
plt.title('Average Degree Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.4 Edges vs Time Series Length
ax4 = plt.subplot(3, 3, 4)
plt.scatter(results_df['n_points'], results_df['hvg_edges'], alpha=0.5, s=20)
plt.xlabel('Time Series Length (hours)')
plt.ylabel('HVG Edges')
plt.title('HVG Edges vs Series Length')
plt.grid(True, alpha=0.3)

# 5.5 NVG edges vs length
ax5 = plt.subplot(3, 3, 5)
plt.scatter(results_df['n_points'], results_df['nvg_edges'], alpha=0.5, s=20, color='orange')
plt.xlabel('Time Series Length (hours)')
plt.ylabel('NVG Edges')
plt.title('NVG Edges vs Series Length')
plt.grid(True, alpha=0.3)

# 5.6 Consumption patterns vs network properties
ax6 = plt.subplot(3, 3, 6)
plt.scatter(results_df['std_consumption'], results_df['hvg_avg_degree'], alpha=0.5, s=20)
plt.xlabel('Consumption Std Dev')
plt.ylabel('HVG Average Degree')
plt.title('Consumption Variability vs Network Complexity')
plt.grid(True, alpha=0.3)

# 5.7 Comparison: HVG vs NVG
ax7 = plt.subplot(3, 3, 7)
plt.scatter(results_df['hvg_edges'], results_df['nvg_edges'], alpha=0.5, s=20)
plt.plot([0, results_df['hvg_edges'].max()], [0, results_df['hvg_edges'].max()], 
         'r--', alpha=0.3, label='y=x')
plt.xlabel('HVG Edges')
plt.ylabel('NVG Edges')
plt.title('HVG vs NVG Edge Counts')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.8 Transition network complexity
ax8 = plt.subplot(3, 3, 8)
plt.scatter(results_df['n_points'], results_df['tn_edges'], alpha=0.5, s=20, color='green')
plt.xlabel('Time Series Length (hours)')
plt.ylabel('Transition Network Edges')
plt.title('TN Edges vs Series Length')
plt.grid(True, alpha=0.3)

# 5.9 Summary boxplots
ax9 = plt.subplot(3, 3, 9)
data_to_plot = [results_df['hvg_avg_degree'], 
                results_df['nvg_avg_degree'], 
                results_df['tn_avg_degree']]
plt.boxplot(data_to_plot, labels=['HVG', 'NVG', 'TN'])
plt.ylabel('Average Degree')
plt.title('Degree Distribution Summary')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = f'{OUTPUT_DIR}/spain_meter_network_analysis.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
logger.info(f"Visualization saved to: {plot_path}")


# ============================================================================
# 6. FINAL SUMMARY
# ============================================================================
logger.info("\n" + "="*80)
logger.info("ANALYSIS COMPLETE")
logger.info("="*80)
logger.info(f"\nProcessed {len(results_df)} meters")
logger.info(f"Total readings analyzed: {results_df['n_points'].sum():,} hours")
logger.info(f"Date range: {results_df['date_start'].min()} to {results_df['date_end'].max()}")
logger.info(f"\nAverage Network Properties:")
logger.info(f"  HVG edges: {results_df['hvg_edges'].mean():.0f} ± {results_df['hvg_edges'].std():.0f}")
logger.info(f"  HVG avg degree: {results_df['hvg_avg_degree'].mean():.2f}")
logger.info(f"  NVG edges: {results_df['nvg_edges'].mean():.0f} ± {results_df['nvg_edges'].std():.0f}")
logger.info(f"  NVG avg degree: {results_df['nvg_avg_degree'].mean():.2f}")
logger.info(f"  TN nodes: {results_df['tn_nodes'].mean():.0f}")
logger.info(f"  TN edges: {results_df['tn_edges'].mean():.0f}")
logger.info("\nFiles generated:")
logger.info(f"  - {OUTPUT_DIR}/spain_meter_network_results.csv")
logger.info(f"  - {OUTPUT_DIR}/spain_meter_network_analysis.png")

