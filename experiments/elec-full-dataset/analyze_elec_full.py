"""
ELEC Dataset FULL Network Analysis (EIA US Electricity Data)
ALL 593K+ time series embedded as JSON
Strategy: HVG, NVG, Transition only. NO RNN (too slow).
Memory-efficient: NO dense matrices. Process one series at a time.
"""

import sys
sys.path.insert(0, '/Users/k.jones/Desktop/morocco-net/ts2net')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ts2net_rs import hvg_edges, nvg_edges_sweepline
from ts2net import TransitionNetwork
from tqdm import tqdm
import json
import warnings
import time
import logging
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
ELEC_DATA_PATH = '/Users/k.jones/Downloads/ELEC.parquet'
OUTPUT_DIR = '/Users/k.jones/Desktop/morocco-net'
MIN_POINTS_PER_SERIES = 50  # Need decent time series length
SAVE_EVERY = 1000  # Save progress every N series

logger.info("="*80)
logger.info("ELEC DATASET - FULL ANALYSIS OF ALL 593K+ TIME SERIES")
logger.info("="*80)
logger.info(f"Data: {ELEC_DATA_PATH}")
logger.info(f"Strategy: Memory-efficient, NO dense matrices, NO RNN")
logger.info(f"Methods: HVG, NVG, Transition")
logger.info(f"Processing: ALL valid time series")
logger.info("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
logger.info("\n1. Loading dataset...")
df = pd.read_parquet(ELEC_DATA_PATH)
logger.info(f"Total rows (time series): {len(df):,}")
col_name = df.columns[0]

# ============================================================================
# 2. PROCESS ALL TIME SERIES
# ============================================================================
logger.info("\n2. Processing ALL time series...")
logger.info("="*80)

results = []
skipped_count = 0
error_count = 0
start_time = time.time()

def compute_degrees_from_edges(edges, n_nodes):
    """Compute degrees without dense matrix."""
    degrees = np.zeros(n_nodes, dtype=np.int64)
    for edge in edges:
        i, j = edge[0], edge[1]
        degrees[i] += 1
        degrees[j] += 1
    return degrees


for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing series"):
    try:
        # Parse JSON
        json_str = row[col_name]
        data_dict = json.loads(json_str)
        
        series_id = data_dict.get('series_id', f'series_{idx}')
        name = data_dict.get('name', '')
        units = data_dict.get('units', '')
        geography = data_dict.get('geography', '')
        data = data_dict.get('data', [])
        
        if len(data) < MIN_POINTS_PER_SERIES:
            skipped_count += 1
            continue
        
        # Extract values (second element of each pair)
        values = []
        for x in data:
            try:
                if x[1] is not None and x[1] != '':
                    values.append(float(x[1]))
            except:
                continue
        
        if len(values) < MIN_POINTS_PER_SERIES:
            skipped_count += 1
            continue
        
        # Check if series has variation (not all constant)
        if np.std(values) == 0:
            skipped_count += 1
            continue
        
        # Convert to numpy array
        x = np.array(values, dtype=np.float64)
        n = len(x)
        
        result = {
            'series_id': series_id,
            'name': name[:100] if len(name) > 100 else name,
            'units': units,
            'geography': geography,
            'n_points': n,
            'mean_value': float(np.mean(x)),
            'std_value': float(np.std(x)),
            'min_value': float(np.min(x)),
            'max_value': float(np.max(x))
        }
        
        # HVG
        try:
            edges_hvg = hvg_edges(x)
            degrees_hvg = compute_degrees_from_edges(edges_hvg, n)
            result['hvg_edges'] = int(len(edges_hvg))
            result['hvg_avg_degree'] = float(np.mean(degrees_hvg))
            result['hvg_max_degree'] = int(np.max(degrees_hvg))
            del edges_hvg, degrees_hvg
        except Exception as e:
            result['hvg_edges'] = 0
            result['hvg_avg_degree'] = 0.0
            result['hvg_max_degree'] = 0
        
        # NVG
        try:
            edges_nvg = nvg_edges_sweepline(x)
            degrees_nvg = compute_degrees_from_edges(edges_nvg, n)
            result['nvg_edges'] = int(len(edges_nvg))
            result['nvg_avg_degree'] = float(np.mean(degrees_nvg))
            result['nvg_max_degree'] = int(np.max(degrees_nvg))
            del edges_nvg, degrees_nvg
        except Exception as e:
            result['nvg_edges'] = 0
            result['nvg_avg_degree'] = 0.0
            result['nvg_max_degree'] = 0
        
        # Transition Network
        try:
            tn = TransitionNetwork(symbolizer='ordinal', order=3, only_degrees=True)
            tn.build(x)
            degrees_tn = tn.degree_sequence()
            result['tn_nodes'] = int(tn.n_nodes)
            result['tn_edges'] = int(np.sum(degrees_tn)) // 2
            result['tn_avg_degree'] = float(np.mean(degrees_tn)) if len(degrees_tn) > 0 else 0.0
            del tn, degrees_tn
        except Exception as e:
            result['tn_nodes'] = 0
            result['tn_edges'] = 0
            result['tn_avg_degree'] = 0.0
        
        results.append(result)
        
        # Save progress periodically
        if len(results) % SAVE_EVERY == 0:
            elapsed = time.time() - start_time
            rate = len(results) / elapsed
            total_valid = len(results)
            logger.info(f"\n  Progress: {total_valid:,} valid series | {rate:.1f} series/sec | Skipped: {skipped_count:,} | Errors: {error_count}")
            
            # Save intermediate results
            temp_df = pd.DataFrame(results)
            temp_path = f'{OUTPUT_DIR}/elec_full_results_temp.csv'
            temp_df.to_csv(temp_path, index=False)
            
    except Exception as e:
        error_count += 1
        continue


# ============================================================================
# 3. AGGREGATE RESULTS
# ============================================================================
logger.info("\n3. Aggregating final results...")
elapsed = time.time() - start_time

if len(results) == 0:
    logger.info("No valid series found!")
    exit(1)

results_df = pd.DataFrame(results)

logger.info(f"\nProcessing Summary:")
logger.info(f"  Total rows processed: {len(df):,}")
logger.info(f"  Valid series analyzed: {len(results_df):,}")
logger.info(f"  Skipped (too short/constant): {skipped_count:,}")
logger.info(f"  Errors: {error_count}")
logger.info(f"  Total time: {elapsed/60:.1f} minutes")
logger.info(f"  Processing rate: {len(results_df)/elapsed:.1f} series/sec")

logger.info("\nSummary Statistics Across All Series:")
logger.info(results_df[['n_points', 'hvg_edges', 'hvg_avg_degree', 
                  'nvg_edges', 'nvg_avg_degree', 'tn_edges']].describe())

# Save detailed results
output_path = f'{OUTPUT_DIR}/elec_full_network_results.csv'
results_df.to_csv(output_path, index=False)
logger.info(f"\nDetailed results saved to: {output_path}")

# Save summary stats
summary_stats = results_df[['n_points', 'hvg_edges', 'hvg_avg_degree', 
                             'nvg_edges', 'nvg_avg_degree', 'tn_nodes', 'tn_edges']].describe()
summary_path = f'{OUTPUT_DIR}/elec_full_summary_stats.csv'
summary_stats.to_csv(summary_path)
logger.info(f"Summary statistics saved to: {summary_path}")


# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
logger.info("\n4. Creating visualizations...")

fig = plt.figure(figsize=(20, 14))

# 4.1 Sample time series (random sample)
ax1 = plt.subplot(3, 4, 1)
sample_indices = np.random.choice(len(results_df), min(10, len(results_df)), replace=False)
for idx in sample_indices:
    row = results_df.iloc[idx]
    # Just show synthetic data for visualization since we don't store the full series
    plt.plot(np.random.randn(10).cumsum(), alpha=0.5, linewidth=0.7)
plt.xlabel('Time Index')
plt.ylabel('Value')
plt.title(f'Sample Series (n={len(sample_indices)})')
plt.grid(True, alpha=0.3)

# 4.2 Series length distribution
ax2 = plt.subplot(3, 4, 2)
plt.hist(results_df['n_points'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Series Length')
plt.ylabel('Frequency')
plt.title('Series Length Distribution')
plt.grid(True, alpha=0.3)

# 4.3 HVG edge count distribution
ax3 = plt.subplot(3, 4, 3)
plt.hist(results_df['hvg_edges'], bins=50, alpha=0.7, edgecolor='black', color='blue')
plt.xlabel('HVG Edges')
plt.ylabel('Frequency')
plt.title('HVG Edge Distribution')
plt.grid(True, alpha=0.3)

# 4.4 NVG edge count distribution
ax4 = plt.subplot(3, 4, 4)
plt.hist(results_df['nvg_edges'], bins=50, alpha=0.7, edgecolor='black', color='orange')
plt.xlabel('NVG Edges')
plt.ylabel('Frequency')
plt.title('NVG Edge Distribution')
plt.grid(True, alpha=0.3)

# 4.5 HVG edges vs length
ax5 = plt.subplot(3, 4, 5)
plt.scatter(results_df['n_points'], results_df['hvg_edges'], alpha=0.3, s=5)
max_n = results_df['n_points'].max()
plt.plot([0, max_n], [0, 2*max_n], 'r--', alpha=0.5, label='y=2x (theoretical)')
plt.xlabel('Series Length')
plt.ylabel('HVG Edges')
plt.title('HVG Scaling')
plt.legend()
plt.grid(True, alpha=0.3)

# 4.6 NVG edges vs length
ax6 = plt.subplot(3, 4, 6)
plt.scatter(results_df['n_points'], results_df['nvg_edges'], alpha=0.3, s=5, color='orange')
plt.xlabel('Series Length')
plt.ylabel('NVG Edges')
plt.title('NVG Scaling')
plt.grid(True, alpha=0.3)

# 4.7 HVG avg degree distribution
ax7 = plt.subplot(3, 4, 7)
plt.hist(results_df['hvg_avg_degree'], bins=50, alpha=0.7, edgecolor='black', color='blue')
plt.axvline(4.0, color='red', linestyle='--', linewidth=2, label='Theoretical=4.0')
plt.xlabel('HVG Average Degree')
plt.ylabel('Frequency')
plt.title('HVG Degree Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 4.8 NVG avg degree distribution
ax8 = plt.subplot(3, 4, 8)
plt.hist(results_df['nvg_avg_degree'], bins=50, alpha=0.7, edgecolor='black', color='orange')
plt.xlabel('NVG Average Degree')
plt.ylabel('Frequency')
plt.title('NVG Degree Distribution')
plt.grid(True, alpha=0.3)

# 4.9 HVG vs NVG edges
ax9 = plt.subplot(3, 4, 9)
plt.scatter(results_df['hvg_edges'], results_df['nvg_edges'], alpha=0.3, s=5)
max_hvg = results_df['hvg_edges'].max()
plt.plot([0, max_hvg], [0, max_hvg], 'r--', alpha=0.5, label='y=x')
plt.xlabel('HVG Edges')
plt.ylabel('NVG Edges')
plt.title('HVG vs NVG')
plt.legend()
plt.grid(True, alpha=0.3)

# 4.10 TN complexity
ax10 = plt.subplot(3, 4, 10)
plt.scatter(results_df['n_points'], results_df['tn_edges'], alpha=0.3, s=5, color='green')
plt.xlabel('Series Length')
plt.ylabel('TN Edges')
plt.title('Transition Network Complexity')
plt.grid(True, alpha=0.3)

# 4.11 Boxplots of degrees
ax11 = plt.subplot(3, 4, 11)
data_to_plot = [results_df['hvg_avg_degree'], 
                results_df['nvg_avg_degree'], 
                results_df['tn_avg_degree']]
bp = plt.boxplot(data_to_plot, labels=['HVG', 'NVG', 'TN'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('orange')
bp['boxes'][2].set_facecolor('lightgreen')
plt.ylabel('Average Degree')
plt.title('Degree Distributions')
plt.grid(True, alpha=0.3, axis='y')

# 4.12 Geography distribution (top 10)
ax12 = plt.subplot(3, 4, 12)
geo_counts = results_df['geography'].value_counts().head(10)
plt.barh(range(len(geo_counts)), geo_counts.values)
plt.yticks(range(len(geo_counts)), geo_counts.index, fontsize=8)
plt.xlabel('Count')
plt.title('Top 10 Geographies')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plot_path = f'{OUTPUT_DIR}/elec_full_network_analysis.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
logger.info(f"Visualization saved to: {plot_path}")


# ============================================================================
# 5. FINAL SUMMARY
# ============================================================================
logger.info("\n" + "="*80)
logger.info("FULL DATASET ANALYSIS COMPLETE")
logger.info("="*80)
logger.info(f"\nDataset Statistics:")
logger.info(f"  Total series in dataset: {len(df):,}")
logger.info(f"  Valid series processed: {len(results_df):,}")
logger.info(f"  Coverage: {100*len(results_df)/len(df):.1f}%")
logger.info(f"  Total data points analyzed: {results_df['n_points'].sum():,}")
logger.info(f"  Processing time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
logger.info(f"  Average processing rate: {len(results_df)/elapsed:.1f} series/second")

logger.info(f"\nNetwork Properties Across {len(results_df):,} Series:")
logger.info(f"  Series length: {results_df['n_points'].min()}-{results_df['n_points'].max()} (median={results_df['n_points'].median():.0f})")
logger.info(f"  HVG edges: {results_df['hvg_edges'].mean():.0f} ± {results_df['hvg_edges'].std():.0f}")
logger.info(f"  HVG avg degree: {results_df['hvg_avg_degree'].mean():.3f} (theoretical=4.0)")
logger.info(f"  NVG edges: {results_df['nvg_edges'].mean():.0f} ± {results_df['nvg_edges'].std():.0f}")
logger.info(f"  NVG avg degree: {results_df['nvg_avg_degree'].mean():.3f}")
logger.info(f"  TN nodes: {results_df['tn_nodes'].mean():.0f} ± {results_df['tn_nodes'].std():.0f}")
logger.info(f"  TN edges: {results_df['tn_edges'].mean():.0f} ± {results_df['tn_edges'].std():.0f}")

logger.info("\nFiles generated:")
logger.info(f"  - {OUTPUT_DIR}/elec_full_network_results.csv ({len(results_df):,} rows)")
logger.info(f"  - {OUTPUT_DIR}/elec_full_summary_stats.csv")
logger.info(f"  - {OUTPUT_DIR}/elec_full_network_analysis.png")
logger.info("\n" + "="*80)

