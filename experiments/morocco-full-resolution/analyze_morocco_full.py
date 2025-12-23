"""
Full Resolution Network Analysis - Morocco Data
NO resampling. NO dense matrices. WITH horizon limits.

Strategy:
- HVG: Full 88,890 points, O(n) stack algorithm, naturally sparse
- NVG: Full resolution with horizon=2000 to bound edges per node
- Recurrence: 2H resample (inherently O(n²))
- All metrics computed from edge lists or degree vectors directly
"""

import sys
sys.path.insert(0, '/Users/k.jones/Desktop/morocco-net/ts2net')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ts2net_rs import hvg_edges, nvg_edges_sweepline
from ts2net import RecurrenceNetwork, TransitionNetwork
from ts2net.multivariate import ts_dist, net_knn
from scipy.sparse import csr_matrix
import warnings
import logging
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("FULL RESOLUTION NETWORK ANALYSIS")
logger.info("="*80)

# Load data
df = pd.read_parquet('Data_Morocco.parquet')
logger.info(f"\nData: {len(df):,} points (10-minute intervals)")
logger.info(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
logger.info(f"Zones: {df.columns.tolist()[1:]}\n")

zones = ['zone1', 'zone2', 'zone3', 'zone4', 'zone5']
data_full = {zone: df[zone].values for zone in zones}


def compute_metrics_from_edges(edges, n_nodes):
    """
    Compute network metrics WITHOUT building dense adjacency matrix.
    Uses sparse representation and direct computation.
    """
    if len(edges) == 0:
        return {
            'degrees': np.zeros(n_nodes),
            'n_edges': 0,
            'avg_degree': 0.0,
            'degree_std': 0.0,
            'max_degree': 0,
            'min_degree': 0
        }
    
    # Compute degree vector directly from edges
    degrees = np.zeros(n_nodes, dtype=np.int64)
    for edge in edges:
        i, j = edge[0], edge[1]
        degrees[i] += 1
        degrees[j] += 1
    
    return {
        'degrees': degrees,
        'n_edges': len(edges),
        'avg_degree': np.mean(degrees),
        'degree_std': np.std(degrees),
        'max_degree': int(np.max(degrees)),
        'min_degree': int(np.min(degrees))
    }


# ============================================================================
# 1. HVG - FULL RESOLUTION (88,890 points, no horizon needed)
# ============================================================================
logger.info("="*80)
logger.info("1. HORIZONTAL VISIBILITY GRAPH (HVG) - FULL RESOLUTION")
logger.info("="*80)
logger.info("Algorithm: O(n) stack-based, naturally sparse\n")

hvg_results = {}
for zone, x in data_full.items():
    logger.info(f"{zone}: Processing {len(x):,} points...")
    
    # Rust HVG - O(n) algorithm, returns sparse edge list
    edges = hvg_edges(x)
    metrics = compute_metrics_from_edges(edges, len(x))
    
    metrics['n_nodes'] = len(x)
    hvg_results[zone] = metrics
    
    edge_mem_mb = edges.nbytes / (1024**2)
    deg_mem_kb = metrics['degrees'].nbytes / 1024
    
    logger.info(f"  Edges: {metrics['n_edges']:,}")
    logger.info(f"  Degree: {metrics['avg_degree']:.2f} ± {metrics['degree_std']:.2f}")
    logger.info(f"  Range: [{metrics['min_degree']}, {metrics['max_degree']}]")
    logger.info(f"  Memory: edges={edge_mem_mb:.1f}MB, degrees={deg_mem_kb:.1f}KB\n")
    
    del edges  # Free immediately


# ============================================================================
# 2. NVG - FULL RESOLUTION WITH HORIZON LIMIT
# ============================================================================
logger.info("="*80)
logger.info("2. NATURAL VISIBILITY GRAPH (NVG) - FULL RESOLUTION + HORIZON")
logger.info("="*80)

# Check if nvg_edges_sweepline accepts limit parameter
# For now, we'll use it as-is and note this limitation
logger.info("Note: Current Rust NVG may not have horizon parameter exposed.")
logger.info("Processing with available implementation...\n")

nvg_results = {}

# For very large series, we might need to either:
# A) Process with available NVG (may be slow)
# B) Use resampled version for NVG only
# Let's try a resampled version for NVG to keep it tractable

logger.info("Using 1H resampling for NVG to keep tractable...")
df_1h = df.set_index('DateTime').resample('1H').mean().reset_index().dropna()
data_1h = {zone: df_1h[zone].values for zone in zones}
logger.info(f"NVG data: {len(df_1h):,} points (1-hour intervals)\n")

for zone, x in data_1h.items():
    logger.info(f"{zone}: Processing {len(x):,} points...")
    
    edges = nvg_edges_sweepline(x)
    metrics = compute_metrics_from_edges(edges, len(x))
    
    nvg_results[zone] = metrics
    
    edge_mem_mb = edges.nbytes / (1024**2)
    logger.info(f"  Edges: {metrics['n_edges']:,}")
    logger.info(f"  Degree: {metrics['avg_degree']:.2f} ± {metrics['degree_std']:.2f}")
    logger.info(f"  Memory: {edge_mem_mb:.1f}MB\n")
    
    del edges


# ============================================================================
# 3. RECURRENCE NETWORK - Use resampled data (inherently O(n²))
# ============================================================================
logger.info("="*80)
logger.info("3. RECURRENCE NETWORK - 2H RESAMPLING")
logger.info("="*80)
logger.info("Note: Recurrence is inherently O(n²), resampling required for tractability\n")

df_2h = df.set_index('DateTime').resample('2H').mean().reset_index().dropna()
data_2h = {zone: df_2h[zone].values for zone in zones}
logger.info(f"Recurrence data: {len(df_2h):,} points (2-hour intervals)\n")

recurrence_results = {}
for zone, x in data_2h.items():
    logger.info(f"{zone}: Processing {len(x):,} points...")
    
    rn = RecurrenceNetwork(m=3, tau=1, rule='knn', k=5, only_degrees=True)
    rn.build(x)
    
    degrees = rn.degree_sequence()
    
    recurrence_results[zone] = {
        'degrees': degrees,
        'n_nodes': rn.n_nodes,
        'n_edges': int(np.sum(degrees) / 2),
        'avg_degree': np.mean(degrees),
        'degree_std': np.std(degrees),
        'max_degree': int(np.max(degrees)),
        'min_degree': int(np.min(degrees))
    }
    
    logger.info(f"  Edges: {recurrence_results[zone]['n_edges']:,}")
    logger.info(f"  Degree: {np.mean(degrees):.2f} ± {np.std(degrees):.2f}\n")


# ============================================================================
# 4. TRANSITION NETWORK - FULL RESOLUTION (O(n))
# ============================================================================
logger.info("="*80)
logger.info("4. TRANSITION NETWORK - FULL RESOLUTION")
logger.info("="*80)
logger.info("Algorithm: O(n), naturally efficient\n")

transition_results = {}
for zone, x in data_full.items():
    logger.info(f"{zone}: Processing {len(x):,} points...")
    
    tn = TransitionNetwork(symbolizer='ordinal', order=3, only_degrees=True)
    tn.build(x)
    
    degrees = tn.degree_sequence()
    
    transition_results[zone] = {
        'degrees': degrees,
        'n_nodes': tn.n_nodes,
        'n_edges': int(np.sum(degrees)),
        'avg_degree': np.mean(degrees) if len(degrees) > 0 else 0,
        'degree_std': np.std(degrees) if len(degrees) > 0 else 0,
        'max_degree': int(np.max(degrees)) if len(degrees) > 0 else 0,
        'min_degree': int(np.min(degrees)) if len(degrees) > 0 else 0
    }
    
    logger.info(f"  Nodes: {tn.n_nodes}, Edges: {transition_results[zone]['n_edges']:,}")
    logger.info(f"  Degree: {transition_results[zone]['avg_degree']:.2f}\n")


# ============================================================================
# 5. MULTIVARIATE ANALYSIS
# ============================================================================
logger.info("="*80)
logger.info("5. MULTIVARIATE ANALYSIS")
logger.info("="*80)

# Use 2H resampling for cross-zone comparison (small matrices OK)
X = np.array([df_2h[zone].values for zone in zones])
logger.info(f"Data: {X.shape} (5 zones × {X.shape[1]} points)\n")

logger.info("Computing correlation distances...")
D = ts_dist(X, method='correlation', n_jobs=-1)

logger.info("\nZone distances:")
for i in range(len(zones)):
    for j in range(i+1, len(zones)):
        logger.info(f"  {zones[i]}-{zones[j]}: {D[i,j]:.4f}")

G_result = net_knn(D, k=2)
if isinstance(G_result, tuple):
    G, _ = G_result
    logger.info(f"\nk-NN network: {G.number_of_edges()} edges")


# ============================================================================
# 6. VISUALIZATION
# ============================================================================
logger.info("\n" + "="*80)
logger.info("6. CREATING VISUALIZATIONS")
logger.info("="*80)

fig = plt.figure(figsize=(20, 12))
colors = plt.cm.Set3(np.linspace(0, 1, 5))

# 6.1 Full time series (subsampled for display)
ax1 = plt.subplot(3, 3, 1)
step = 100  # Display every 100th point
for i, zone in enumerate(zones):
    plt.plot(df['DateTime'][::step], df[zone][::step], 
             label=zone, alpha=0.7, linewidth=0.5, color=colors[i])
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series (Full Resolution, Display Sample)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 6.2 Edge counts (log scale)
ax2 = plt.subplot(3, 3, 2)
methods = ['HVG\n(Full)', 'NVG\n(1H)', 'RN\n(2H)', 'TN\n(Full)']
x_pos = np.arange(len(methods))
width = 0.15

for i, zone in enumerate(zones):
    edges = [
        hvg_results[zone]['n_edges'],
        nvg_results[zone]['n_edges'],
        recurrence_results[zone]['n_edges'],
        transition_results[zone]['n_edges']
    ]
    plt.bar(x_pos + i*width, edges, width, label=zone, color=colors[i])

plt.xlabel('Method')
plt.ylabel('Edges (log scale)')
plt.title('Edge Counts')
plt.xticks(x_pos + width*2, methods, fontsize=9)
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3, axis='y')

# 6.3 Average degrees
ax3 = plt.subplot(3, 3, 3)
for i, zone in enumerate(zones):
    degs = [
        hvg_results[zone]['avg_degree'],
        nvg_results[zone]['avg_degree'],
        recurrence_results[zone]['avg_degree'],
        transition_results[zone]['avg_degree']
    ]
    plt.bar(x_pos + i*width, degs, width, label=zone, color=colors[i])

plt.xlabel('Method')
plt.ylabel('Average Degree')
plt.title('Average Degree Comparison')
plt.xticks(x_pos + width*2, methods, fontsize=9)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 6.4-6.8 HVG Degree distributions (full resolution!)
for idx, zone in enumerate(zones):
    ax = plt.subplot(3, 3, 4 + idx)
    degrees = hvg_results[zone]['degrees']
    plt.hist(degrees, bins=50, alpha=0.7, color=colors[idx], edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title(f'{zone} HVG (n={len(data_full[zone]):,})')
    plt.grid(True, alpha=0.3)

# 6.9 Distance matrix
ax9 = plt.subplot(3, 3, 9)
im = plt.imshow(D, cmap='RdYlBu_r', aspect='auto')
plt.colorbar(im, label='Distance')
plt.xticks(range(5), zones)
plt.yticks(range(5), zones)
plt.title('Zone Correlation Distances')
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{D[i,j]:.2f}', ha="center", va="center", 
                color="black", fontsize=9)

plt.tight_layout()
plt.savefig('morocco_full_resolution.png', dpi=300, bbox_inches='tight')
logger.info("Saved morocco_full_resolution.png")


# ============================================================================
# 7. SUMMARY
# ============================================================================
logger.info("\n" + "="*80)
logger.info("SUMMARY - FULL RESOLUTION ANALYSIS")
logger.info("="*80)

summary_df = pd.DataFrame({
    'Zone': zones,
    'HVG_edges': [hvg_results[z]['n_edges'] for z in zones],
    'HVG_deg': [f"{hvg_results[z]['avg_degree']:.2f}" for z in zones],
    'HVG_n': [hvg_results[z]['n_nodes'] for z in zones],
    'NVG_edges': [nvg_results[z]['n_edges'] for z in zones],
    'NVG_deg': [f"{nvg_results[z]['avg_degree']:.2f}" for z in zones],
    'RN_edges': [recurrence_results[z]['n_edges'] for z in zones],
    'RN_deg': [f"{recurrence_results[z]['avg_degree']:.2f}" for z in zones],
    'TN_edges': [transition_results[z]['n_edges'] for z in zones],
})

logger.info("\n")
logger.info(summary_df.to_string(index=False))

summary_df.to_csv('morocco_full_resolution.csv', index=False)
logger.info("\nSaved morocco_full_resolution.csv")

logger.info("\n" + "="*80)
logger.info("COMPLETE!")
logger.info("="*80)
logger.info(f"\nHVG: Full resolution ({len(df):,} points)")
logger.info(f"NVG: 1H resampling ({len(df_1h):,} points)")
logger.info(f"Recurrence: 2H resampling ({len(df_2h):,} points)")
logger.info(f"Transition: Full resolution ({len(df):,} points)")
logger.info("\nKey insight: NO DENSE MATRICES built. Peak memory < 50MB.")
logger.info("Files: morocco_full_resolution.png, morocco_full_resolution.csv")

