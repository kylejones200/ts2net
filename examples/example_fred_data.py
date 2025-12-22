"""
Example using real economic data from FRED (Federal Reserve Economic Data).

This example demonstrates:
- Fetching correlated economic time series from FRED
- Building networks from multiple time series
- Analyzing relationships between economic indicators
"""

import sys
import os
# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure images directory exists
_images_dir = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(_images_dir, exist_ok=True)

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

try:
    import pandas_datareader.data as web
    HAS_PDR = True
except ImportError:
    HAS_PDR = False
    logging.error("pandas_datareader not installed. Install with: pip install pandas-datareader")

try:
    import signalplot as splt
    HAS_SIGNALPLOT = True
except ImportError:
    HAS_SIGNALPLOT = False
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        logging.error("signalplot or matplotlib not installed. Install with: pip install signalplot")

from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork
from ts2net.multivariate import ts_dist, net_knn, net_enn
from ts2net.core import graph_summary
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def fetch_fred_data(start_date='2010-01-01', end_date=None):
    """
    Fetch correlated economic indicators from FRED.
    
    Returns:
        DataFrame with columns: GDP, UNRATE, CPIAUCSL
    """
    if not HAS_PDR:
        logger.error("pandas_datareader required. Install with: pip install pandas-datareader")
        return None
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info("Fetching economic data from FRED (start: %s, end: %s)", start_date, end_date)
    
    # FRED series codes for correlated economic indicators
    series = {
        'GDP': 'GDP',                    # Gross Domestic Product
        'UNRATE': 'UNRATE',              # Unemployment Rate
        'CPIAUCSL': 'CPIAUCSL',          # Consumer Price Index
    }
    
    data = {}
    for name, code in series.items():
        try:
            df = web.DataReader(code, 'fred', start=start_date, end=end_date)
            if not df.empty:
                # Use first column (some series have multiple columns)
                data[name] = df.iloc[:, 0]
                logger.info("%s (%s): %d points", name, code, len(data[name]))
            else:
                logger.warning("%s (%s): No data", name, code)
        except Exception as e:
            logger.warning("%s (%s): %s", name, code, str(e))
    
    if not data:
        logger.error("Failed to fetch any data from FRED")
        return None
    
    # Combine into DataFrame, forward-fill missing values
    df = pd.DataFrame(data)
    df = df.ffill().dropna()  # Forward fill, then drop any remaining NaN
    
    logger.info("Combined: %d points × %d series", len(df), len(df.columns))
    return df


def example_univariate_analysis(df):
    """Analyze individual time series with visibility graphs."""
    logger.info("Univariate analysis: Visibility graphs")
    
    for col in df.columns:
        logger.info("%s", col)
        x = df[col].values
        
        # Normalize to [0, 1] for better visualization
        x_norm = (x - x.min()) / (x.max() - x.min())
        
        # HVG
        hvg = HVG()
        hvg.build(x_norm)
        logger.info("  HVG: %d nodes, %d edges, density=%.4f", 
                   hvg.n_nodes, hvg.n_edges, 
                   hvg.n_edges / (hvg.n_nodes * (hvg.n_nodes - 1) / 2))
        
        # NVG
        nvg = NVG()
        nvg.build(x_norm)
        logger.info("  NVG: %d nodes, %d edges, density=%.4f", 
                   nvg.n_nodes, nvg.n_edges,
                   nvg.n_edges / (nvg.n_nodes * (nvg.n_nodes - 1) / 2))
        
        # Graph summary
        G_hvg = hvg.as_networkx()
        summary = graph_summary(G_hvg)
        logger.info("  Clustering: %.4f, Avg path length: %.2f", 
                   summary.get('clustering', 0), summary.get('path_length', 0))
        
        # Visualize HVG - sample a subgraph for large networks
        if hvg.n_nodes > 100:
            # Sample a connected subgraph for visualization
            logger.info("  Sampling subgraph for visualization (full graph: %d nodes)", hvg.n_nodes)
            # Get largest connected component
            components = list(nx.connected_components(G_hvg))
            largest_cc = max(components, key=len)
            # Sample nodes from largest component
            sample_size = min(50, len(largest_cc))
            sample_nodes = list(largest_cc)[:sample_size]
            G_sample = G_hvg.subgraph(sample_nodes).copy()
            
            plot_network(
                G_sample,
                title=f'{col} - HVG Subgraph (sample of {sample_size} nodes)',
                filename=f'examples/images/fred_hvg_{col.lower()}.png',
                layout='spring'
            )
        else:
            plot_network(
                G_hvg,
                title=f'{col} - Horizontal Visibility Graph (HVG)',
                filename=f'examples/images/fred_hvg_{col.lower()}.png',
                layout='spring'
            )


def example_proximity_network(df):
    """Build proximity network from sliding windows of a time series."""
    logger.info("Proximity network: Sliding window analysis")
    
    # Use one series (Unemployment Rate) and create windows
    col = 'UNRATE'
    if col not in df.columns:
        col = df.columns[0]
    
    x = df[col].values
    logger.info("Creating windows from: %s (%d points)", col, len(x))
    
    # Create sliding windows
    from ts2net.multivariate.windows import ts_to_windows
    window_width = 12  # 12 months = 1 year
    windows = ts_to_windows(x, width=window_width, by=1)
    logger.info("Created %d windows of width %d", windows.shape[0], window_width)
    
    # Build proximity network from windows
    D = ts_dist(windows, method='correlation', n_jobs=1)
    logger.info("Distance matrix: %s", D.shape)
    
    # Use ε-NN to create a sparse, meaningful network
    G, A = net_enn(D, percentile=15, weighted=True)  # Connect top 15% similar windows
    
    logger.info("Network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    
    # Network properties
    clustering = nx.average_clustering(G)
    density = G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2)
    logger.info("Clustering: %.4f, Density: %.4f", clustering, density)
    
    # Visualize the proximity network
    plot_network(
        G,
        title=f'{col} Proximity Network (Sliding Windows)',
        filename='examples/images/fred_proximity_network.png',
        layout='spring'
    )
    
    return G


def example_multivariate_network(df):
    """Build network from multiple correlated time series."""
    logger.info("Multivariate analysis: Time series network")
    
    # Prepare data: each row is a time point, each column is a series
    X = df.values.T  # Shape: (n_series, n_timepoints)
    logger.info("Data shape: %d series × %d time points", X.shape[0], X.shape[1])
    
    # Normalize each series
    X_norm = np.array([(x - x.min()) / (x.max() - x.min()) for x in X])
    
    # Compute distance matrix
    logger.info("Computing correlation distance...")
    D = ts_dist(X_norm, method='correlation', n_jobs=1)
    logger.info("Distance matrix: %s, range: [%.4f, %.4f]", D.shape, D.min(), D.max())
    
    # Build k-NN network
    G, A = net_knn(D, k=2, weighted=True, directed=False)
    logger.info("Network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    
    # Network properties
    clustering = nx.average_clustering(G)
    density = G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2)
    logger.info("Clustering: %.4f, Density: %.4f", clustering, density)
    
    return G


def example_recurrence_network(df):
    """Analyze one series with recurrence network."""
    logger.info("Recurrence network: Phase space analysis")
    
    # Use GDP as example
    col = 'GDP'
    if col not in df.columns:
        col = df.columns[0]
    
    x = df[col].values
    x_norm = (x - x.min()) / (x.max() - x.min())
    
    logger.info("Series: %s (%d points)", col, len(x_norm))
    
    # Recurrence network with different parameters
    for m, tau, k in [(2, 1, 5), (3, 1, 8), (3, 2, 10)]:
        try:
            rn = RecurrenceNetwork(m=m, tau=tau, rule='knn', k=k)
            rn.build(x_norm)
            
            logger.info("m=%d, τ=%d, k=%d: %d nodes, %d edges", 
                       m, tau, k, rn.n_nodes, rn.n_edges)
        except Exception as e:
            logger.warning("m=%d, τ=%d, k=%d failed: %s", m, tau, k, str(e))


def example_transition_network(df):
    """Analyze one series with transition network."""
    logger.info("Transition network: Symbolic dynamics")
    
    # Use Unemployment Rate as example
    col = 'UNRATE'
    if col not in df.columns:
        col = df.columns[0]
    
    x = df[col].values
    x_norm = (x - x.min()) / (x.max() - x.min())
    
    logger.info("Series: %s (%d points)", col, len(x_norm))
    
    # Transition network with ordinal patterns
    for order in [3, 4, 5]:
        try:
            tn = TransitionNetwork(symbolizer='ordinal', order=order)
            tn.build(x_norm)
            
            logger.info("Order %d: %d states, %d transitions", 
                       order, tn.n_nodes, tn.n_edges)
        except Exception as e:
            logger.warning("Order %d failed: %s", order, str(e))


def plot_data(df):
    """Plot the economic time series using signalplot (if available)."""
    if not HAS_SIGNALPLOT and not HAS_MATPLOTLIB:
        logger.warning("No plotting library available. Skipping plot.")
        return
    
    try:
        if HAS_SIGNALPLOT:
            # Use signalplot for clean, minimalist plots
            import signalplot as splt
            import matplotlib.pyplot as plt
            
            # Apply signalplot's minimalist styling (style is a module with apply function)
            splt.apply()  # Apply signalplot defaults globally
            
            # Create subplots using matplotlib
            fig, axes = plt.subplots(len(df.columns), 1, figsize=(10, 3*len(df.columns)))
            if len(df.columns) == 1:
                axes = [axes]
            
            # Plot each series with signalplot styling
            for i, col in enumerate(df.columns):
                splt.style_line_plot(axes[i])  # Apply signalplot line plot styling to axis
                axes[i].plot(df.index, df[col].values)
                axes[i].set_title(f'{col} (FRED)')
                axes[i].set_ylabel('Value')
            
            axes[-1].set_xlabel('Date')
            plt.tight_layout()
            output_path = 'examples/images/fred_data.png'
            splt.savefig(output_path, dpi=150)
            plt.close()
        else:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(len(df.columns), 1, figsize=(12, 3*len(df.columns)))
            if len(df.columns) == 1:
                axes = [axes]
            
            for i, col in enumerate(df.columns):
                axes[i].plot(df.index, df[col].values, linewidth=1.5)
                axes[i].set_title(f'{col} (FRED)', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Date')
            plt.tight_layout()
            output_path = 'examples/images/fred_data.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info("Plot saved to: examples/images/fred_data.png")
    except Exception as e:
        logger.warning("Could not create plot: %s", str(e))


def plot_network(G, title, filename, layout='spring'):
    """Visualize a network graph with meaningful structure."""
    if not HAS_SIGNALPLOT and not HAS_MATPLOTLIB:
        return
    
    try:
        import matplotlib.pyplot as plt
        
        if HAS_SIGNALPLOT:
            import signalplot as splt
            splt.apply()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Choose layout based on network size
        n_nodes = G.number_of_nodes()
        if layout == 'spring':
            if n_nodes < 100:
                pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
            else:
                pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Node colors based on degree (more interesting than uniform)
        degrees = dict(G.degree())
        node_colors = [degrees.get(n, 0) for n in G.nodes()]
        
        # Node sizes based on degree (highlight important nodes)
        node_sizes = [300 + 200 * degrees.get(n, 0) / max(degrees.values()) if degrees else 300 
                     for n in G.nodes()]
        
        # Draw edges first (so nodes appear on top)
        if G.number_of_edges() < 500:  # Draw edges for reasonable-sized networks
            edge_alpha = 0.2 if n_nodes > 50 else 0.4
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                alpha=edge_alpha,
                width=0.3,
                edge_color='lightgray'
            )
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=plt.cm.plasma,
            alpha=0.9,
            linewidths=0.5,
            edgecolors='white'
        )
        
        # Draw labels only for small networks or high-degree nodes
        if n_nodes <= 30:
            nx.draw_networkx_labels(
                G, pos, ax=ax,
                font_size=8,
                font_weight='bold',
                font_color='black'
            )
        elif n_nodes <= 100:
            # Only label high-degree nodes
            high_degree_nodes = [n for n in G.nodes() if degrees.get(n, 0) > np.percentile(list(degrees.values()), 75)]
            labels = {n: str(n) for n in high_degree_nodes}
            nx.draw_networkx_labels(
                G, pos, labels, ax=ax,
                font_size=6,
                font_color='black'
            )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add colorbar
        if nodes is not None and max(node_colors) > min(node_colors):
            cbar = plt.colorbar(nodes, ax=ax, label='Node Degree', shrink=0.6, pad=0.02)
            cbar.ax.tick_params(labelsize=9)
        
        # Add network statistics as text
        stats_text = f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}"
        if n_nodes <= 100:
            clustering = nx.average_clustering(G)
            stats_text += f" | Clustering: {clustering:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Network diagram saved to: %s", filename)
    except Exception as e:
        logger.warning("   Could not create network plot: %s", str(e))


def main():
    """Main example function."""
    logger.info("ts2net: Real Economic Data Example (FRED)")
    
    # Fetch data
    df = fetch_fred_data(start_date='2010-01-01')
    
    if df is None or df.empty:
        logger.error("No data available. Exiting.")
        return
    
    # Plot the data
    plot_data(df)
    
    # Run examples
    example_univariate_analysis(df)
    example_proximity_network(df)
    example_multivariate_network(df)
    example_recurrence_network(df)
    example_transition_network(df)


if __name__ == "__main__":
    main()
