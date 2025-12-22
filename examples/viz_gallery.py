"""
Visualization Gallery: Five Flagship Figures

This example demonstrates all five flagship visualization functions
on the same dataset, showing clean, scalable plots for time series
network analysis.
"""

import sys
import os
# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork
    from ts2net.viz import (
        plot_series_with_events,
        plot_degree_profile,
        plot_degree_ccdf,
        plot_method_comparison,
        plot_window_feature_map,
        plot_hvg_small,
    )
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Make sure ts2net is installed: pip install ts2net")
    sys.exit(1)

# Create output directory
_images_dir = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(_images_dir, exist_ok=True)


def generate_sample_data(n=500):
    """Generate sample time series with change points."""
    np.random.seed(42)
    
    # Create series with three regimes
    x1 = np.random.randn(n // 3) * 0.5 + 2.0
    x2 = np.random.randn(n // 3) * 1.5 + 0.0
    x3 = np.random.randn(n - 2 * (n // 3)) * 0.8 + 1.0
    
    x = np.concatenate([x1, x2, x3])
    
    # Add trend
    t = np.arange(len(x))
    x = x + 0.01 * t
    
    # Change points
    events = [n // 3, 2 * n // 3]
    
    return x, events


def main():
    """Generate all five flagship figures."""
    print("=" * 60)
    print("ts2net Visualization Gallery")
    print("=" * 60)
    print()
    
    # Generate data
    x, events = generate_sample_data(n=500)
    print(f"Generated time series with {len(x)} points")
    print(f"Change points at indices: {events}")
    print()
    
    # ============================================================
    # Figure 1: Time series with change points and window boundaries
    # ============================================================
    print("📊 Figure 1: Time series with change points...")
    
    # Create window boundaries (sliding windows)
    window_size = 50
    window_bounds = [(i, i + window_size) 
                     for i in range(0, len(x) - window_size, 25)]
    
    fig1, ax1 = plot_series_with_events(
        x, 
        events=np.array(events),
        window_bounds=window_bounds[:10],  # Show first 10 windows
        title="Time Series with Change Points and Window Boundaries"
    )
    
    output_path1 = os.path.join(_images_dir, 'viz_figure1_series_with_events.png')
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"   Saved: {output_path1}")
    print()
    
    # ============================================================
    # Figure 2: Degree profile across time
    # ============================================================
    print("📊 Figure 2: Degree profile...")
    
    # Build HVG and get degrees
    hvg = HVG()
    g = hvg.build(x)
    degrees = g.degree_sequence()
    
    fig2, ax2 = plot_degree_profile(
        degrees,
        title="HVG Degree Profile (Local Complexity Proxy)"
    )
    
    output_path2 = os.path.join(_images_dir, 'viz_figure2_degree_profile.png')
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"   Saved: {output_path2}")
    print()
    
    # ============================================================
    # Figure 3: Degree distribution as CCDF
    # ============================================================
    print("📊 Figure 3: Degree CCDF...")
    
    fig3, ax3 = plot_degree_ccdf(
        degrees,
        title="HVG Degree Distribution (CCDF)"
    )
    
    output_path3 = os.path.join(_images_dir, 'viz_figure3_degree_ccdf.png')
    fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"   Saved: {output_path3}")
    print()
    
    # ============================================================
    # Figure 4: Method comparison panel
    # ============================================================
    print("📊 Figure 4: Method comparison...")
    
    # Build networks with different methods
    methods_data = {}
    
    # HVG
    hvg = HVG()
    g_hvg = hvg.build(x)
    stats_hvg = g_hvg.stats()
    methods_data['HVG'] = {
        'n_nodes': stats_hvg['n_nodes'],
        'n_edges': stats_hvg['n_edges'],
        'avg_degree': stats_hvg['avg_degree'],
        'density': stats_hvg['density']
    }
    
    # NVG
    nvg = NVG()
    g_nvg = nvg.build(x)
    stats_nvg = g_nvg.stats()
    methods_data['NVG'] = {
        'n_nodes': stats_nvg['n_nodes'],
        'n_edges': stats_nvg['n_edges'],
        'avg_degree': stats_nvg['avg_degree'],
        'density': stats_nvg['density']
    }
    
    # Recurrence Network
    rn = RecurrenceNetwork(m=3, rule='knn', k=10)
    g_rn = rn.build(x)
    stats_rn = g_rn.stats()
    methods_data['Recurrence'] = {
        'n_nodes': stats_rn['n_nodes'],
        'n_edges': stats_rn['n_edges'],
        'avg_degree': stats_rn['avg_degree'],
        'density': stats_rn['density']
    }
    
    # Transition Network
    tn = TransitionNetwork(symbolizer='ordinal', order=3)
    g_tn = tn.build(x)
    # Transition networks may have different node structure, use stats
    stats_tn = g_tn.stats()
    methods_data['Transition'] = {
        'n_nodes': stats_tn['n_nodes'],
        'n_edges': stats_tn['n_edges'],
        'avg_degree': stats_tn['avg_degree'],
        'density': stats_tn['density']
    }
    
    fig4, axes4 = plot_method_comparison(methods_data)
    
    output_path4 = os.path.join(_images_dir, 'viz_figure4_method_comparison.png')
    fig4.savefig(output_path4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"   Saved: {output_path4}")
    print()
    
    # ============================================================
    # Figure 5: Window level feature map
    # ============================================================
    print("📊 Figure 5: Window feature map...")
    
    # Compute window-level features
    window_size = 50
    step = 25
    window_features = {}
    
    deg_mean_list = []
    deg_std_list = []
    edge_count_list = []
    
    for i in range(0, len(x) - window_size, step):
        window = x[i:i + window_size]
        
        # Build HVG for this window
        hvg_window = HVG()
        g_window = hvg_window.build(window)
        deg_window = g_window.degree_sequence()
        
        deg_mean_list.append(np.mean(deg_window))
        deg_std_list.append(np.std(deg_window))
        edge_count_list.append(g_window.n_edges)
    
    window_features = {
        'Mean Degree': np.array(deg_mean_list),
        'Degree Std': np.array(deg_std_list),
        'Edge Count': np.array(edge_count_list),
    }
    
    time_labels = [f'T{i*step}' for i in range(len(deg_mean_list))]
    
    fig5, ax5 = plot_window_feature_map(
        window_features,
        time_labels=time_labels,
    )
    ax5.set_title("Window-Level Feature Map", fontfamily='DejaVu Sans', fontsize=13, pad=10)
    
    output_path5 = os.path.join(_images_dir, 'viz_figure5_window_feature_map.png')
    fig5.savefig(output_path5, dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"   Saved: {output_path5}")
    print()
    
    # ============================================================
    # Bonus: Small n graph drawing (for small series)
    # ============================================================
    print("📊 Bonus: Small n graph drawing...")
    
    # Use a small subset
    x_small = x[:100]
    hvg_small = HVG()
    g_small = hvg_small.build(x_small)
    
    if g_small.edges is not None:
        fig6, ax6 = plot_hvg_small(
            x_small,
            g_small.edges[:200],  # Limit edges for clarity
            title="HVG Graph Layout (Small Series)"
        )
        
        output_path6 = os.path.join(_images_dir, 'viz_bonus_hvg_small.png')
        fig6.savefig(output_path6, dpi=150, bbox_inches='tight')
        plt.close(fig6)
        print(f"   Saved: {output_path6}")
    else:
        print("   Skipped (edges not available)")
    print()
    
    print("=" * 60)
    print("✅ Gallery complete!")
    print(f"All figures saved to: {_images_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
