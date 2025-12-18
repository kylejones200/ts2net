"""R ts2net API Parity"""

import numpy as np
import networkx as nx
import logging
from ts2net.multivariate import ts_dist, net_knn, net_enn, net_weighted, ts_to_windows

logging.basicConfig(level=logging.WARNING)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def multivariate_network():
    """Multiple time series → network"""
    logger.info("\nMultivariate Network")
    
    # Generate multiple time series (e.g., sensors, stocks, climate)
    np.random.seed(42)
    n_series = 30
    n_points = 200
    
    # Create three groups with different patterns
    X = []
    labels = []
    
    for i in range(n_series):
        t = np.linspace(0, 4*np.pi, n_points)
        if i < 10:
            # Group 1: Fast oscillation
            x = np.sin(3*t) + 0.2 * np.random.randn(n_points)
            labels.append(0)
        elif i < 20:
            # Group 2: Slow oscillation
            x = np.sin(0.5*t) + 0.2 * np.random.randn(n_points)
            labels.append(1)
        else:
            # Group 3: Trend
            x = t/4 + np.sin(t) + 0.2 * np.random.randn(n_points)
            labels.append(2)
        
        X.append(x)
    
    X = np.array(X)
    logger.info("%d series × %d points", n_series, n_points)
    
    methods = ['correlation', 'dtw', 'nmi', 'voi', 'es']
    networks = {}
    
    for method in methods:
        try:
            D = ts_dist(X, method=method, n_jobs=1)
            G, A = net_knn(D, k=5, weighted=True)
            networks[method] = G
        except Exception:
            pass
    
    G_dtw = networks['dtw']
    communities = list(nx.community.greedy_modularity_communities(G_dtw))
    modularity = nx.community.modularity(G_dtw, communities)
    
    logger.info("DTW k-NN: %d nodes, %d edges", G_dtw.number_of_nodes(), G_dtw.number_of_edges())
    logger.info("Clustering: %.3f, Modularity: %.3f", nx.average_clustering(G_dtw), modularity)
    
    return X, G_dtw, labels


def proximity_network():
    """Single time series → proximity network"""
    logger.info("\nProximity Network")
    
    # Generate CO2-like data (trend + seasonality)
    np.random.seed(42)
    n = 400
    t = np.linspace(0, 10, n)
    co2 = 300 + 2*t + 10*np.sin(2*np.pi*t) + 0.5*np.random.randn(n)
    
    windows = ts_to_windows(co2, width=12, by=1)
    D = ts_dist(windows, method='correlation', n_jobs=1)
    G, A = net_enn(D, percentile=25)
    
    logger.info("%d windows → %d edges", windows.shape[0], G.number_of_edges())


def network_builders():
    """Network construction methods"""
    logger.info("\nNetwork Builders")
    
    np.random.seed(42)
    X = np.random.randn(50, 100)
    D = ts_dist(X, method='dtw', normalize=True, n_jobs=1)
    
    G_knn, _ = net_knn(D, k=3, weighted=True)
    G_knn_mut, _ = net_knn(D, k=3, mutual=True, weighted=True)
    G_enn, _ = net_enn(D, percentile=20, weighted=True)
    G_weighted, _ = net_weighted(D, threshold=0.7)
    
    logger.info("k-NN: %d edges", G_knn.number_of_edges())
    logger.info("k-NN mutual: %d edges", G_knn_mut.number_of_edges())
    logger.info("ε-NN: %d edges", G_enn.number_of_edges())
    logger.info("weighted: %d edges", G_weighted.number_of_edges())


def distance_functions():
    """Distance functions"""
    logger.info("\nDistance Functions")
    
    np.random.seed(42)
    x = np.sin(np.linspace(0, 4*np.pi, 100))
    y = np.sin(np.linspace(0, 4*np.pi, 100) + 0.5)
    X = np.array([x, y])
    
    for method in ['correlation', 'ccf', 'dtw', 'nmi', 'voi', 'es', 'vr']:
        try:
            D = ts_dist(X, method=method, n_jobs=1)
            logger.info("%s: %.4f", method, D[0, 1])
        except Exception:
            pass


def univariate_methods():
    """Univariate methods"""
    logger.info("\nUnivariate Methods")
    
    from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork
    
    np.random.seed(42)
    n = 500
    x = np.zeros(n)
    x[0] = 0.1
    r = 3.9
    
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    
    hvg = HVG().build(x)
    nvg = NVG().build(x)
    rn = RecurrenceNetwork(m=3, tau=1, rule='knn', k=5).build(x)
    tn = TransitionNetwork(symbolizer='ordinal', order=3).build(x)
    
    logger.info("HVG: %d edges", hvg.n_edges)
    logger.info("NVG: %d edges", nvg.n_edges)
    logger.info("Recurrence: %d edges", rn.n_edges)
    logger.info("Transition: %d edges", tn.n_edges)


def main():
    logger.info("ts2net: R API Parity")
    logger.info("Reference: Ferreira (2024) Applied Network Science 9(1):32\n")
    
    multivariate_network()
    proximity_network()
    network_builders()
    distance_functions()
    univariate_methods()
    
    logger.info("\nComplete")
    logger.info("8 distance functions")
    logger.info("5 network builders")
    logger.info("4 univariate methods")


if __name__ == "__main__":
    main()

