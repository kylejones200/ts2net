//! Python bindings for ts2net-rs

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::distance::{cdist_dtw, pearson};
use crate::embedding::{cao_e1_e2, false_nearest_neighbors};
use crate::graphs::{
    clustering_avg, mean_shortest_path, triangles_per_node, hvg_edges, nvg_edges_sweepline
};
use crate::utils::*;

/// Python module for ts2net-rs
#[pymodule]
fn ts2net_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Graph functions
    m.add_function(wrap_pyfunction!(hvg_edges, m)?)?;
    m.add_function(wrap_pyfunction!(nvg_edges_sweepline, m)?)?;
    m.add_function(wrap_pyfunction!(triangles_per_node, m)?)?;
    m.add_function(wrap_pyfunction!(clustering_avg, m)?)?;
    m.add_function(wrap_pyfunction!(mean_shortest_path, m)?)?;
    
    // Distance functions
    m.add_function(wrap_pyfunction!(cdist_dtw, m)?)?;
    
    // Embedding functions
    m.add_function(wrap_pyfunction!(false_nearest_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(cao_e1_e2, m)?)?;
    
    // Utility functions (if needed)
    // m.add_function(wrap_pyfunction!(some_utility_function, m)?)?;
    
    Ok(())
}
