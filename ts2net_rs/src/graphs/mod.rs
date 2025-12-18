//! Graph-related functionality including visibility graphs and graph metrics

use ndarray::{s, Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod visibility;
pub use visibility::*;

/// Calculate triangles per node in a graph
#[pyfunction]
pub fn triangles_per_node(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<Py<PyArray1<usize>>> {
    let edges = edges.as_array();
    let mut triangles = vec![0; n];
    let adj = crate::utils::build_adj(n, edges.rows().into_iter().map(|r| (r[0], r[1])), true);
    
    for i in 0..n {
        let neighbors = &adj[i];
        for &j in neighbors {
            if j <= i { continue; } // Avoid double counting
            for &k in neighbors {
                if k <= j { continue; } // Avoid double counting
                if adj[j].binary_search(&k).is_ok() {
                    triangles[i] += 1;
                    triangles[j] += 1;
                    triangles[k] += 1;
                }
            }
        }
    }
    
    Ok(Array1::from(triangles).into_pyarray(py).to_owned())
}

/// Calculate average clustering coefficient
#[pyfunction]
pub fn clustering_avg(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<f64> {
    let edges = edges.as_array();
    let adj = crate::utils::build_adj(n, edges.rows().into_iter().map(|r| (r[0], r[1])), true);
    
    let mut total = 0.0;
    let mut count = 0;
    
    for i in 0..n {
        let neighbors = &adj[i];
        let k = neighbors.len();
        if k < 2 { continue; } // Need at least 2 neighbors for triangles
        
        let mut triangles = 0;
        for &j in neighbors {
            for &l in neighbors {
                if j >= l { continue; } // Avoid double counting
                if adj[j].binary_search(&l).is_ok() {
                    triangles += 1;
                }
            }
        }
        
        total += (2.0 * triangles as f64) / (k * (k - 1)) as f64;
        count += 1;
    }
    
    Ok(if count > 0 { total / count as f64 } else { 0.0 })
}

/// Calculate mean shortest path length
#[pyfunction]
pub fn mean_shortest_path(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<f64> {
    use std::collections::VecDeque;
    
    let edges = edges.as_array();
    let adj = crate::utils::build_adj(n, edges.rows().into_iter().map(|r| (r[0], r[1])), true);
    
    let mut total = 0.0;
    let mut count = 0;
    
    for start in 0..n {
        let mut dist = vec![usize::MAX; n];
        let mut queue = VecDeque::new();
        
        dist[start] = 0;
        queue.push_back(start);
        
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        
        for &d in &dist {
            if d != usize::MAX && d > 0 {
                total += d as f64;
                count += 1;
            }
        }
    }
    
    Ok(if count > 0 { total / count as f64 } else { 0.0 })
}
