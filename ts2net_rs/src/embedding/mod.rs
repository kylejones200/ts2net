//! Time series embedding and related functionality

use ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::VecDeque;

/// Compute false nearest neighbors for determining embedding dimension
#[pyfunction]
pub fn false_nearest_neighbors(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    m_max: usize,
    tau: usize,
    rtol: f64,
    atol: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x = x.as_array();
    let n = x.len();
    let mut fnn = Array1::<f64>::zeros(m_max);
    
    for m in 1..=m_max {
        let mut count = 0;
        let mut total = 0;
        
        for i in 0..n - (m + 1) * tau {
            let mut min_dist = f64::INFINITY;
            let mut min_j = 0;
            
            // Find nearest neighbor in m-dimensional space
            for j in 0..n - (m + 1) * tau {
                if i == j { continue; }
                
                let mut dist = 0.0;
                for k in 0..m {
                    let d = x[i + k * tau] - x[j + k * tau];
                    dist += d * d;
                }
                
                if dist < min_dist {
                    min_dist = dist;
                    min_j = j;
                }
            }
            
            // Compute distance in (m+1)-dimensional space
            let d_m = min_dist.sqrt();
            let d_m_plus_1 = (d_m * d_m + 
                (x[i + m * tau] - x[min_j + m * tau]).powi(2)).sqrt();
            
            // Check FNN criterion
            if d_m_plus_1 / d_m > 1.0 + rtol || d_m_plus_1 > atol {
                count += 1;
            }
            total += 1;
        }
        
        fnn[m-1] = if total > 0 { count as f64 / total as f64 } else { 0.0 };
    }
    
    Ok(fnn.into_pyarray(py).to_owned())
}

/// Compute E1 and E2 statistics for determining embedding dimension
#[pyfunction]
pub fn cao_e1_e2(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    m_max: usize,
    tau: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let x = x.as_array();
    let n = x.len();
    let mut e1 = Array1::<f64>::zeros(m_max);
    let mut e2 = Array1::<f64>::zeros(m_max);
    
    for m in 1..=m_max {
        let mut a1 = 0.0;
        let mut a2 = 0.0;
        
        for i in 0..n - (m + 1) * tau {
            let mut min_dist = f64::INFINITY;
            let mut min_j = 0;
            
            // Find nearest neighbor in m-dimensional space
            for j in 0..n - (m + 1) * tau {
                if i == j { continue; }
                
                let mut dist = 0.0;
                for k in 0..m {
                    let d = x[i + k * tau] - x[j + k * tau];
                    dist += d * d;
                }
                
                if dist < min_dist {
                    min_dist = dist;
                    min_j = j;
                }
            }
            
            // Compute distances in (m+1)-dimensional space
            let d_m = min_dist.sqrt();
            let d_m_plus_1 = (d_m * d_m + 
                (x[i + m * tau] - x[min_j + m * tau]).powi(2)).sqrt();
            
            a1 += d_m_plus_1 / d_m;
            a2 += (x[i + m * tau] - x[min_j + m * tau]).abs() / d_m;
        }
        
        let n_eff = (n - (m + 1) * tau) as f64;
        e1[m-1] = if n_eff > 0.0 { a1 / n_eff } else { 0.0 };
        e2[m-1] = if n_eff > 0.0 { a2 / n_eff } else { 0.0 };
    }
    
    let py_e1 = e1.into_pyarray(py).to_owned();
    let py_e2 = e2.into_pyarray(py).to_owned();
    
    Ok((py_e1, py_e2))
}

/// Generate time-delay embedding of a time series
pub fn time_delay_embedding(ts: &[f64], m: usize, tau: usize) -> Array2<f64> {
    let n = ts.len().saturating_sub((m - 1) * tau);
    let mut embedded = Array2::<f64>::zeros((n, m));
    
    for i in 0..n {
        for j in 0..m {
            embedded[[i, j]] = ts[i + j * tau];
        }
    }
    
    embedded
}
