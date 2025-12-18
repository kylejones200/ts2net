//! Utility functions and helpers

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Convert a 1D numpy array to an ndarray
#[inline]
pub fn as_1d<'a>(y: PyReadonlyArray1<'a, f64>) -> PyResult<Array1<f64>> {
    let a = y.as_array();
    if a.ndim() != 1 {
        return Err(PyValueError::new_err("expected 1-D"));
    }
    Ok(a.to_owned())
}

/// Convert a 2D numpy array to an ndarray
#[inline]
pub fn as_2d<'a>(x: PyReadonlyArray2<'a, f64>) -> PyResult<Array2<f64>> {
    let a = x.as_array();
    if a.ndim() != 2 {
        return Err(PyValueError::new_err("expected 2-D"));
    }
    Ok(a.to_owned())
}

/// Build an adjacency list from edge indices
pub fn build_adj(n: usize, edges: &[(usize, usize)], undirected: bool) -> Vec<Vec<usize>> {
    let mut adj = vec![vec![]; n];
    for &(i, j) in edges {
        adj[i].push(j);
        if undirected {
            adj[j].push(i);
        }
    }
    adj
}

/// Count the number of common elements between two sorted slices
pub fn count_intersection(a: &[usize], b: &[usize]) -> usize {
    let mut count = 0;
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}
