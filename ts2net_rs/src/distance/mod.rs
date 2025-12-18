//! Distance metrics and related functionality

use ndarray::{s, Array1, Array2};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute DTW distance between two time series with optional Sakoe-Chiba band
pub fn dtw_pair(a: &[f64], b: &[f64], band: Option<usize>) -> f64 {
    let n = a.len();
    let m = b.len();
    
    let band = band.unwrap_or_else(|| n.max(m));
    let band = band.max((n as isize - m as isize).unsigned_abs());
    
    // Initialize DP table with infinity
    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    
    for i in 1..=n {
        let lower = 1.max(i.saturating_sub(band));
        let upper = m.min(i + band);
        
        for j in lower..=upper {
            let cost = (a[i - 1] - b[j - 1]).powi(2);
            dp[i][j] = cost + dp[i-1][j-1].min(dp[i-1][j].min(dp[i][j-1]));
        }
    }
    
    dp[n][m].sqrt()
}

/// Compute pairwise DTW distances between time series
#[pyfunction]
pub fn cdist_dtw(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    band: Option<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    let x = x.as_array();
    let n = x.nrows();
    let mut dist = Array2::<f64>::zeros((n, n));
    
    for i in 0..n {
        for j in i+1..n {
            let d = dtw_pair(
                x.row(i).to_slice().unwrap(),
                x.row(j).to_slice().unwrap(),
                band
            );
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }
    
    Ok(dist.into_pyarray(py).to_owned())
}

/// Compute Pearson correlation coefficient
pub fn pearson(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);
    
    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
    
    for (&xi, &yi) in x.iter().zip(y) {
        let x_diff = xi - x_mean;
        let y_diff = yi - y_mean;
        xy += x_diff * y_diff;
        xx += x_diff * x_diff;
        yy += y_diff * y_diff;
    }
    
    xy / (xx * yy).sqrt()
}
