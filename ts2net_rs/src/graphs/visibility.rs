use ndarray::{Array1, Array2, ArrayView1};
use std::cmp::Ordering;

/// Compute edges for a Horizontal Visibility Graph (HVG)
pub fn hvg_edges(y: &Array1<f64>) -> Array2<usize> {
    let n = y.len();
    let mut edges = Vec::with_capacity(2 * n);
    let mut stack = Vec::with_capacity(n);

    for j in 0..n {
        while let Some(&i) = stack.last() {
            if y[i] < y[j] {
                stack.pop();
                edges.push((i, j));
            } else {
                break;
            }
        }
        if let Some(&i) = stack.last() {
            edges.push((i, j));
        }
        stack.push(j);
    }

    // Convert to ndarray
    let m = edges.len();
    let mut result = Array2::zeros((m, 2));
    for (idx, (i, j)) in edges.into_iter().enumerate() {
        result[[idx, 0]] = i;
        result[[idx, 1]] = j;
    }
    result
}

/// Compute edges for a Natural Visibility Graph (NVG) using sweep line algorithm
pub fn nvg_edges_sweepline(y: &Array1<f64>) -> Array2<usize> {
    let n = y.len();
    let mut edges = Vec::with_capacity(2 * n);

    for i in 0..n - 1 {
        let yi = y[i];
        let mut slope_max = f64::NEG_INFINITY;
        
        for j in (i + 1)..n {
            let slope = (y[j] - yi) / ((j - i) as f64);
            if slope > slope_max {
                edges.push((i, j));
                slope_max = slope;
            }
        }
    }

    // Convert to ndarray
    let m = edges.len();
    let mut result = Array2::zeros((m, 2));
    for (idx, (i, j)) in edges.into_iter().enumerate() {
        result[[idx, 0]] = i;
        result[[idx, 1]] = j;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_hvg_edges() {
        let y = array![1.0, 2.0, 1.0, 2.0];
        let edges = hvg_edges(&y);
        assert_eq!(edges.shape(), &[3, 2]);
        assert_eq!(edges.row(0), array![0, 1].view());
        assert_eq!(edges.row(1), array![1, 2].view());
        assert_eq!(edges.row(2), array![2, 3].view());
    }

    #[test]
    fn test_nvg_edges() {
        let y = array![1.0, 2.0, 1.0, 2.0];
        let edges = nvg_edges_sweepline(&y);
        assert_eq!(edges.shape(), &[3, 2]);
        assert_eq!(edges.row(0), array![0, 1].view());
        assert_eq!(edges.row(1), array![1, 2].view());
        assert_eq!(edges.row(2), array![2, 3].view());
    }
}
