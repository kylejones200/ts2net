//! Horizontal and natural visibility graphs.
//!
//! Both builders come in two forms: an edge-list form, and a degree form that
//! never materialises the edge list and so stays O(n) in memory on long series.

use ndarray::{s, Array1, Array2};

/// True when `i` and `j` are within `limit` samples of each other.
///
/// `None` means no horizon, so every pair qualifies.
#[inline]
pub fn within_horizon(i: usize, j: usize, limit: Option<usize>) -> bool {
    match limit {
        Some(l) => j.abs_diff(i) <= l,
        None => true,
    }
}

/// Edge list of the horizontal visibility graph of `y`, shaped `[m, 2]`.
///
/// Two samples are connected when every sample strictly between them lies
/// below both. Runs in O(n) via a monotonic stack.
pub fn hvg_edges(y: &Array1<f64>) -> Array2<i64> {
    let n = y.len();
    let mut ei = Vec::<i64>::with_capacity(2 * n);
    let mut ej = Vec::<i64>::with_capacity(2 * n);
    let mut stack: Vec<usize> = Vec::with_capacity(n);
    for j in 0..n {
        while let Some(&i) = stack.last() {
            if y[i] < y[j] {
                stack.pop();
                ei.push(i as i64);
                ej.push(j as i64);
            } else {
                break;
            }
        }
        if let Some(&i) = stack.last() {
            ei.push(i as i64);
            ej.push(j as i64);
        }
        stack.push(j);
    }
    edges_to_array(ei, ej)
}

/// Edge list of the natural visibility graph of `y`, shaped `[m, 2]`.
///
/// Two samples are connected when the straight line between them passes above
/// every sample in between. Runs in O(n^2) worst case via a forward sweep that
/// tracks the running maximum slope.
pub fn nvg_edges_sweepline(y: &Array1<f64>) -> Array2<i64> {
    let n = y.len();
    let mut ei = Vec::<i64>::with_capacity(2 * n);
    let mut ej = Vec::<i64>::with_capacity(2 * n);
    // Guard n < 2: `0..n - 1` underflows for an empty series.
    if n < 2 {
        return edges_to_array(ei, ej);
    }
    for i in 0..n - 1 {
        let yi = y[i];
        let mut slope_max = f64::NEG_INFINITY;
        for j in i + 1..n {
            let s = (y[j] - yi) / ((j - i) as f64);
            if s > slope_max {
                ei.push(i as i64);
                ej.push(j as i64);
                slope_max = s;
            }
        }
    }
    edges_to_array(ei, ej)
}

#[inline]
fn edges_to_array(ei: Vec<i64>, ej: Vec<i64>) -> Array2<i64> {
    let m = ei.len();
    let mut out = Array2::<i64>::zeros((m, 2));
    out.slice_mut(s![.., 0]).assign(&Array1::from(ei));
    out.slice_mut(s![.., 1]).assign(&Array1::from(ej));
    out
}

/// HVG degree sequences and edge count without materialising the edge list.
///
/// Returns `(in_degree, out_degree, n_edges)`. When `directed` is false the
/// undirected degree is accumulated into the second element and the first is
/// left at zero. `limit` optionally restricts edges to a horizon in samples.
pub fn hvg_degrees(
    y: &Array1<f64>,
    directed: bool,
    limit: Option<usize>,
) -> (Array1<usize>, Array1<usize>, usize) {
    let n = y.len();
    let mut in_deg = Array1::<usize>::zeros(n);
    let mut out_deg = Array1::<usize>::zeros(n);
    let mut n_edges = 0usize;
    let mut stack: Vec<usize> = Vec::with_capacity(n);

    let mut add_edge = |i: usize, j: usize| {
        if directed {
            out_deg[i] += 1;
            in_deg[j] += 1;
        } else {
            out_deg[i] += 1;
            out_deg[j] += 1;
        }
        n_edges += 1;
    };

    for j in 0..n {
        while let Some(i) = stack.last().copied() {
            if y[i] < y[j] {
                stack.pop();
                if within_horizon(i, j, limit) {
                    add_edge(i, j);
                }
            } else {
                break;
            }
        }
        if let Some(i) = stack.last().copied() {
            if within_horizon(i, j, limit) {
                add_edge(i, j);
            }
        }
        stack.push(j);
    }
    (in_deg, out_deg, n_edges)
}

/// NVG degrees and edge count with an optional horizon limit (undirected).
pub fn nvg_degrees(y: &Array1<f64>, limit: Option<usize>) -> (Array1<usize>, usize) {
    let n = y.len();
    let mut degrees = Array1::<usize>::zeros(n);
    let mut n_edges = 0usize;
    if n < 2 {
        return (degrees, n_edges);
    }
    for i in 0..n - 1 {
        let yi = y[i];
        let mut slope_max = f64::NEG_INFINITY;
        let j_end = match limit {
            Some(l) => (i + 1 + l).min(n),
            None => n,
        };
        for j in (i + 1)..j_end {
            let s = (y[j] - yi) / ((j - i) as f64);
            if s > slope_max {
                degrees[i] += 1;
                degrees[j] += 1;
                n_edges += 1;
                slope_max = s;
            }
        }
    }
    (degrees, n_edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn pairs(edges: &Array2<i64>) -> Vec<(i64, i64)> {
        (0..edges.nrows())
            .map(|r| (edges[[r, 0]], edges[[r, 1]]))
            .collect()
    }

    #[test]
    fn hvg_on_a_zigzag() {
        // For [1, 2, 1, 2] the sample at index 2 (y=1) sits below both peaks,
        // so 1 and 3 see each other horizontally as well.
        let y = array![1.0, 2.0, 1.0, 2.0];
        let mut got = pairs(&hvg_edges(&y));
        got.sort_unstable();
        assert_eq!(got, vec![(0, 1), (1, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn nvg_on_a_zigzag() {
        let y = array![1.0, 2.0, 1.0, 2.0];
        let mut got = pairs(&nvg_edges_sweepline(&y));
        got.sort_unstable();
        assert_eq!(got, vec![(0, 1), (1, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn a_monotone_series_is_a_path_under_hvg_and_complete_under_nvg() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mut hvg = pairs(&hvg_edges(&y));
        hvg.sort_unstable();
        assert_eq!(hvg, vec![(0, 1), (1, 2), (2, 3)]);

        // A convex increasing ramp leaves every later point visible.
        let convex = array![1.0, 2.0, 4.0, 8.0];
        let mut nvg = pairs(&nvg_edges_sweepline(&convex));
        nvg.sort_unstable();
        assert_eq!(nvg, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn short_series_do_not_panic() {
        for y in [array![], array![1.0]] {
            assert_eq!(hvg_edges(&y).nrows(), 0);
            assert_eq!(nvg_edges_sweepline(&y).nrows(), 0);
            assert_eq!(nvg_degrees(&y, None).1, 0);
        }
    }

    #[test]
    fn degrees_agree_with_the_edge_list() {
        let y = array![1.0, 2.0, 1.0, 2.0, 0.5, 3.0];
        let edges = hvg_edges(&y);
        let (in_deg, out_deg, n_edges) = hvg_degrees(&y, false, None);
        assert_eq!(n_edges, edges.nrows());
        let mut expected = vec![0usize; y.len()];
        for (u, v) in pairs(&edges) {
            expected[u as usize] += 1;
            expected[v as usize] += 1;
        }
        assert_eq!(out_deg.to_vec(), expected);
        assert!(in_deg.iter().all(|&d| d == 0));

        let nvg = nvg_edges_sweepline(&y);
        let (deg, n) = nvg_degrees(&y, None);
        assert_eq!(n, nvg.nrows());
        let mut expected = vec![0usize; y.len()];
        for (u, v) in pairs(&nvg) {
            expected[u as usize] += 1;
            expected[v as usize] += 1;
        }
        assert_eq!(deg.to_vec(), expected);
    }

    #[test]
    fn directed_hvg_splits_in_and_out_degree() {
        let y = array![1.0, 2.0, 1.0, 2.0];
        let (in_deg, out_deg, n_edges) = hvg_degrees(&y, true, None);
        assert_eq!(n_edges, 4);
        assert_eq!(in_deg.sum(), 4);
        assert_eq!(out_deg.sum(), 4);
    }

    #[test]
    fn a_horizon_drops_long_range_edges() {
        let y = array![1.0, 2.0, 1.0, 2.0];
        let (_, _, unlimited) = hvg_degrees(&y, false, None);
        let (_, _, limited) = hvg_degrees(&y, false, Some(1));
        assert_eq!(unlimited, 4);
        // The (1, 3) edge spans two samples and is excluded.
        assert_eq!(limited, 3);
    }
}
