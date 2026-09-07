//! Recurrence networks.

use ndarray::{Array2, Axis};

use crate::distance::{pair_dist, Metric};

/// Epsilon-recurrence adjacency matrix over the rows of `x`.
///
/// Entry `[i, j]` is 1 when the two embedded points are within `eps` under
/// `metric`. `theiler` excludes pairs whose index separation is at most that
/// many samples, suppressing the trivial diagonal correlation.
pub fn rn_adj_epsilon(x: &Array2<f64>, eps: f64, metric: Metric, theiler: usize) -> Array2<u8> {
    let n = x.len_of(Axis(0));
    let mut adj = Array2::<u8>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            if theiler > 0 && j.saturating_sub(i) <= theiler {
                continue;
            }
            let d = pair_dist(x.row(i), x.row(j), metric);
            if d <= eps {
                adj[[i, j]] = 1;
                adj[[j, i]] = 1;
            }
        }
    }
    adj
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn close_points_recur_and_far_points_do_not() {
        let x = array![[0.0], [0.1], [5.0]];
        let a = rn_adj_epsilon(&x, 0.5, Metric::Euclidean, 0);
        assert_eq!(a[[0, 1]], 1);
        assert_eq!(a[[1, 0]], 1);
        assert_eq!(a[[0, 2]], 0);
        // The diagonal is never set.
        assert_eq!(a[[0, 0]], 0);
    }

    #[test]
    fn the_theiler_window_suppresses_near_diagonal_pairs() {
        let x = array![[0.0], [0.1], [0.2]];
        let a = rn_adj_epsilon(&x, 1.0, Metric::Euclidean, 1);
        assert_eq!(a[[0, 1]], 0);
        assert_eq!(a[[1, 2]], 0);
        assert_eq!(a[[0, 2]], 1);
    }
}
