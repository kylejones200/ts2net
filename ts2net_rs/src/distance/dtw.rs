//! Dynamic time warping distances.

use ndarray::{Array2, Axis};
use rayon::prelude::*;

/// Sakoe-Chiba banded DTW distance between two sequences.
///
/// `band` is the half-width of the warping window in samples; `None` computes
/// the unconstrained distance. A band narrower than `|len(a) - len(b)|` is
/// widened to that difference so the warping path can always reach the corner.
pub fn dtw_pair(a: &[f64], b: &[f64], band: Option<usize>) -> f64 {
    let n = a.len();
    let m = b.len();
    let inf = f64::INFINITY;
    let mut dp = vec![vec![inf; m + 1]; n + 1];
    dp[0][0] = 0.0;
    for i in 1..=n {
        // When no band is given use the full row (jmin=1, jmax=m).
        // Previously `band.unwrap_or(usize::MAX)` then `i + w` wrapped
        // to 0 in release mode, making jmax=0 and leaving every cell at inf.
        let (jmin, jmax) = match band {
            None => (1, m),
            Some(mut w) => {
                let dm = if n > m { n - m } else { m - n };
                if w < dm {
                    w = dm;
                }
                (1usize.max(i.saturating_sub(w)), m.min(i.saturating_add(w)))
            }
        };
        for j in jmin..=jmax {
            let cost = (a[i - 1] - b[j - 1]).powi(2);
            let v = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = cost + v;
        }
    }
    dp[n][m].sqrt()
}

/// Symmetric all-pairs DTW distance matrix over the rows of `x`.
pub fn cdist_dtw(x: &Array2<f64>, band: Option<usize>) -> Array2<f64> {
    let n = x.len_of(Axis(0));
    let mut out = Array2::<f64>::zeros((n, n));

    // Collect results in a thread-safe way
    let results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let ai = x.row(i).to_owned();
            let mut local_results = Vec::new();
            for j in (i + 1)..n {
                let d = dtw_pair(ai.as_slice().unwrap(), x.row(j).as_slice().unwrap(), band);
                local_results.push((i, j, d));
            }
            local_results
        })
        .collect();

    // Apply results to output matrix
    for (i, j, d) in results {
        out[[i, j]] = d;
        out[[j, i]] = d;
    }
    out
}

/// Rectangular DTW distance matrix between the rows of `a` and the rows of `b`.
pub fn cdist_dtw_rectangular(a: &Array2<f64>, b: &Array2<f64>, band: Option<usize>) -> Array2<f64> {
    let na = a.len_of(Axis(0));
    let nb = b.len_of(Axis(0));
    let mut out = Array2::<f64>::zeros((na, nb));
    let results: Vec<(usize, usize, f64)> = (0..na)
        .into_par_iter()
        .flat_map(|i| {
            let ai = a.row(i).to_owned();
            let mut local = Vec::new();
            for j in 0..nb {
                let d = dtw_pair(ai.as_slice().unwrap(), b.row(j).as_slice().unwrap(), band);
                local.push((i, j, d));
            }
            local
        })
        .collect();
    for (i, j, d) in results {
        out[[i, j]] = d;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn identical_sequences_have_zero_distance() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(dtw_pair(&a, &a, None), 0.0);
    }

    #[test]
    fn unbanded_matches_banded_when_band_covers_everything() {
        let a = [0.0, 1.0, 3.0, 2.0, 5.0];
        let b = [0.0, 2.0, 2.0, 4.0, 4.0];
        assert_relative_eq!(dtw_pair(&a, &b, None), dtw_pair(&a, &b, Some(a.len())));
    }

    #[test]
    fn cdist_is_symmetric_with_zero_diagonal() {
        let x = array![[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
        let d = cdist_dtw(&x, None);
        for i in 0..3 {
            assert_relative_eq!(d[[i, i]], 0.0);
            for j in 0..3 {
                assert_relative_eq!(d[[i, j]], d[[j, i]]);
            }
        }
    }

    #[test]
    fn rectangular_against_self_matches_square() {
        let x = array![[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]];
        let sq = cdist_dtw(&x, None);
        let rect = cdist_dtw_rectangular(&x, &x, None);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(sq[[i, j]], rect[[i, j]]);
            }
        }
    }
}
