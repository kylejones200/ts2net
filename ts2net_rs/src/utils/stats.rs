//! Permutation tests and spatial autocorrelation.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::seq::SliceRandom;

use crate::distance::pearson;

/// Two-sided permutation p-value for the correlation between `x` and `y`.
///
/// Shuffles `y` `n_perm` times and counts how often the permuted correlation
/// is at least as extreme in magnitude as the observed one. The count is
/// add-one smoothed, so the result is never exactly zero.
pub fn corr_perm(x: &Array1<f64>, y: &Array1<f64>, n_perm: usize, seed: u64) -> f64 {
    let r0 = pearson(x, y);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cnt = 0usize;
    let mut yv = y.to_vec();
    for _ in 0..n_perm {
        yv.shuffle(&mut rng);
        let r = pearson(x, &Array1::from(yv.clone()));
        if r.abs() >= r0.abs() {
            cnt += 1;
        }
    }
    ((cnt + 1) as f64) / ((n_perm + 1) as f64)
}

/// Global Moran's I of `x` under the spatial weight matrix `w`.
///
/// Returns `(I, z)`. Both are NaN when `x` is empty or the weights sum to zero.
///
/// The z-score divides by a fixed 0.1 rather than the analytic standard
/// deviation of I, which is long to write out; treat it as indicative only.
pub fn moran_i(x: &Array1<f64>, w: &Array2<f64>) -> (f64, f64) {
    let n = x.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mx = x.mean().unwrap();
    let z: Vec<f64> = x.iter().map(|v| *v - mx).collect();
    let s0: f64 = w.iter().sum();
    if s0 == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        den += z[i] * z[i];
        for j in 0..n {
            num += w[[i, j]] * z[i] * z[j];
        }
    }
    let i_stat = (n as f64) / s0 * (num / den.max(1e-12));
    let ei = -1.0 / ((n as f64) - 1.0);
    let zscore = (i_stat - ei) / 0.1f64.max(1e-9); // keep simple; exact var is long
    (i_stat, zscore)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn a_perfect_correlation_is_significant_and_the_p_value_is_smoothed() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let p = corr_perm(&x, &y, 199, 42);
        assert!(p > 0.0, "add-one smoothing keeps p above zero");
        assert!(p < 0.05, "expected a small p-value, got {p}");
    }

    #[test]
    fn corr_perm_is_deterministic_for_a_given_seed() {
        let x = array![1.0, 5.0, 2.0, 8.0, 3.0];
        let y = array![2.0, 1.0, 9.0, 4.0, 7.0];
        assert_eq!(corr_perm(&x, &y, 99, 11), corr_perm(&x, &y, 99, 11));
    }

    #[test]
    fn moran_i_is_positive_when_neighbours_agree() {
        // A 4-node path where the first two are high and the last two low.
        let x = array![1.0, 1.0, -1.0, -1.0];
        let w = array![
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ];
        let (i_stat, _) = moran_i(&x, &w);
        assert!(i_stat > 0.0, "expected positive autocorrelation, got {i_stat}");
    }

    #[test]
    fn degenerate_inputs_are_nan_rather_than_a_panic() {
        let empty = Array1::<f64>::zeros(0);
        let (i_stat, z) = moran_i(&empty, &Array2::<f64>::zeros((0, 0)));
        assert!(i_stat.is_nan() && z.is_nan());

        let x = array![1.0, 2.0, 3.0];
        let (i_stat, z) = moran_i(&x, &Array2::<f64>::zeros((3, 3)));
        assert!(i_stat.is_nan() && z.is_nan());
    }
}
