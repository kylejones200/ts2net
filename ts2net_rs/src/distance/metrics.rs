//! Point-to-point metrics and correlation.

use ndarray::{Array1, ArrayView1};

/// Metric used for point-to-point distances in embedded (phase) space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Metric {
    /// L2 / straight-line distance. The fallback for unrecognised names.
    #[default]
    Euclidean,
    /// L1 / taxicab distance.
    Manhattan,
    /// L-infinity / maximum-coordinate distance.
    Chebyshev,
}

impl Metric {
    /// Resolve a metric name, falling back to [`Metric::Euclidean`].
    ///
    /// The fallback (rather than an error) is deliberate: it preserves the
    /// behaviour the Python bindings have always had for unknown names.
    pub fn from_name(name: &str) -> Self {
        match name {
            "manhattan" => Metric::Manhattan,
            "chebyshev" => Metric::Chebyshev,
            _ => Metric::Euclidean,
        }
    }
}

/// Distance between two equal-length points under `metric`.
#[inline]
pub fn pair_dist(a: ArrayView1<f64>, b: ArrayView1<f64>, metric: Metric) -> f64 {
    match metric {
        Metric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        Metric::Chebyshev => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max),
        Metric::Euclidean => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt(),
    }
}

/// Pearson correlation coefficient. Returns `0.0` if either input is constant.
pub fn pearson(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len();
    if n == 0 {
        // `mean()` returns None on an empty array; unwrapping it panicked.
        return 0.0;
    }
    let mx = x.mean().unwrap();
    let my = y.mean().unwrap();
    let mut num = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for i in 0..n {
        let a = x[i] - mx;
        let b = y[i] - my;
        num += a * b;
        sx += a * a;
        sy += b * b;
    }
    if sx == 0.0 || sy == 0.0 {
        0.0
    } else {
        num / (sx.sqrt() * sy.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn metric_names_fall_back_to_euclidean() {
        assert_eq!(Metric::from_name("manhattan"), Metric::Manhattan);
        assert_eq!(Metric::from_name("chebyshev"), Metric::Chebyshev);
        assert_eq!(Metric::from_name("euclidean"), Metric::Euclidean);
        assert_eq!(Metric::from_name("nonsense"), Metric::Euclidean);
    }

    #[test]
    fn metrics_agree_with_hand_computation() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_relative_eq!(pair_dist(a.view(), b.view(), Metric::Euclidean), 5.0);
        assert_relative_eq!(pair_dist(a.view(), b.view(), Metric::Manhattan), 7.0);
        assert_relative_eq!(pair_dist(a.view(), b.view(), Metric::Chebyshev), 4.0);
    }

    #[test]
    fn pearson_is_one_for_a_perfect_line_and_zero_when_constant() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 4.0, 6.0, 8.0];
        assert_relative_eq!(pearson(&x, &y), 1.0, epsilon = 1e-12);
        let c = array![1.0, 1.0, 1.0, 1.0];
        assert_relative_eq!(pearson(&x, &c), 0.0);
    }
}
