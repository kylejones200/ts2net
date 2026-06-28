//! Polynomial feature library matching PySINDy / sklearn defaults.

use numpy::ndarray::{Array2, Axis};

fn combination_counts(degree: usize, n_features: usize, include_bias: bool) -> Vec<Vec<usize>> {
    let mut powers = Vec::new();
    if include_bias {
        powers.push(vec![0; n_features]);
    }
    for d in 1..=degree {
        generate_combinations(n_features, &mut vec![0; n_features], 0, d, &mut powers);
    }
    powers
}

fn generate_combinations(
    n_features: usize,
    current: &mut [usize],
    feat: usize,
    remaining: usize,
    out: &mut Vec<Vec<usize>>,
) {
    if remaining == 0 {
        out.push(current.to_vec());
        return;
    }
    for f in feat..n_features {
        current[f] += 1;
        generate_combinations(n_features, current, f, remaining - 1, out);
        current[f] -= 1;
    }
}

fn feature_name(powers: &[usize], state_names: &[String]) -> String {
    let mut parts = Vec::new();
    for (i, &p) in powers.iter().enumerate() {
        if p == 0 {
            continue;
        }
        if p == 1 {
            parts.push(state_names[i].clone());
        } else {
            parts.push(format!("{}^{}", state_names[i], p));
        }
    }
    if parts.is_empty() {
        "1".to_string()
    } else {
        parts.join(" ")
    }
}

/// Build Θ(X) and feature names (include_bias + full interactions up to ``degree``).
pub fn polynomial_library(
    x: &Array2<f64>,
    degree: usize,
    state_names: &[String],
) -> (Array2<f64>, Vec<String>) {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let powers = combination_counts(degree, n_features, true);
    let n_out = powers.len();
    let mut theta = Array2::<f64>::zeros((n_samples, n_out));
    let mut names = Vec::with_capacity(n_out);

    for (j, pow) in powers.iter().enumerate() {
        names.push(feature_name(pow, state_names));
        for i in 0..n_samples {
            let mut val = 1.0;
            for (f, &p) in pow.iter().enumerate() {
                if p > 0 {
                    val *= x[[i, f]].powi(p as i32);
                }
            }
            theta[[i, j]] = val;
        }
    }
    (theta, names)
}

/// Stack trajectories vertically before building the library.
pub fn polynomial_library_stack(
    trajectories: &[Array2<f64>],
    degree: usize,
    state_names: &[String],
) -> (Array2<f64>, Vec<String>) {
    if trajectories.len() == 1 {
        return polynomial_library(&trajectories[0], degree, state_names);
    }
    let views: Vec<_> = trajectories.iter().map(|a| a.view()).collect();
    let stacked = numpy::ndarray::concatenate(Axis(0), &views).expect("stack trajectories");
    polynomial_library(&stacked, degree, state_names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn degree_one_names() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let names = vec!["x".into(), "y".into()];
        let (_, feats) = polynomial_library(&x, 1, &names);
        assert_eq!(feats, vec!["1", "x", "y"]);
    }
}
