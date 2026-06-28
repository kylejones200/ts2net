//! SINDy fit orchestration.

mod finite_diff;
mod polynomial;
mod stlsq;

use numpy::ndarray::{Array1, Array2, Axis};

use finite_diff::finite_difference_stack;
use polynomial::polynomial_library_stack;
use stlsq::{drop_nan_rows, stlsq, StlsqConfig};

#[derive(Debug, Clone)]
pub struct SindyConfig {
    pub polynomial_degree: usize,
    pub threshold: f64,
    pub alpha: f64,
    pub differentiation_order: usize,
    pub max_iter: usize,
}

impl Default for SindyConfig {
    fn default() -> Self {
        Self {
            polynomial_degree: 3,
            threshold: 0.1,
            alpha: 0.05,
            differentiation_order: 2,
            max_iter: 20,
        }
    }
}

pub struct SindyFit {
    pub coefficients: Array2<f64>,
    pub feature_names: Vec<String>,
}

pub fn fit_single(
    x: &Array2<f64>,
    t: &Array1<f64>,
    x_dot: Option<&Array2<f64>>,
    state_names: &[String],
    config: &SindyConfig,
) -> Result<SindyFit, String> {
    fit_many(&[x.clone()], &[t.clone()], x_dot.map(|d| vec![d.clone()]), state_names, config)
}

pub fn fit_many(
    trajectories: &[Array2<f64>],
    times: &[Array1<f64>],
    x_dot: Option<Vec<Array2<f64>>>,
    state_names: &[String],
    config: &SindyConfig,
) -> Result<SindyFit, String> {
    if trajectories.is_empty() {
        return Err("at least one trajectory required".into());
    }
    let n_vars = trajectories[0].ncols();
    if state_names.len() != n_vars {
        return Err(format!(
            "state_names length ({}) must match n_coords ({})",
            state_names.len(),
            n_vars
        ));
    }

    let x_dot_all = match x_dot {
        Some(dots) => {
            if dots.len() != trajectories.len() {
                return Err("x_dot list length must match trajectories".into());
            }
            if dots.len() == 1 {
                dots.into_iter().next().unwrap()
            } else {
                let views: Vec<_> = dots.iter().map(|a| a.view()).collect();
                numpy::ndarray::concatenate(Axis(0), &views).map_err(|e| e.to_string())?
            }
        }
        None => finite_difference_stack(trajectories, times, config.differentiation_order),
    };

    let (theta, feature_names) =
        polynomial_library_stack(trajectories, config.polynomial_degree, state_names);
    let (theta, x_dot_all) = drop_nan_rows(&theta, &x_dot_all);

    let stlsq_cfg = StlsqConfig {
        threshold: config.threshold,
        alpha: config.alpha,
        max_iter: config.max_iter,
        unbias: true,
    };
    let coefficients = stlsq(&theta, &x_dot_all, &stlsq_cfg);

    Ok(SindyFit {
        coefficients,
        feature_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array1;

    fn linspace(start: f64, end: f64, n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64))
    }

    fn tutorial1_x(t: &Array1<f64>) -> Array2<f64> {
        let n = t.len();
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = 3.0 * (-2.0 * t[i]).exp();
            x[[i, 1]] = 0.5 * t[i].exp();
        }
        x
    }

    #[test]
    fn tutorial1_linear_system() {
        let t = linspace(0.0, 1.0, 80);
        let x = tutorial1_x(&t);
        let cfg = SindyConfig {
            polynomial_degree: 1,
            threshold: 0.1,
            alpha: 0.05,
            differentiation_order: 2,
            max_iter: 20,
        };
        let names = vec!["x".into(), "y".into()];
        let fit = fit_single(&x, &t, None, &names, &cfg).unwrap();
        let xi = fit.feature_names.iter().position(|s| s == "x").unwrap();
        let yi = fit.feature_names.iter().position(|s| s == "y").unwrap();
        assert!((fit.coefficients[[0, xi]] + 2.0).abs() < 0.05);
        assert!((fit.coefficients[[1, yi]] - 1.0).abs() < 0.05);
    }
}
