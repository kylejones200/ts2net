//! Sequentially thresholded least squares (STLSQ) with optional unbiasing.

use numpy::ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct StlsqConfig {
    pub threshold: f64,
    pub alpha: f64,
    pub max_iter: usize,
    pub unbias: bool,
}

impl Default for StlsqConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            alpha: 0.05,
            max_iter: 20,
            unbias: true,
        }
    }
}

fn ridge_regression(x: &Array2<f64>, y: &Array1<f64>, alpha: f64) -> Array1<f64> {
    if alpha == 0.0 {
        return lstsq(x, y);
    }
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut aug = Array2::<f64>::zeros((n_samples + n_features, n_features));
    let mut rhs = vec![0.0; n_samples + n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            aug[[i, j]] = x[[i, j]];
        }
        rhs[i] = y[i];
    }
    let sqrt_alpha = alpha.sqrt();
    for j in 0..n_features {
        aug[[n_samples + j, j]] = sqrt_alpha;
    }
    solve_linear_square(&aug, &rhs)
}

fn lstsq(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let n_features = x.ncols();
    let mut xtx = Array2::<f64>::zeros((n_features, n_features));
    let mut xty = Array1::<f64>::zeros(n_features);

    for i in 0..x.nrows() {
        for a in 0..n_features {
            xty[a] += x[[i, a]] * y[i];
            for b in 0..n_features {
                xtx[[a, b]] += x[[i, a]] * x[[i, b]];
            }
        }
    }
    solve_linear(&xtx, &xty)
}

fn solve_linear(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[[i, j]];
        }
        aug[i][n] = b[i];
    }
    gauss_elim(&mut aug)
}

fn solve_linear_square(a: &Array2<f64>, b: &[f64]) -> Array1<f64> {
    let n = a.ncols();
    let m = a.nrows();
    let mut ata = Array2::<f64>::zeros((n, n));
    let mut atb = Array1::<f64>::zeros(n);
    for i in 0..m {
        for j in 0..n {
            atb[j] += a[[i, j]] * b[i];
            for k in 0..n {
                ata[[j, k]] += a[[i, j]] * a[[i, k]];
            }
        }
    }
    solve_linear(&ata, &atb)
}

fn gauss_elim(aug: &mut [Vec<f64>]) -> Array1<f64> {
    let n = aug.len();
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[pivot][col].abs() {
                pivot = row;
            }
        }
        aug.swap(col, pivot);
        let div = aug[col][col];
        if div.abs() < 1e-15 {
            continue;
        }
        for j in col..=n {
            aug[col][j] /= div;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    Array1::from_iter((0..n).map(|i| aug[i][n]))
}

fn support_equal(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (*x != 0.0) == (*y != 0.0))
}

fn fit_target(
    theta: &Array2<f64>,
    y: &Array1<f64>,
    config: &StlsqConfig,
) -> Array1<f64> {
    let n_samples = theta.nrows();
    let n_features = theta.ncols();
    let mut ind = vec![true; n_features];
    let mut history: Vec<Array1<f64>> = vec![lstsq(theta, y)];

    for _ in 0..config.max_iter {
        let n_selected: usize = ind.iter().filter(|&&b| b).count();
        if n_selected == 0 {
            break;
        }

        let active: Vec<usize> = (0..n_features).filter(|&i| ind[i]).collect();
        let mut x_sub = Array2::<f64>::zeros((n_samples, active.len()));
        for (col, &feat) in active.iter().enumerate() {
            for row in 0..n_samples {
                x_sub[[row, col]] = theta[[row, feat]];
            }
        }

        let coef_sub = ridge_regression(&x_sub, y, config.alpha);
        let mut optvar = Array1::<f64>::zeros(n_features);
        for (col, &feat) in active.iter().enumerate() {
            optvar[feat] = coef_sub[col];
        }

        let mut new_ind = vec![false; n_features];
        for i in 0..n_features {
            if optvar[i].abs() >= config.threshold {
                new_ind[i] = true;
            } else {
                optvar[i] = 0.0;
            }
        }

        history.push(optvar.clone());
        let no_change = if history.len() > 1 {
            support_equal(&history[history.len() - 1], &history[history.len() - 2])
        } else {
            false
        };

        ind = new_ind;
        if ind.iter().filter(|&&b| b).count() == n_selected || no_change {
            break;
        }
    }

    let mut coef = history.last().cloned().unwrap_or_else(|| Array1::zeros(n_features));

    if config.unbias {
        let support: Vec<usize> = (0..n_features)
            .filter(|&i| coef[i].abs() > 1e-14)
            .collect();
        if !support.is_empty() {
            let mut x_sub = Array2::<f64>::zeros((n_samples, support.len()));
            for (col, &feat) in support.iter().enumerate() {
                for row in 0..n_samples {
                    x_sub[[row, col]] = theta[[row, feat]];
                }
            }
            let w = lstsq(&x_sub, y);
            coef.fill(0.0);
            for (col, &feat) in support.iter().enumerate() {
                coef[feat] = w[col];
            }
        }
    }

    coef
}

/// Fit STLSQ for all target columns. Returns (n_targets × n_features).
pub fn stlsq(theta: &Array2<f64>, x_dot: &Array2<f64>, config: &StlsqConfig) -> Array2<f64> {
    let n_targets = x_dot.ncols();
    let n_features = theta.ncols();
    let mut coef = Array2::<f64>::zeros((n_targets, n_features));
    for target in 0..n_targets {
        let y = x_dot.column(target).to_owned();
        let w = fit_target(theta, &y, config);
        for j in 0..n_features {
            coef[[target, j]] = w[j];
        }
    }
    coef
}

/// Drop rows containing NaN in ``x_dot`` (and matching rows in ``theta``).
pub fn drop_nan_rows(theta: &Array2<f64>, x_dot: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut keep = Vec::new();
    for i in 0..theta.nrows() {
        let row_nan = (0..x_dot.ncols()).any(|j| x_dot[[i, j]].is_nan());
        if !row_nan {
            keep.push(i);
        }
    }
    let n = keep.len();
    let mut t_out = Array2::<f64>::zeros((n, theta.ncols()));
    let mut d_out = Array2::<f64>::zeros((n, x_dot.ncols()));
    for (r, &i) in keep.iter().enumerate() {
        for j in 0..theta.ncols() {
            t_out[[r, j]] = theta[[i, j]];
        }
        for j in 0..x_dot.ncols() {
            d_out[[r, j]] = x_dot[[i, j]];
        }
    }
    (t_out, d_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sindy::finite_diff::finite_difference;
    use crate::sindy::polynomial::polynomial_library;
    use numpy::ndarray::{array, Array2};

    #[test]
    fn recovers_linear_relation() {
        let theta = array![[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]];
        let x_dot = array![[3.0], [5.0], [7.0]];
        let cfg = StlsqConfig {
            threshold: 0.5,
            alpha: 0.0,
            max_iter: 20,
            unbias: true,
        };
        let coef = stlsq(&theta, &x_dot, &cfg);
        assert!((coef[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((coef[[0, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gauss_elim_2x2() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let w = solve_linear(&a, &b);
        let expected = array![0.09090909090909091, 0.6363636363636364];
        for i in 0..2 {
            assert!((w[i] - expected[i]).abs() < 1e-9, "w={:?}", w.to_vec());
        }
    }

    #[test]
    fn normal_equations_debug() {
        let xtx = array![
            [80.0, 104.17124436435753, 68.80260894561403],
            [104.17124436435753, 179.1140931227067, 75.9331959901493],
            [68.80260894561403, 75.9331959901493, 64.14393070863758],
        ];
        let xty = array![-208.36258239841283, -358.26065037046715, -151.88130736699208];
        let mut reg = xtx.clone();
        for i in 0..3 {
            reg[[i, i]] += 0.05;
        }
        let w = solve_linear(&reg, &xty);
        assert!((w[0] + 0.052).abs() < 0.01, "w={:?}", w.to_vec());
        assert!((w[1] + 1.986).abs() < 0.05, "w={:?}", w.to_vec());
    }

    #[test]
    fn ridge_matches_sklearn() {
        let n = 80;
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            t.push(i as f64 / 79.0);
        }
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = 3.0 * (-2.0 * t[i]).exp();
            x[[i, 1]] = 0.5 * t[i].exp();
        }
        let xd = finite_difference(&x, &t, 2);
        let names = vec!["x".into(), "y".into()];
        let (theta, _) = polynomial_library(&x, 1, &names);
        let y = xd.column(0).to_owned();
        let w = ridge_regression(&theta, &y, 0.05);
        assert!((w[0] + 0.052).abs() < 0.01, "w0={:?}", w.to_vec());
        assert!((w[1] + 1.986).abs() < 0.05, "w1={:?}", w.to_vec());
    }
}
