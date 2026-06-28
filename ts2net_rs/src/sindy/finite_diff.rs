//! Finite-difference time derivatives (PySINDy-compatible for uniform grids).

use numpy::ndarray::{Array1, Array2, Axis};

fn factorial(n: usize) -> f64 {
    (1..=n).product::<usize>() as f64
}

fn n_stencil(order: usize) -> usize {
    2 * ((1 + 1) / 2) - 1 + order
}

fn n_stencil_forward(order: usize) -> usize {
    1 + order
}

fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Array1<f64> {
    let n = b.len();
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }
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

fn constant_coefficients(order: usize, dt: f64) -> Array1<f64> {
    let n = n_stencil(order);
    let mut mat = vec![vec![0.0; n]; n];
    let half = (n - 1) / 2;
    for pow in 0..n {
        for point in 0..n {
            let x = dt * (point as f64 - half as f64);
            mat[pow][point] = x.powi(pow as i32);
        }
    }
    let mut b = vec![0.0; n];
    b[1] = factorial(1);
    solve_linear(&mat, &b)
}

fn boundary_coefficients(order: usize, dt: f64) -> Array2<f64> {
    let n = n_stencil(order);
    let nf = n_stencil_forward(order);
    let left_len = (n - 1) / 2;
    let right_len = if order % 2 == 0 {
        (n - 1) / 2
    } else {
        1 + (n - 1) / 2
    };
    let n_bound = left_len + right_len;

    let mut tinds: Vec<i32> = Vec::with_capacity(n_bound);
    for i in 0..left_len {
        tinds.push(i as i32);
    }
    for i in 0..right_len {
        tinds.push(-1 - i as i32);
    }

    let mut left = vec![vec![0i32; left_len]; nf];
    for i in 0..nf {
        for j in 0..left_len {
            left[i][j] = i as i32;
        }
    }
    let mut right = vec![vec![0i32; right_len]; nf];
    for i in 0..nf {
        for j in 0..right_len {
            right[i][j] = -1 - i as i32;
        }
    }

    let mut stencil_inds = vec![vec![0i32; n_bound]; nf];
    for i in 0..nf {
        for j in 0..left_len {
            stencil_inds[i][j] = left[i][j];
        }
        for j in 0..right_len {
            stencil_inds[i][left_len + j] = right[i][j];
        }
    }

    let mut coeffs = Array2::<f64>::zeros((n_bound, nf));
    for b in 0..n_bound {
        let mut mat = vec![vec![0.0; nf]; nf];
        for pow in 0..nf {
            for j in 0..nf {
                let delta = dt * (stencil_inds[j][b] - tinds[b]) as f64;
                mat[pow][j] = delta.powi(pow as i32);
            }
        }
        let mut rhs = vec![0.0; nf];
        rhs[1] = factorial(1);
        let col = solve_linear(&mat, &rhs);
        for j in 0..nf {
            coeffs[[b, j]] = col[j];
        }
    }
    coeffs
}

fn is_uniform(t: &[f64]) -> Option<f64> {
    if t.len() < 2 {
        return None;
    }
    let dt = t[1] - t[0];
    if t.windows(2).all(|w| (w[1] - w[0] - dt).abs() < 1e-10) {
        Some(dt)
    } else {
        None
    }
}

/// Differentiate ``x`` (n_time × n_vars) along time.
pub fn finite_difference(x: &Array2<f64>, t: &[f64], order: usize) -> Array2<f64> {
    let n_time = x.nrows();
    let n_vars = x.ncols();
    let mut x_dot = Array2::<f64>::from_elem((n_time, n_vars), f64::NAN);
    if n_time < 2 {
        return x_dot;
    }

    let dt = match is_uniform(t) {
        Some(dt) if dt.abs() > 0.0 => dt,
        _ => return x_dot,
    };

    let n_st = n_stencil(order);
    let half = (n_st - 1) / 2;
    let interior_coeffs = constant_coefficients(order, dt);

    if n_time > n_st - 1 {
        let interior_len = n_time - (n_st - 1);
        let mut interior = Array2::<f64>::zeros((interior_len, n_vars));
        for k in 0..n_st {
            let start = k;
            let end = n_time - (n_st - k - 1);
            for i in start..end {
                for v in 0..n_vars {
                    interior[[i - start, v]] += interior_coeffs[k] * x[[i, v]];
                }
            }
        }
        for i in 0..interior_len {
            for v in 0..n_vars {
                x_dot[[i + half, v]] = interior[[i, v]];
            }
        }
    }

    let left_len = (n_st - 1) / 2;
    let right_len = if order % 2 == 0 {
        (n_st - 1) / 2
    } else {
        1 + (n_st - 1) / 2
    };
    let n_bound = left_len + right_len;
    let nf = n_stencil_forward(order);

    let mut tinds: Vec<i32> = Vec::with_capacity(n_bound);
    for i in 0..left_len {
        tinds.push(i as i32);
    }
    for i in 0..right_len {
        tinds.push(-1 - i as i32);
    }

    let mut stencil_inds = vec![vec![0i32; n_bound]; nf];
    for j in 0..nf {
        for i in 0..left_len {
            stencil_inds[j][i] = j as i32;
        }
        for i in 0..right_len {
            stencil_inds[j][left_len + i] = -1 - j as i32;
        }
    }

    let bound_coeffs = boundary_coefficients(order, dt);

    for b in 0..n_bound {
        let row = py_index(n_time, tinds[b]);
        for v in 0..n_vars {
            let mut val = 0.0;
            for j in 0..nf {
                let idx = py_index(n_time, tinds[b] + stencil_inds[j][b]);
                val += bound_coeffs[[b, j]] * x[[idx, v]];
            }
            x_dot[[row, v]] = val;
        }
    }

    x_dot
}

#[inline]
fn py_index(n: usize, idx: i32) -> usize {
    let mut i = idx;
    if i < 0 {
        i += n as i32;
    }
    i as usize
}

/// Stack per-trajectory derivatives.
pub fn finite_difference_stack(
    trajectories: &[Array2<f64>],
    times: &[Array1<f64>],
    order: usize,
) -> Array2<f64> {
    let mut parts = Vec::new();
    for (x, t) in trajectories.iter().zip(times.iter()) {
        let t_slice: Vec<f64> = t.iter().copied().collect();
        parts.push(finite_difference(x, &t_slice, order));
    }
    if parts.len() == 1 {
        return parts.into_iter().next().unwrap();
    }
    let views: Vec<_> = parts.iter().map(|a| a.view()).collect();
    numpy::ndarray::concatenate(Axis(0), &views).expect("concatenate derivatives")
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array2;

    #[test]
    fn central_diff_coeffs() {
        let dt = 1.0 / 79.0;
        let c = constant_coefficients(2, dt);
        assert!((c[0] + 39.5).abs() < 1e-6, "c0={}", c[0]);
        assert!(c[1].abs() < 1e-6, "c1={}", c[1]);
        assert!((c[2] - 39.5).abs() < 1e-6, "c2={} full={:?}", c[2], c.to_vec());
    }

    #[test]
    fn tutorial1_derivative_at_40() {
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
        assert!((xd[[40, 0]] + 2.18).abs() < 0.1, "got {}", xd[[40, 0]]);
        assert!((xd[[40, 1]] - 0.83).abs() < 0.1, "got {}", xd[[40, 1]]);
    }
}
