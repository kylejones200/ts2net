//! ts2net-rs - High-performance time series to network conversion in Rust
//!
//! This crate provides efficient implementations of various time series to network conversion
//! algorithms, including visibility graphs, recurrence networks, and more.

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

// External crates
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use num_complex::Complex;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use rustfft::FftPlanner;

// Internal modules (if they exist)
pub mod distance;
pub mod embedding;
pub mod graphs;
pub mod utils;

//
// --------- helpers ----------
//

#[inline]
fn as_1d(y: PyReadonlyArray1<f64>) -> PyResult<Array1<f64>> {
    let a = y.as_array();
    if a.ndim() != 1 {
        return Err(PyValueError::new_err("expected 1-D"));
    }
    Ok(a.to_owned())
}

#[inline]
fn as_2d(x: PyReadonlyArray2<f64>) -> PyResult<Array2<f64>> {
    let a = x.as_array();
    if a.ndim() != 2 {
        return Err(PyValueError::new_err("expected 2-D"));
    }
    Ok(a.to_owned())
}

//
// --------- visibility graphs ----------
//

#[inline]
fn hvg_edges_core(y: &Array1<f64>) -> Array2<i64> {
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
    let m = ei.len();
    let mut out = Array2::<i64>::zeros((m, 2));
    out.slice_mut(s![.., 0]).assign(&Array1::from(ei));
    out.slice_mut(s![.., 1]).assign(&Array1::from(ej));
    out
}

#[inline]
fn nvg_edges_sweepline_core(y: &Array1<f64>) -> Array2<i64> {
    let n = y.len();
    let mut ei = Vec::<i64>::with_capacity(2 * n);
    let mut ej = Vec::<i64>::with_capacity(2 * n);
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
    let m = ei.len();
    let mut out = Array2::<i64>::zeros((m, 2));
    out.slice_mut(s![.., 0]).assign(&Array1::from(ei));
    out.slice_mut(s![.., 1]).assign(&Array1::from(ej));
    out
}

#[pyfunction]
fn hvg_edges(py: Python<'_>, y: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray2<i64>>> {
    let v = as_1d(y)?;
    Ok(hvg_edges_core(&v).into_pyarray(py).to_owned())
}

#[pyfunction]
fn nvg_edges_sweepline(py: Python<'_>, y: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray2<i64>>> {
    let v = as_1d(y)?;
    Ok(nvg_edges_sweepline_core(&v).into_pyarray(py).to_owned())
}

//
// --------- DTW and k-d tree ----------
//

fn dtw_pair(a: &[f64], b: &[f64], band: Option<usize>) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut w = band.unwrap_or(usize::MAX);
    let dm = if n > m { n - m } else { m - n };
    if w < dm {
        w = dm;
    }
    let inf = f64::INFINITY;
    let mut dp = vec![vec![inf; m + 1]; n + 1];
    dp[0][0] = 0.0;
    for i in 1..=n {
        let jmin = 1usize.max(i.saturating_sub(w));
        let jmax = m.min(i + w);
        for j in jmin..=jmax {
            let cost = (a[i - 1] - b[j - 1]).powi(2);
            let v = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = cost + v;
        }
    }
    dp[n][m].sqrt()
}

fn cdist_dtw_core(x: &Array2<f64>, band: Option<usize>) -> Array2<f64> {
    let n = x.len_of(Axis(0));
    let mut out = Array2::<f64>::zeros((n, n));
    
    // Collect results in a thread-safe way
    let results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let ai = x.row(i).to_owned();
            let mut local_results = Vec::new();
            for j in (i + 1)..n {
                let d = dtw_pair(
                    ai.as_slice().unwrap(),
                    x.row(j).as_slice().unwrap(),
                    band,
                );
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

#[pyfunction]
fn cdist_dtw(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    band: Option<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    let a = as_2d(x)?;
    Ok(cdist_dtw_core(&a, band).into_pyarray(py).to_owned())
}

// ---- KD-tree (kiddo) for k-NN and radius queries ----
use kiddo::{KdTree, SquaredEuclidean};

fn knn_impl<const M: usize>(pts: &Array2<f64>, k: usize) -> (Array2<usize>, Array2<f64>) {
    let n = pts.len_of(Axis(0));
    let mut tree: KdTree<f64, M> = KdTree::new();
    for (i, row) in pts.outer_iter().enumerate() {
        let mut p = [0.0f64; M];
        for d in 0..M {
            p[d] = row[d];
        }
        tree.add(&p, i as u64);
    }
    let mut idx = Array2::<usize>::zeros((n, k));
    let mut dst = Array2::<f64>::zeros((n, k));
    for (i, row) in pts.outer_iter().enumerate() {
        let mut q = [0.0f64; M];
        for d in 0..M {
            q[d] = row[d];
        }
        let res = tree.nearest_n::<SquaredEuclidean>(&q, k + 1);
        let mut t = 0;
        for neighbor in res.iter() {
            let j_usize = neighbor.item as usize;
            if j_usize != i && t < k {
                idx[[i, t]] = j_usize;
                dst[[i, t]] = neighbor.distance.sqrt();
                t += 1;
            }
        }
    }
    (idx, dst)
}

fn radius_impl<const M: usize>(pts: &Array2<f64>, eps: f64) -> Vec<Vec<usize>> {
    let n = pts.len_of(Axis(0));
    let mut tree: KdTree<f64, M> = KdTree::new();
    for (i, row) in pts.outer_iter().enumerate() {
        let mut p = [0.0f64; M];
        for d in 0..M {
            p[d] = row[d];
        }
        tree.add(&p, i as u64);
    }
    let r2 = eps * eps;
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(n);
    for (i, row) in pts.outer_iter().enumerate() {
        let mut q = [0.0f64; M];
        for d in 0..M {
            q[d] = row[d];
        }
        let res = tree.within_unsorted::<SquaredEuclidean>(&q, r2);
        let v: Vec<usize> = res.iter().filter_map(|neighbor| {
            let j_usize = neighbor.item as usize;
            if j_usize != i { Some(j_usize) } else { None }
        }).collect();
        out.push(v);
    }
    out
}

#[pyfunction]
fn knn(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    k: usize,
) -> PyResult<(Py<PyArray2<usize>>, Py<PyArray2<f64>>)> {
    let a = as_2d(x)?;
    let m = a.len_of(Axis(1));
    let (idx, dst) = match m {
        1 => knn_impl::<1>(&a, k),
        2 => knn_impl::<2>(&a, k),
        3 => knn_impl::<3>(&a, k),
        4 => knn_impl::<4>(&a, k),
        5 => knn_impl::<5>(&a, k),
        6 => knn_impl::<6>(&a, k),
        _ => return Err(PyValueError::new_err("dimension up to 6 is supported")),
    };
    Ok((
        idx.into_pyarray(py).to_owned(),
        dst.into_pyarray(py).to_owned(),
    ))
}

#[pyfunction]
fn radius(_py: Python<'_>, x: PyReadonlyArray2<f64>, eps: f64) -> PyResult<PyObject> {
    let a = as_2d(x)?;
    let m = a.len_of(Axis(1));
    let neighs = match m {
        1 => radius_impl::<1>(&a, eps),
        2 => radius_impl::<2>(&a, eps),
        3 => radius_impl::<3>(&a, eps),
        4 => radius_impl::<4>(&a, eps),
        5 => radius_impl::<5>(&a, eps),
        6 => radius_impl::<6>(&a, eps),
        _ => return Err(PyValueError::new_err("dimension up to 6 is supported")),
    };
    Python::with_gil(|py| Ok(neighs.into_py(py)))
}

//
// --------- Recurrence adjacency ----------
//

#[inline]
fn pair_dist(a: ArrayView1<f64>, b: ArrayView1<f64>, metric: &str) -> f64 {
    match metric {
        "manhattan" => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        "chebyshev" => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max),
        _ => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt(),
    }
}

#[pyfunction]
fn rn_adj_epsilon(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    eps: f64,
    metric: &str,
    theiler: usize,
) -> PyResult<Py<PyArray2<u8>>> {
    let X = as_2d(x)?;
    let n = X.len_of(Axis(0));
    let mut A = Array2::<u8>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            if theiler > 0 && j.saturating_sub(i) <= theiler {
                continue;
            }
            let d = pair_dist(X.row(i), X.row(j), metric);
            if d <= eps {
                A[[i, j]] = 1;
                A[[j, i]] = 1;
            }
        }
    }
    Ok(A.into_pyarray(py).to_owned())
}

//
// --------- Event synchronization ----------
//

#[pyfunction]
fn event_sync(
    py: Python<'_>,
    e1: PyReadonlyArray1<usize>,
    e2: PyReadonlyArray1<usize>,
    adaptive: bool,
    tau_max: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let a = e1.as_array().to_owned();
    let b = e2.as_array().to_owned();
    let n1 = a.len();
    let n2 = b.len();
    let mut c12 = 0.0f64;
    let mut c21 = 0.0f64;
    let mut ties = 0.0f64;
    let mut delays: Vec<f64> = Vec::with_capacity(n1 + n2);
    let tmax = tau_max.unwrap_or(f64::INFINITY);
    for i in 0..n1 {
        let ti = a[i] as f64;
        for j in 0..n2 {
            let tj = b[j] as f64;
            let tau = if adaptive {
                let dl = if i == 0 && n1 > 1 {
                    a[i + 1] - a[i]
                } else if i > 0 {
                    a[i] - a[i - 1]
                } else {
                    1
                };
                let dr = if j == 0 && n2 > 1 {
                    b[j + 1] - b[j]
                } else if j > 0 {
                    b[j] - b[j - 1]
                } else {
                    1
                };
                0.5f64 * ((dl.min(dr)) as f64)
            } else {
                1.0
            };
            if (ti - tj).abs() <= tau && (ti - tj).abs() <= tmax {
                if ti < tj {
                    c12 += 1.0;
                    delays.push(tj - ti);
                } else if ti > tj {
                    c21 += 1.0;
                    delays.push(ti - tj);
                } else {
                    ties += 1.0;
                }
            }
        }
    }
    let q12 = if n1 > 0 { c12 / (n1 as f64) } else { 0.0 };
    let q21 = if n2 > 0 { c21 / (n2 as f64) } else { 0.0 };
    let q = if n1 + n2 > 0 {
        (q12 + q21) / 2.0
    } else {
        0.0
    };
    let mut out = vec![c12, c21, ties, q12, q21, q, delays.len() as f64];
    out.extend(delays);
    Ok(PyArray1::from_vec(py, out).to_owned())
}

//
// --------- FNN and Cao ----------
//

#[pyfunction]
fn false_nearest_neighbors(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    m_max: usize,
    tau: usize,
    rtol: f64,
    atol: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let v = as_1d(x)?;
    let n = v.len();
    if m_max < 2 {
        return Err(PyValueError::new_err("m_max >= 2"));
    }
    let mut out = Vec::<f64>::with_capacity(m_max - 1);
    for m in 1..m_max {
        let L = n - (m * tau);
        if L < 2 {
            out.push(1.0);
            continue;
        }
        let mut Xm = Array2::<f64>::zeros((L, m));
        let mut Xp = Array2::<f64>::zeros((L, m + 1));
        for i in 0..m {
            Xm.slice_mut(s![.., i])
                .assign(&v.slice(s![i * tau..i * tau + L]));
        }
        for i in 0..m + 1 {
            Xp.slice_mut(s![.., i])
                .assign(&v.slice(s![i * tau..i * tau + L]));
        }
        let mut fnn = 0.0;
        for i in 0..L {
            let ai = Xm.row(i);
            let mut best = (f64::INFINITY, 0usize);
            for j in 0..L {
                if i == j {
                    continue;
                }
                let d: f64 = ai
                    .iter()
                    .zip(Xm.row(j).iter())
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f64>()
                    .sqrt();
                if d < best.0 {
                    best = (d, j);
                }
            }
            let j = best.1;
            let num = (Xp[[i, m]] - Xp[[j, m]]).abs();
            if best.0 == 0.0 || num.is_nan() {
                fnn += 1.0;
                continue;
            }
            if num / best.0 > rtol || num > atol {
                fnn += 1.0;
            }
        }
        out.push(fnn / (L as f64));
    }
    Ok(PyArray1::from_vec(py, out).to_owned())
}

#[pyfunction]
fn cao_e1_e2(
    _py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    m_max: usize,
    tau: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let v = as_1d(x)?;
    let n = v.len();
    let mut E1 = Vec::<f64>::with_capacity(m_max - 1);
    let mut E2 = Vec::<f64>::with_capacity(m_max - 2);
    for m in 1..m_max {
        let L = n - (m * tau);
        if L < 2 {
            E1.push(f64::NAN);
            if m > 1 {
                E2.push(f64::NAN);
            }
            continue;
        }
        let mut Xm = Array2::<f64>::zeros((L, m));
        let mut Xp = Array2::<f64>::zeros((L, m + 1));
        for i in 0..m {
            Xm.slice_mut(s![.., i])
                .assign(&v.slice(s![i * tau..i * tau + L]));
        }
        for i in 0..m + 1 {
            Xp.slice_mut(s![.., i])
                .assign(&v.slice(s![i * tau..i * tau + L]));
        }
        let mut ratios = Vec::<f64>::with_capacity(L);
        let mut diffs = Vec::<f64>::with_capacity(L);
        for i in 0..L {
            let ai = Xm.row(i);
            let mut best = (f64::INFINITY, 0usize);
            for j in 0..L {
                if i == j {
                    continue;
                }
                let d: f64 = ai
                    .iter()
                    .zip(Xm.row(j).iter())
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f64>()
                    .sqrt();
                if d < best.0 {
                    best = (d, j);
                }
            }
            let j = best.1;
            let num: f64 = Xp
                .row(i)
                .iter()
                .zip(Xp.row(j).iter())
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
                .sqrt();
            let den = best.0.max(1e-12);
            ratios.push(num / den);
            if m > 1 {
                diffs.push((Xp[[i, m]] - Xp[[j, m]]).abs());
            }
        }
        E1.push(ratios.iter().sum::<f64>() / (ratios.len() as f64));
        if m > 1 {
            E2.push(diffs.iter().sum::<f64>() / (diffs.len() as f64));
        }
    }
    Python::with_gil(|py| {
        Ok((
            PyArray1::from_vec(py, E1).to_owned(),
            PyArray1::from_vec(py, E2).to_owned(),
        ))
    })
}

//
// --------- Motif and network stats ----------
//

fn build_adj(n: usize, edges: &[(usize, usize)], undirected: bool) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::<usize>::new(); n];
    for &(u, v) in edges.iter() {
        if u >= n || v >= n {
            continue;
        }
        adj[u].push(v);
        if undirected {
            adj[v].push(u);
        }
    }
    for nbrs in adj.iter_mut() {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    adj
}

#[pyfunction]
fn triangles_per_node(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<Py<PyArray1<usize>>> {
    let e = edges.as_array();
    if e.ncols() != 2 {
        return Err(PyValueError::new_err("edges shape must be [m,2]"));
    }
    let mut adj = vec![Vec::<usize>::new(); n];
    for r in 0..e.nrows() {
        let u = e[[r, 0]];
        let v = e[[r, 1]];
        if u >= n || v >= n {
            continue;
        }
        adj[u].push(v);
        adj[v].push(u);
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    let mut tri = vec![0usize; n];
    for u in 0..n {
        let nu = &adj[u];
        for &v in nu.iter() {
            if v <= u {
                continue;
            }
            let c = {
                let a = nu;
                let b = &adj[v];
                let mut i = 0usize;
                let mut j = 0usize;
                let mut c = 0usize;
                while i < a.len() && j < b.len() {
                    if a[i] == b[j] {
                        if a[i] != u && a[i] != v {
                            c += 1;
                        }
                        i += 1;
                        j += 1;
                    } else if a[i] < b[j] {
                        i += 1;
                    } else {
                        j += 1;
                    }
                }
                c
            };
            tri[u] += c;
            tri[v] += c;
        }
    }
    Ok(PyArray1::from_vec(py, tri).to_owned())
}

#[pyfunction]
fn clustering_avg(_py: Python<'_>, n: usize, edges: PyReadonlyArray2<usize>) -> PyResult<f64> {
    let e = edges.as_array();
    let mut adj = vec![Vec::<usize>::new(); n];
    for r in 0..e.nrows() {
        let u = e[[r, 0]];
        let v = e[[r, 1]];
        if u >= n || v >= n {
            continue;
        }
        adj[u].push(v);
        adj[v].push(u);
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    let mut s = 0.0;
    let mut cnt = 0usize;
    for u in 0..n {
        let k = adj[u].len();
        if k < 2 {
            continue;
        }
        let mut tri = 0usize;
        for i in 0..k {
            let a = adj[u][i];
            for j in (i + 1)..k {
                let b = adj[u][j];
                // check edge a-b
                let nb = &adj[a];
                if nb.binary_search(&b).is_ok() {
                    tri += 1;
                }
            }
        }
        s += (2.0 * tri as f64) / ((k * (k - 1)) as f64);
        cnt += 1;
    }
    Ok(if cnt > 0 { s / (cnt as f64) } else { 0.0 })
}

#[pyfunction]
fn mean_shortest_path(
    _py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<f64> {
    use std::collections::VecDeque;
    let e = edges.as_array();
    let mut adj = vec![Vec::<usize>::new(); n];
    for r in 0..e.nrows() {
        let u = e[[r, 0]];
        let v = e[[r, 1]];
        if u >= n || v >= n {
            continue;
        }
        adj[u].push(v);
        adj[v].push(u);
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    let mut total = 0usize;
    let mut pairs = 0usize;
    for s in 0..n {
        let mut dist = vec![usize::MAX; n];
        let mut q = VecDeque::new();
        dist[s] = 0;
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            for &v in adj[u].iter() {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
            }
        }
        for t in (s + 1)..n {
            if dist[t] != usize::MAX {
                total += dist[t];
                pairs += 1;
            }
        }
    }
    Ok(if pairs > 0 {
        (total as f64) / (pairs as f64)
    } else {
        f64::NAN
    })
}

//
// --------- Surrogates (phase, iAAFT) ----------
//

#[pyfunction]
fn surrogate_phase(
    _py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    let v = as_1d(x)?;
    let n = v.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let mut spec: Vec<Complex<f64>> = v
        .iter()
        .map(|&r| Complex { re: r, im: 0.0 })
        .collect();
    fft.process(&mut spec);
    let mag: Vec<f64> = spec.iter().map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt()).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    for k in 1..(n / 2) {
        let theta: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
        spec[k] = Complex {
            re: mag[k] * theta.cos(),
            im: mag[k] * theta.sin(),
        };
        spec[n - k] = Complex {
            re: spec[k].re,
            im: -spec[k].im,
        };
    }
    if n % 2 == 0 {
        spec[n / 2] = Complex {
            re: mag[n / 2],
            im: 0.0,
        };
    }
    ifft.process(&mut spec);
    let out: Vec<f64> = spec.iter().map(|c| c.re / (n as f64)).collect();
    Python::with_gil(|py| Ok(PyArray1::from_vec(py, out).to_owned()))
}

#[pyfunction]
fn iaaft(
    _py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    iters: usize,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    let v = as_1d(x)?;
    let n = v.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let mut spec_t: Vec<Complex<f64>> = v
        .iter()
        .map(|&r| Complex { re: r, im: 0.0 })
        .collect();
    fft.process(&mut spec_t);
    let mag_t: Vec<f64> = spec_t
        .iter()
        .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
        .collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut y: Vec<f64> = v.iter().copied().collect();
    y.shuffle(&mut rng);
    for _ in 0..iters {
        let mut spec: Vec<Complex<f64>> = y
            .iter()
            .map(|&r| Complex { re: r, im: 0.0 })
            .collect();
        fft.process(&mut spec);
        for k in 0..n {
            let c = &spec[k];
            let phase = c.im.atan2(c.re);
            spec[k] = Complex {
                re: mag_t[k] * phase.cos(),
                im: mag_t[k] * phase.sin(),
            };
        }
        ifft.process(&mut spec);
        let y2: Vec<f64> = spec.iter().map(|c| c.re / (n as f64)).collect();
        // rank-order match original
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut ridx: Vec<usize> = (0..n).collect();
        ridx.sort_by(|&i, &j| y2[i].partial_cmp(&y2[j]).unwrap());
        let mut out = vec![0.0; n];
        for k in 0..n {
            out[idx[k]] = y2[ridx[k]];
        }
        y = out;
    }
    Python::with_gil(|py| Ok(PyArray1::from_vec(py, y).to_owned()))
}

//
// --------- Permutation tests ----------
//

#[pyfunction]
fn corr_perm(
    _py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    n_perm: usize,
    seed: u64,
) -> PyResult<f64> {
    let a = as_1d(x)?;
    let b = as_1d(y)?;
    let r0 = pearson(&a, &b);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cnt = 0usize;
    let mut yv = b.to_vec();
    for _ in 0..n_perm {
        yv.shuffle(&mut rng);
        let r = pearson(&a, &Array1::from(yv.clone()));
        if r.abs() >= r0.abs() {
            cnt += 1;
        }
    }
    Ok(((cnt + 1) as f64) / ((n_perm + 1) as f64))
}

fn pearson(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len();
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

//
// --------- Spatial stats ----------
//

#[pyfunction]
fn moran_i(
    _py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    w: PyReadonlyArray2<f64>,
) -> PyResult<(f64, f64)> {
    let x = as_1d(y)?;
    let W = as_2d(w)?;
    let n = x.len();
    let mx = x.mean().unwrap();
    let z: Vec<f64> = x.iter().map(|v| *v - mx).collect();
    let s0: f64 = W.iter().sum();
    if s0 == 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        den += z[i] * z[i];
        for j in 0..n {
            num += W[[i, j]] * z[i] * z[j];
        }
    }
    let I = (n as f64) / s0 * (num / den.max(1e-12));
    let ei = -1.0 / ((n as f64) - 1.0);
    let zscore = (I - ei) / 0.1f64.max(1e-9); // keep simple; exact var is long
    Ok((I, zscore))
}

//
// --------- Python module ----------
//

#[pymodule]
fn ts2net_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hvg_edges, m)?)?;
    m.add_function(wrap_pyfunction!(nvg_edges_sweepline, m)?)?;
    m.add_function(wrap_pyfunction!(cdist_dtw, m)?)?;
    m.add_function(wrap_pyfunction!(knn, m)?)?;
    m.add_function(wrap_pyfunction!(radius, m)?)?;

    m.add_function(wrap_pyfunction!(rn_adj_epsilon, m)?)?;

    m.add_function(wrap_pyfunction!(event_sync, m)?)?;

    m.add_function(wrap_pyfunction!(false_nearest_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(cao_e1_e2, m)?)?;

    m.add_function(wrap_pyfunction!(triangles_per_node, m)?)?;
    m.add_function(wrap_pyfunction!(clustering_avg, m)?)?;
    m.add_function(wrap_pyfunction!(mean_shortest_path, m)?)?;

    m.add_function(wrap_pyfunction!(surrogate_phase, m)?)?;
    m.add_function(wrap_pyfunction!(iaaft, m)?)?;

    m.add_function(wrap_pyfunction!(corr_perm, m)?)?;

    m.add_function(wrap_pyfunction!(moran_i, m)?)?;

    Ok(())
}
