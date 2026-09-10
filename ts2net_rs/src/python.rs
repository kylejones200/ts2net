//! PyO3 bindings.
//!
//! This module is a thin adapter: every function here converts NumPy arrays to
//! [`ndarray`] types, calls into the pure-Rust algorithm modules, and converts
//! the result back. Algorithms themselves live in [`crate::distance`],
//! [`crate::embedding`], [`crate::graphs`], [`crate::sindy`] and
//! [`crate::utils`], none of which depend on Python.

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::distance::{self, Metric};
use crate::embedding;
use crate::graphs;
use crate::sindy;
use crate::utils;

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

#[inline]
fn as_edges(edges: PyReadonlyArray2<usize>) -> PyResult<Vec<(usize, usize)>> {
    let e = edges.as_array();
    if e.ncols() != 2 {
        return Err(PyValueError::new_err("edges shape must be [m,2]"));
    }
    Ok((0..e.nrows()).map(|r| (e[[r, 0]], e[[r, 1]])).collect())
}

//
// --------- visibility graphs ----------
//

#[pyfunction]
fn hvg_edges(py: Python<'_>, y: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray2<i64>>> {
    let v = as_1d(y)?;
    Ok(graphs::hvg_edges(&v).into_pyarray(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (y, directed=false, limit=None))]
fn hvg_degrees(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    directed: bool,
    limit: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let v = as_1d(y)?;
    let lim = limit.map(|l| l as usize);
    let (in_deg, out_deg, n_edges) = graphs::hvg_degrees(&v, directed, lim);
    let dict = PyDict::new(py);
    dict.set_item("n_edges", n_edges)?;
    dict.set_item("directed", directed)?;
    if directed {
        dict.set_item("in_degree", in_deg.into_pyarray(py).unbind())?;
        dict.set_item("out_degree", out_deg.into_pyarray(py).unbind())?;
    } else {
        dict.set_item("degree", out_deg.into_pyarray(py).unbind())?;
    }
    Ok(dict.into())
}

#[pyfunction]
fn nvg_edges_sweepline(py: Python<'_>, y: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray2<i64>>> {
    let v = as_1d(y)?;
    Ok(graphs::nvg_edges_sweepline(&v).into_pyarray(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (y, limit=None))]
fn nvg_degrees(py: Python<'_>, y: PyReadonlyArray1<f64>, limit: Option<u64>) -> PyResult<Py<PyAny>> {
    let v = as_1d(y)?;
    let lim = limit.map(|l| l as usize);
    let (degrees, n_edges) = graphs::nvg_degrees(&v, lim);
    let dict = PyDict::new(py);
    dict.set_item("n_edges", n_edges)?;
    dict.set_item("degree", degrees.into_pyarray(py).unbind())?;
    Ok(dict.into())
}

//
// --------- DTW and k-d tree ----------
//

#[pyfunction]
#[pyo3(signature = (x, band=None))]
fn cdist_dtw(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    band: Option<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let a = as_2d(x)?;
    // Use u64 at the Python boundary (usize is platform-dependent and causes
    // a segfault with keyword args under PyO3 0.19 + Python 3.14).
    let band_usize = band.map(|b| b as usize);
    Ok(distance::cdist_dtw(&a, band_usize).into_pyarray(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (a, b, band=None))]
fn cdist_dtw_rectangular(
    py: Python<'_>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
    band: Option<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let aa = as_2d(a)?;
    let bb = as_2d(b)?;
    let band_usize = band.map(|v| v as usize);
    Ok(distance::cdist_dtw_rectangular(&aa, &bb, band_usize)
        .into_pyarray(py)
        .unbind())
}

#[pyfunction]
fn knn(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    k: usize,
) -> PyResult<(Py<PyArray2<usize>>, Py<PyArray2<f64>>)> {
    let a = as_2d(x)?;
    let (idx, dst) = distance::knn(&a, k).map_err(PyValueError::new_err)?;
    Ok((
        idx.into_pyarray(py).unbind(),
        dst.into_pyarray(py).unbind(),
    ))
}

#[pyfunction]
fn radius(py: Python<'_>, x: PyReadonlyArray2<f64>, eps: f64) -> PyResult<Py<PyAny>> {
    let a = as_2d(x)?;
    let neighs = distance::radius(&a, eps).map_err(PyValueError::new_err)?;
    Ok(neighs.into_pyobject(py)?.unbind())
}

//
// --------- Recurrence adjacency ----------
//

#[pyfunction]
fn rn_adj_epsilon(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    eps: f64,
    metric: &str,
    theiler: usize,
) -> PyResult<Py<PyArray2<u8>>> {
    let points = as_2d(x)?;
    let adj = graphs::rn_adj_epsilon(&points, eps, Metric::from_name(metric), theiler);
    Ok(adj.into_pyarray(py).unbind())
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
    let result = distance::event_sync(
        a.as_slice().unwrap(),
        b.as_slice().unwrap(),
        adaptive,
        tau_max,
    );
    Ok(PyArray1::from_vec(py, result.into_packed()).unbind())
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
    let out = embedding::false_nearest_neighbors(&v, m_max, tau, rtol, atol)
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, out).unbind())
}

#[pyfunction]
#[pyo3(signature = (x, rule="mutual_information", max_lag=None, bins=None))]
fn select_delay(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    rule: &str,
    max_lag: Option<usize>,
    bins: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let v = as_1d(x)?;
    let parsed = match rule {
        "mutual_information" => embedding::DelayRule::MutualInformationFirstMinimum,
        "autocorrelation" => embedding::DelayRule::AutocorrelationFirstZero,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown delay rule {other:?}; expected 'mutual_information' or \
                 'autocorrelation'"
            )))
        }
    };
    // A tenth of the series is the usual window: long enough to contain the
    // first minimum of a slowly varying signal, short enough that the estimate
    // at the far end still has most of the data behind it.
    let lag = max_lag.unwrap_or_else(|| (v.len() / 10).clamp(1, 1000));
    let chosen = embedding::select_delay(&v, parsed, lag, bins)
        .map_err(PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("delay", chosen.delay)?;
    dict.set_item(
        "rule",
        match chosen.rule {
            embedding::DelayRule::MutualInformationFirstMinimum => "mutual_information",
            embedding::DelayRule::AutocorrelationFirstZero => "autocorrelation",
        },
    )?;
    dict.set_item("converged", chosen.converged)?;
    dict.set_item("max_lag", lag)?;
    dict.set_item("curve", chosen.curve.into_pyarray(py).unbind())?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (x, max_lag, bins=None))]
fn mutual_information_curve(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    max_lag: usize,
    bins: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let v = as_1d(x)?;
    let curve = embedding::mutual_information_curve(&v, max_lag, bins)
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, curve).unbind())
}

#[pyfunction]
fn autocorrelation_curve(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    max_lag: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let v = as_1d(x)?;
    let curve =
        embedding::autocorrelation_curve(&v, max_lag).map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, curve).unbind())
}

#[pyfunction]
fn cao_e1_e2(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    m_max: usize,
    tau: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let v = as_1d(x)?;
    let (e1, e2) = embedding::cao_e1_e2(&v, m_max, tau).map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, e1).unbind(),
        PyArray1::from_vec(py, e2).unbind(),
    ))
}

//
// --------- Motif and network stats ----------
//

#[pyfunction]
fn triangles_per_node(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<Py<PyArray1<usize>>> {
    let e = as_edges(edges)?;
    Ok(PyArray1::from_vec(py, graphs::triangles_per_node(n, &e)).unbind())
}

#[pyfunction]
fn ego_edge_counts(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<Py<PyArray1<usize>>> {
    let e = as_edges(edges)?;
    Ok(PyArray1::from_vec(py, graphs::ego_edge_counts(n, &e)).unbind())
}

#[pyfunction]
fn core_numbers(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<Py<PyArray1<usize>>> {
    let e = as_edges(edges)?;
    Ok(PyArray1::from_vec(py, graphs::core_numbers(n, &e)).unbind())
}

#[pyfunction]
fn clustering_avg(_py: Python<'_>, n: usize, edges: PyReadonlyArray2<usize>) -> PyResult<f64> {
    let e = as_edges(edges)?;
    Ok(graphs::clustering_avg(n, &e))
}

#[pyfunction]
fn mean_shortest_path(_py: Python<'_>, n: usize, edges: PyReadonlyArray2<usize>) -> PyResult<f64> {
    let e = as_edges(edges)?;
    Ok(graphs::mean_shortest_path(n, &e))
}

//
// --------- Surrogates (phase, iAAFT) ----------
//

#[pyfunction]
fn surrogate_phase(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    let v = as_1d(x)?;
    Ok(PyArray1::from_vec(py, utils::surrogate_phase(&v, seed)).unbind())
}

#[pyfunction]
fn iaaft(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    iters: usize,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    let v = as_1d(x)?;
    Ok(PyArray1::from_vec(py, utils::iaaft(&v, iters, seed)).unbind())
}

#[pyfunction]
fn iaaft_legacy(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    iters: usize,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    let v = as_1d(x)?;
    Ok(PyArray1::from_vec(py, utils::iaaft_legacy(&v, iters, seed)).unbind())
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
    Ok(utils::corr_perm(&a, &b, n_perm, seed))
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
    let weights = as_2d(w)?;
    Ok(utils::moran_i(&x, &weights))
}

//
// --------- SINDy ----------
//

#[pyfunction]
#[pyo3(signature = (
    x,
    t,
    state_names,
    polynomial_degree=3,
    threshold=0.1,
    alpha=0.05,
    differentiation_order=2,
    max_iter=20,
))]
#[allow(clippy::too_many_arguments)]
fn fit_sindy_rust(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    t: PyReadonlyArray1<f64>,
    state_names: Vec<String>,
    polynomial_degree: usize,
    threshold: f64,
    alpha: f64,
    differentiation_order: usize,
    max_iter: usize,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let x_arr = as_2d(x)?;
    let t_arr = as_1d(t)?;
    let config = sindy::SindyConfig {
        polynomial_degree,
        threshold,
        alpha,
        differentiation_order,
        max_iter,
    };
    let fit =
        sindy::fit_single(&x_arr, &t_arr, None, &state_names, &config).map_err(PyValueError::new_err)?;
    Ok((fit.coefficients.into_pyarray(py).into(), fit.feature_names))
}

#[pyfunction]
#[pyo3(signature = (
    x_list,
    t_list,
    state_names,
    polynomial_degree=3,
    threshold=0.1,
    alpha=0.05,
    differentiation_order=2,
    max_iter=20,
))]
#[allow(clippy::too_many_arguments)]
fn fit_sindy_rust_multi(
    py: Python<'_>,
    x_list: Vec<PyReadonlyArray2<f64>>,
    t_list: Vec<PyReadonlyArray1<f64>>,
    state_names: Vec<String>,
    polynomial_degree: usize,
    threshold: f64,
    alpha: f64,
    differentiation_order: usize,
    max_iter: usize,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    if x_list.len() != t_list.len() {
        return Err(PyValueError::new_err(
            "x_list and t_list must have the same length",
        ));
    }
    if x_list.is_empty() {
        return Err(PyValueError::new_err("x_list must be non-empty"));
    }
    let mut trajectories = Vec::with_capacity(x_list.len());
    let mut times = Vec::with_capacity(t_list.len());
    for (x, t) in x_list.into_iter().zip(t_list.into_iter()) {
        trajectories.push(as_2d(x)?);
        times.push(as_1d(t)?);
    }
    let config = sindy::SindyConfig {
        polynomial_degree,
        threshold,
        alpha,
        differentiation_order,
        max_iter,
    };
    let fit = sindy::fit_many(&trajectories, &times, None, &state_names, &config)
        .map_err(PyValueError::new_err)?;
    Ok((fit.coefficients.into_pyarray(py).into(), fit.feature_names))
}

//
// --------- Python module ----------
//

#[pymodule]
fn ts2net_rs(m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hvg_edges, m)?)?;
    m.add_function(wrap_pyfunction!(hvg_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(nvg_edges_sweepline, m)?)?;
    m.add_function(wrap_pyfunction!(nvg_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(cdist_dtw, m)?)?;
    m.add_function(wrap_pyfunction!(cdist_dtw_rectangular, m)?)?;
    m.add_function(wrap_pyfunction!(knn, m)?)?;
    m.add_function(wrap_pyfunction!(radius, m)?)?;

    m.add_function(wrap_pyfunction!(rn_adj_epsilon, m)?)?;

    m.add_function(wrap_pyfunction!(event_sync, m)?)?;

    m.add_function(wrap_pyfunction!(false_nearest_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(cao_e1_e2, m)?)?;
    m.add_function(wrap_pyfunction!(select_delay, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_information_curve, m)?)?;
    m.add_function(wrap_pyfunction!(autocorrelation_curve, m)?)?;

    m.add_function(wrap_pyfunction!(triangles_per_node, m)?)?;
    m.add_function(wrap_pyfunction!(ego_edge_counts, m)?)?;
    m.add_function(wrap_pyfunction!(core_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(clustering_avg, m)?)?;
    m.add_function(wrap_pyfunction!(mean_shortest_path, m)?)?;

    m.add_function(wrap_pyfunction!(surrogate_phase, m)?)?;
    m.add_function(wrap_pyfunction!(iaaft, m)?)?;
    m.add_function(wrap_pyfunction!(iaaft_legacy, m)?)?;

    m.add_function(wrap_pyfunction!(corr_perm, m)?)?;

    m.add_function(wrap_pyfunction!(moran_i, m)?)?;

    m.add_function(wrap_pyfunction!(fit_sindy_rust, m)?)?;
    m.add_function(wrap_pyfunction!(fit_sindy_rust_multi, m)?)?;

    Ok(())
}
