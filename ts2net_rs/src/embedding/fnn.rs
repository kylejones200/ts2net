//! Embedding-dimension selection from a scalar series.

use ndarray::{s, Array1, Array2};

/// Delay-embed `v` into `dim` columns of length `l` with lag `tau`.
fn delay_embed(v: &Array1<f64>, dim: usize, tau: usize, l: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((l, dim));
    for i in 0..dim {
        out.slice_mut(s![.., i])
            .assign(&v.slice(s![i * tau..i * tau + l]));
    }
    out
}

/// Euclidean nearest neighbour of row `i`, as `(distance, index)`.
fn nearest_neighbor(points: &Array2<f64>, i: usize) -> (f64, usize) {
    let l = points.nrows();
    let ai = points.row(i);
    let mut best = (f64::INFINITY, 0usize);
    for j in 0..l {
        if i == j {
            continue;
        }
        let d: f64 = ai
            .iter()
            .zip(points.row(j).iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt();
        if d < best.0 {
            best = (d, j);
        }
    }
    best
}

/// False-nearest-neighbour fraction for embedding dimensions `1..m_max`.
///
/// A neighbour is false when unfolding into one more dimension separates it by
/// more than `rtol` relative to its current distance, or by more than `atol` in
/// absolute terms. Dimensions with fewer than two embedded points score 1.0.
pub fn false_nearest_neighbors(
    v: &Array1<f64>,
    m_max: usize,
    tau: usize,
    rtol: f64,
    atol: f64,
) -> Result<Vec<f64>, String> {
    let n = v.len();
    if m_max < 2 {
        return Err("m_max >= 2".to_string());
    }
    let mut out = Vec::<f64>::with_capacity(m_max - 1);
    for m in 1..m_max {
        // `n - m * tau` underflows once the embedding outruns the series.
        let l = match n.checked_sub(m * tau) {
            Some(l) if l >= 2 => l,
            _ => {
                out.push(1.0);
                continue;
            }
        };
        let xm = delay_embed(v, m, tau, l);
        let xp = delay_embed(v, m + 1, tau, l);
        let mut fnn = 0.0;
        for i in 0..l {
            let best = nearest_neighbor(&xm, i);
            let j = best.1;
            let num = (xp[[i, m]] - xp[[j, m]]).abs();
            if best.0 == 0.0 || num.is_nan() {
                fnn += 1.0;
                continue;
            }
            if num / best.0 > rtol || num > atol {
                fnn += 1.0;
            }
        }
        out.push(fnn / (l as f64));
    }
    Ok(out)
}

/// Cao's E1 and E2 statistics for embedding dimensions `1..m_max`.
///
/// E1 saturates at the minimum sufficient embedding dimension; E2 stays near
/// 1.0 for stochastic data and departs from it for deterministic data. E1 has
/// `m_max - 1` entries and E2 has `m_max - 2`. Dimensions with fewer than two
/// embedded points score NaN.
pub fn cao_e1_e2(v: &Array1<f64>, m_max: usize, tau: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n = v.len();
    if m_max < 2 {
        // Without this the `m_max - 2` capacity below underflows.
        return Err("m_max >= 2".to_string());
    }
    let mut e1 = Vec::<f64>::with_capacity(m_max - 1);
    let mut e2 = Vec::<f64>::with_capacity(m_max - 2);
    for m in 1..m_max {
        let l = match n.checked_sub(m * tau) {
            Some(l) if l >= 2 => l,
            _ => {
                e1.push(f64::NAN);
                if m > 1 {
                    e2.push(f64::NAN);
                }
                continue;
            }
        };
        let xm = delay_embed(v, m, tau, l);
        let xp = delay_embed(v, m + 1, tau, l);
        let mut ratios = Vec::<f64>::with_capacity(l);
        let mut diffs = Vec::<f64>::with_capacity(l);
        for i in 0..l {
            let best = nearest_neighbor(&xm, i);
            let j = best.1;
            let num: f64 = xp
                .row(i)
                .iter()
                .zip(xp.row(j).iter())
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
                .sqrt();
            let den = best.0.max(1e-12);
            ratios.push(num / den);
            if m > 1 {
                diffs.push((xp[[i, m]] - xp[[j, m]]).abs());
            }
        }
        e1.push(ratios.iter().sum::<f64>() / (ratios.len() as f64));
        if m > 1 {
            e2.push(diffs.iter().sum::<f64>() / (diffs.len() as f64));
        }
    }
    Ok((e1, e2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sine(n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|i| (i as f64 * 0.3).sin()))
    }

    #[test]
    fn m_max_below_two_is_rejected_rather_than_panicking() {
        let v = sine(64);
        assert!(false_nearest_neighbors(&v, 1, 1, 10.0, 2.0).is_err());
        assert!(cao_e1_e2(&v, 1, 1).is_err());
        assert!(cao_e1_e2(&v, 0, 1).is_err());
    }

    #[test]
    fn output_lengths_follow_m_max() {
        let v = sine(200);
        let fnn = false_nearest_neighbors(&v, 5, 3, 10.0, 2.0).unwrap();
        assert_eq!(fnn.len(), 4);
        let (e1, e2) = cao_e1_e2(&v, 5, 3).unwrap();
        assert_eq!(e1.len(), 4);
        assert_eq!(e2.len(), 3);
    }

    #[test]
    fn fnn_fractions_stay_in_the_unit_interval() {
        let v = sine(200);
        for f in false_nearest_neighbors(&v, 5, 3, 10.0, 2.0).unwrap() {
            assert!((0.0..=1.0).contains(&f), "fraction out of range: {f}");
        }
    }

    #[test]
    fn a_sine_unfolds_by_dimension_two() {
        // A clean sine is a closed curve in 2-D, so almost no neighbour at
        // m = 2 is false.
        let v = sine(300);
        let fnn = false_nearest_neighbors(&v, 4, 5, 10.0, 2.0).unwrap();
        assert!(fnn[1] < 0.1, "expected few false neighbours, got {}", fnn[1]);
    }

    #[test]
    fn an_embedding_longer_than_the_series_degrades_instead_of_overflowing() {
        let v = array![1.0, 2.0, 3.0, 4.0];
        let fnn = false_nearest_neighbors(&v, 4, 10, 10.0, 2.0).unwrap();
        assert_eq!(fnn, vec![1.0, 1.0, 1.0]);
        let (e1, e2) = cao_e1_e2(&v, 4, 10).unwrap();
        assert!(e1.iter().all(|x| x.is_nan()));
        assert!(e2.iter().all(|x| x.is_nan()));
    }
}
