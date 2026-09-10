//! k-d tree neighbour queries over embedded points.

use kiddo::{KdTree, SquaredEuclidean};
use ndarray::{Array2, Axis};

/// Largest embedding dimension the k-d tree backend is instantiated for.
///
/// `kiddo` takes the dimension as a const generic, so each width has to be
/// monomorphised explicitly. The limit is therefore an implementation choice,
/// not a mathematical one, and it used to sit at 6 -- below the dimension real
/// signals frequently reconstruct into, and below what convergent cross
/// mapping needs, since that is nearest-neighbour search in a shadow manifold.
/// Raised to 16, which covers the reconstructions this library produces; the
/// cost is compile time and binary size for the unused widths.
pub const MAX_KDTREE_DIM: usize = 16;

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
        let v: Vec<usize> = res
            .iter()
            .filter_map(|neighbor| {
                let j_usize = neighbor.item as usize;
                if j_usize != i {
                    Some(j_usize)
                } else {
                    None
                }
            })
            .collect();
        out.push(v);
    }
    out
}

/// `k` nearest neighbours of every row of `pts`, excluding the point itself.
///
/// Returns `(indices, distances)`, both shaped `[n, k]`. Errors when the
/// column count exceeds [`MAX_KDTREE_DIM`].
/// Dispatch a call on the runtime dimension to its monomorphised instance.
///
/// One arm per supported width, because the width is a const generic. Written
/// as a macro so `knn` and `radius` cannot drift apart, which is how a ceiling
/// ends up applying to one entry point and not the other.
macro_rules! dispatch_on_dimension {
    ($dim:expr, $call:ident, $($arg:expr),* $(,)?) => {
        match $dim {
            1 => Ok($call::<1>($($arg),*)),
            2 => Ok($call::<2>($($arg),*)),
            3 => Ok($call::<3>($($arg),*)),
            4 => Ok($call::<4>($($arg),*)),
            5 => Ok($call::<5>($($arg),*)),
            6 => Ok($call::<6>($($arg),*)),
            7 => Ok($call::<7>($($arg),*)),
            8 => Ok($call::<8>($($arg),*)),
            9 => Ok($call::<9>($($arg),*)),
            10 => Ok($call::<10>($($arg),*)),
            11 => Ok($call::<11>($($arg),*)),
            12 => Ok($call::<12>($($arg),*)),
            13 => Ok($call::<13>($($arg),*)),
            14 => Ok($call::<14>($($arg),*)),
            15 => Ok($call::<15>($($arg),*)),
            16 => Ok($call::<16>($($arg),*)),
            other => Err(format!(
                "dimension up to {MAX_KDTREE_DIM} is supported, got {other}"
            )),
        }
    };
}

pub fn knn(pts: &Array2<f64>, k: usize) -> Result<(Array2<usize>, Array2<f64>), String> {
    dispatch_on_dimension!(pts.len_of(Axis(1)), knn_impl, pts, k)
}

/// Indices of all neighbours within `eps` of each row of `pts`, self excluded.
///
/// Errors when the column count exceeds [`MAX_KDTREE_DIM`].
pub fn radius(pts: &Array2<f64>, eps: f64) -> Result<Vec<Vec<usize>>, String> {
    dispatch_on_dimension!(pts.len_of(Axis(1)), radius_impl, pts, eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn knn_finds_the_adjacent_point_on_a_line() {
        let pts = array![[0.0], [1.0], [10.0]];
        let (idx, dst) = knn(&pts, 1).unwrap();
        assert_eq!(idx[[0, 0]], 1);
        assert_eq!(idx[[1, 0]], 0);
        assert_eq!(idx[[2, 0]], 1);
        assert!((dst[[0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn radius_excludes_self_and_respects_eps() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]];
        let n = radius(&pts, 2.0).unwrap();
        assert_eq!(n[0], vec![1]);
        assert_eq!(n[1], vec![0]);
        assert!(n[2].is_empty());
    }

    #[test]
    fn dimensions_above_the_limit_are_rejected() {
        let pts = Array2::<f64>::zeros((3, MAX_KDTREE_DIM + 1));
        assert!(knn(&pts, 1).is_err());
        assert!(radius(&pts, 1.0).is_err());
    }

    #[test]
    fn every_supported_dimension_is_actually_dispatched() {
        // The ceiling and the dispatch table must agree. A mismatch would
        // reject a width the constant advertises, or advertise one the table
        // cannot serve.
        for dim in 1..=MAX_KDTREE_DIM {
            let pts = Array2::<f64>::from_shape_fn((5, dim), |(i, j)| {
                (i * dim + j) as f64
            });
            assert!(knn(&pts, 2).is_ok(), "knn failed at dimension {dim}");
            assert!(radius(&pts, 1e6).is_ok(), "radius failed at dimension {dim}");
        }
    }

    #[test]
    fn the_ceiling_covers_a_typical_reconstruction() {
        // Lorenz needs 3, but real signals reconstruct higher; 6 was below
        // what this library's own embedding routines can produce.
        assert!(MAX_KDTREE_DIM >= 10);
    }
}
