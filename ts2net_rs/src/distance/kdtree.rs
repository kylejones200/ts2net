//! k-d tree neighbour queries over embedded points.

use kiddo::{KdTree, SquaredEuclidean};
use ndarray::{Array2, Axis};

/// Largest embedding dimension the k-d tree backend is instantiated for.
///
/// `kiddo` takes the dimension as a const generic, so each width has to be
/// monomorphised explicitly; widths above this fall back to the caller.
pub const MAX_KDTREE_DIM: usize = 6;

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
pub fn knn(pts: &Array2<f64>, k: usize) -> Result<(Array2<usize>, Array2<f64>), String> {
    match pts.len_of(Axis(1)) {
        1 => Ok(knn_impl::<1>(pts, k)),
        2 => Ok(knn_impl::<2>(pts, k)),
        3 => Ok(knn_impl::<3>(pts, k)),
        4 => Ok(knn_impl::<4>(pts, k)),
        5 => Ok(knn_impl::<5>(pts, k)),
        6 => Ok(knn_impl::<6>(pts, k)),
        _ => Err(format!(
            "dimension up to {MAX_KDTREE_DIM} is supported"
        )),
    }
}

/// Indices of all neighbours within `eps` of each row of `pts`, self excluded.
///
/// Errors when the column count exceeds [`MAX_KDTREE_DIM`].
pub fn radius(pts: &Array2<f64>, eps: f64) -> Result<Vec<Vec<usize>>, String> {
    match pts.len_of(Axis(1)) {
        1 => Ok(radius_impl::<1>(pts, eps)),
        2 => Ok(radius_impl::<2>(pts, eps)),
        3 => Ok(radius_impl::<3>(pts, eps)),
        4 => Ok(radius_impl::<4>(pts, eps)),
        5 => Ok(radius_impl::<5>(pts, eps)),
        6 => Ok(radius_impl::<6>(pts, eps)),
        _ => Err(format!(
            "dimension up to {MAX_KDTREE_DIM} is supported"
        )),
    }
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
}
