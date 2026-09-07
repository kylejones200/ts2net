//! Structural statistics over an edge list.

use std::collections::VecDeque;

/// Build a sorted, de-duplicated adjacency list over `n` nodes.
///
/// Edges naming a node outside `0..n` are skipped rather than rejected, which
/// is the behaviour the Python bindings have always had.
pub fn build_adj(n: usize, edges: &[(usize, usize)], undirected: bool) -> Vec<Vec<usize>> {
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

/// Number of triangles incident on each node of the undirected graph.
///
/// Matches `networkx.triangles`. The edge sweep below visits each triangle at
/// a node once per incident edge, and a node lies on exactly two edges of
/// every triangle containing it, so the accumulated count is halved. That sum
/// is always even, making the division exact.
pub fn triangles_per_node(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let adj = build_adj(n, edges, true);
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
    for t in tri.iter_mut() {
        debug_assert!(*t % 2 == 0, "per-node triangle corners must be even");
        *t /= 2;
    }
    tri
}

/// Mean local clustering coefficient over nodes of degree at least two.
///
/// Returns `0.0` when no node qualifies.
pub fn clustering_avg(n: usize, edges: &[(usize, usize)]) -> f64 {
    let adj = build_adj(n, edges, true);
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
    if cnt > 0 {
        s / (cnt as f64)
    } else {
        0.0
    }
}

/// Mean shortest path length over all reachable node pairs.
///
/// Unreachable pairs are excluded rather than treated as infinite; returns
/// `NaN` when no pair is connected.
pub fn mean_shortest_path(n: usize, edges: &[(usize, usize)]) -> f64 {
    let adj = build_adj(n, edges, true);
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
    if pairs > 0 {
        (total as f64) / (pairs as f64)
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const TRIANGLE: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];

    #[test]
    fn build_adj_sorts_dedups_and_skips_out_of_range() {
        let adj = build_adj(3, &[(0, 1), (0, 1), (1, 0), (0, 9)], true);
        assert_eq!(adj[0], vec![1]);
        assert_eq!(adj[1], vec![0]);
        assert!(adj[2].is_empty());
    }

    #[test]
    fn directed_build_adj_records_one_direction() {
        let adj = build_adj(2, &[(0, 1)], false);
        assert_eq!(adj[0], vec![1]);
        assert!(adj[1].is_empty());
    }

    #[test]
    fn a_triangle_has_one_triangle_at_every_node() {
        assert_eq!(triangles_per_node(3, &TRIANGLE), vec![1, 1, 1]);
        assert_relative_eq!(clustering_avg(3, &TRIANGLE), 1.0);
        assert_relative_eq!(mean_shortest_path(3, &TRIANGLE), 1.0);
    }

    #[test]
    fn a_path_has_no_triangles_and_averages_over_reachable_pairs() {
        let path = [(0, 1), (1, 2)];
        assert_eq!(triangles_per_node(3, &path), vec![0, 0, 0]);
        assert_relative_eq!(clustering_avg(3, &path), 0.0);
        // Distances 1, 2, 1 over three pairs.
        assert_relative_eq!(mean_shortest_path(3, &path), 4.0 / 3.0);
    }

    #[test]
    fn a_triangle_sharing_an_edge_with_a_second_triangle() {
        // 0-1-2 and 0-2-3 share the edge 0-2. networkx.triangles gives
        // {0: 2, 1: 1, 2: 2, 3: 1}.
        let edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 0)];
        assert_eq!(triangles_per_node(4, &edges), vec![2, 1, 2, 1]);
    }

    #[test]
    fn a_four_clique_has_three_triangles_at_every_node() {
        let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        assert_eq!(triangles_per_node(4, &edges), vec![3, 3, 3, 3]);
    }

    #[test]
    fn disconnected_components_only_average_connected_pairs() {
        // Two isolated edges: both pairs are at distance 1, the cross pairs
        // are unreachable and excluded.
        assert_relative_eq!(mean_shortest_path(4, &[(0, 1), (2, 3)]), 1.0);
    }

    #[test]
    fn an_edgeless_graph_is_nan_and_zero() {
        assert!(mean_shortest_path(3, &[]).is_nan());
        assert_relative_eq!(clustering_avg(3, &[]), 0.0);
    }
}
