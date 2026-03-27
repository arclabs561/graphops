//! Triangle counting and clustering coefficients.
//!
//! Operates on **undirected** graphs via the `Graph` trait. For directed graphs,
//! the results are only meaningful if the adapter exposes symmetric edges.

use crate::graph::Graph;

/// Count the number of triangles in an undirected graph.
///
/// Uses the sorted-adjacency intersection approach: for each edge (u, v) where u < v,
/// count common neighbors w > v. This avoids triple-counting.
///
/// Time: O(m * sqrt(m)) where m = number of edges.
///
/// ```
/// use graphops::graph::Graph;
/// use graphops::triangle::triangle_count;
///
/// struct G(Vec<Vec<usize>>);
/// impl Graph for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors(&self, n: usize) -> Vec<usize> { self.0[n].clone() }
/// }
///
/// // K3: 0-1-2 (complete triangle)
/// let g = G(vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
/// assert_eq!(triangle_count(&g), 1);
/// ```
pub fn triangle_count<G: Graph>(graph: &G) -> usize {
    let n = graph.node_count();

    // Build sorted adjacency sets for fast intersection.
    let adj: Vec<Vec<usize>> = (0..n)
        .map(|u| {
            let mut nbrs = graph.neighbors(u);
            nbrs.retain(|&v| v < n);
            nbrs.sort_unstable();
            nbrs.dedup();
            nbrs
        })
        .collect();

    let mut count = 0usize;
    for u in 0..n {
        for &v in &adj[u] {
            if v <= u {
                continue;
            }
            // Count common neighbors w > v.
            count += sorted_intersection_count_above(&adj[u], &adj[v], v);
        }
    }
    count
}

/// Count elements that appear in both sorted slices and are > threshold.
fn sorted_intersection_count_above(a: &[usize], b: &[usize], threshold: usize) -> usize {
    let mut i = a.partition_point(|&x| x <= threshold);
    let mut j = b.partition_point(|&x| x <= threshold);
    let mut count = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

/// Per-node triangle counts (each triangle counted once per participating node).
fn node_triangle_counts<G: Graph>(graph: &G) -> Vec<usize> {
    let n = graph.node_count();
    let adj: Vec<Vec<usize>> = (0..n)
        .map(|u| {
            let mut nbrs = graph.neighbors(u);
            nbrs.retain(|&v| v < n);
            nbrs.sort_unstable();
            nbrs.dedup();
            nbrs
        })
        .collect();

    let mut tri = vec![0usize; n];
    for u in 0..n {
        for &v in &adj[u] {
            if v <= u {
                continue;
            }
            let common = sorted_intersection_count_above(&adj[u], &adj[v], v);
            // This triangle involves u, v, and each common neighbor w.
            // We're iterating with u < v < w, so credit each once here
            // and aggregate per-node after.
            // Actually, count per-node: each triangle (u,v,w) credits all three.
            tri[u] += common;
            tri[v] += common;
            // For w, we need to iterate:
            let mut i = adj[u].partition_point(|&x| x <= v);
            let mut j = adj[v].partition_point(|&x| x <= v);
            while i < adj[u].len() && j < adj[v].len() {
                match adj[u][i].cmp(&adj[v][j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        tri[adj[u][i]] += 1;
                        i += 1;
                        j += 1;
                    }
                }
            }
        }
    }
    tri
}

/// Local clustering coefficient for each node.
///
/// `C(v) = 2 * triangles(v) / (deg(v) * (deg(v) - 1))`, or 0.0 if `deg < 2`.
///
/// ```
/// use graphops::graph::Graph;
/// use graphops::triangle::clustering_coefficients;
///
/// struct G(Vec<Vec<usize>>);
/// impl Graph for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors(&self, n: usize) -> Vec<usize> { self.0[n].clone() }
/// }
///
/// // K4: every node has clustering coefficient 1.0
/// let g = G(vec![
///     vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2],
/// ]);
/// let cc = clustering_coefficients(&g);
/// for &c in &cc {
///     assert!((c - 1.0).abs() < 1e-12);
/// }
/// ```
pub fn clustering_coefficients<G: Graph>(graph: &G) -> Vec<f64> {
    let n = graph.node_count();
    let tri = node_triangle_counts(graph);

    (0..n)
        .map(|u| {
            let mut nbrs = graph.neighbors(u);
            nbrs.retain(|&v| v < n);
            nbrs.sort_unstable();
            nbrs.dedup();
            let deg = nbrs.len();
            if deg < 2 {
                0.0
            } else {
                2.0 * tri[u] as f64 / (deg * (deg - 1)) as f64
            }
        })
        .collect()
}

/// Global clustering coefficient: `3 * triangles / connected_triples`.
///
/// A connected triple is a path of length 2 (three nodes, two edges).
/// Returns 0.0 if there are no connected triples.
pub fn global_clustering_coefficient<G: Graph>(graph: &G) -> f64 {
    let n = graph.node_count();
    let triangles = triangle_count(graph);

    let mut triples = 0usize;
    for u in 0..n {
        let mut nbrs = graph.neighbors(u);
        nbrs.retain(|&v| v < n);
        nbrs.sort_unstable();
        nbrs.dedup();
        let d = nbrs.len();
        if d >= 2 {
            triples += d * (d - 1) / 2;
        }
    }

    if triples == 0 {
        0.0
    } else {
        3.0 * triangles as f64 / triples as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct G(Vec<Vec<usize>>);
    impl Graph for G {
        fn node_count(&self) -> usize {
            self.0.len()
        }
        fn neighbors(&self, n: usize) -> Vec<usize> {
            self.0[n].clone()
        }
    }

    #[test]
    fn triangle_count_k4() {
        // K4 has C(4,3) = 4 triangles.
        let g = G(vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ]);
        assert_eq!(triangle_count(&g), 4);
    }

    #[test]
    fn triangle_count_k3() {
        let g = G(vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
        assert_eq!(triangle_count(&g), 1);
    }

    #[test]
    fn triangle_count_path_is_zero() {
        // Path 0-1-2: no triangles.
        let g = G(vec![vec![1], vec![0, 2], vec![1]]);
        assert_eq!(triangle_count(&g), 0);
    }

    #[test]
    fn triangle_count_empty() {
        let g = G(vec![]);
        assert_eq!(triangle_count(&g), 0);
    }

    #[test]
    fn clustering_coefficients_k4_all_one() {
        let g = G(vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ]);
        let cc = clustering_coefficients(&g);
        for &c in &cc {
            assert!((c - 1.0).abs() < 1e-12, "expected 1.0, got {c}");
        }
    }

    #[test]
    fn clustering_coefficient_star_is_zero() {
        // Star: center 0, leaves 1..4. No triangles.
        let g = G(vec![vec![1, 2, 3, 4], vec![0], vec![0], vec![0], vec![0]]);
        let cc = clustering_coefficients(&g);
        // Center has deg 4, but no triangles.
        assert_eq!(cc[0], 0.0);
        // Leaves have deg 1, so coefficient is 0 by definition.
        for i in 1..5 {
            assert_eq!(cc[i], 0.0);
        }
    }

    #[test]
    fn global_clustering_k4() {
        let g = G(vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ]);
        let gc = global_clustering_coefficient(&g);
        assert!(
            (gc - 1.0).abs() < 1e-12,
            "K4 global clustering should be 1.0, got {gc}"
        );
    }

    #[test]
    fn global_clustering_path_is_zero() {
        let g = G(vec![vec![1], vec![0, 2], vec![1]]);
        assert_eq!(global_clustering_coefficient(&g), 0.0);
    }
}
