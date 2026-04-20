//! Centrality measures beyond PageRank/betweenness.
//!
//! This module provides distance-based centralities (closeness, harmonic) and
//! the HITS hub/authority algorithm. All operate on the `Graph` trait.

use crate::graph::Graph;

/// Closeness centrality: the reciprocal of average shortest-path distance from a node
/// to all other reachable nodes.
///
/// Uses BFS (unweighted). Returns 0.0 for isolated nodes or nodes that cannot reach
/// any other node.
///
/// For disconnected graphs, only reachable nodes contribute to the average.
/// The score is normalized by `(reachable - 1) / (n - 1)` (Wasserman-Faust convention)
/// so that nodes in small components don't get inflated scores.
///
/// Time: O(V * (V + E)).
pub fn closeness_centrality<G: Graph>(graph: &G) -> Vec<f64> {
    let n = graph.node_count();
    let mut scores = vec![0.0f64; n];
    if n <= 1 {
        return scores;
    }

    for (v, score) in scores.iter_mut().enumerate() {
        let dist = bfs_distances(graph, v);
        let mut total_dist = 0u64;
        let mut reachable = 0usize;
        for (u, &d) in dist.iter().enumerate() {
            if u != v {
                if let Some(d) = d {
                    total_dist += d as u64;
                    reachable += 1;
                }
            }
        }
        if reachable > 0 && total_dist > 0 {
            let avg = total_dist as f64 / reachable as f64;
            // Wasserman-Faust normalization for disconnected graphs.
            let norm = reachable as f64 / (n - 1) as f64;
            *score = norm / avg;
        }
    }
    scores
}

/// Harmonic centrality: `sum of 1/d(v, u)` for all `u != v`.
///
/// Better behaved than closeness for disconnected graphs because unreachable
/// nodes contribute 0 (rather than making the metric undefined).
///
/// Normalized by `1 / (n - 1)` so the maximum is 1.0.
///
/// Time: O(V * (V + E)).
pub fn harmonic_centrality<G: Graph>(graph: &G) -> Vec<f64> {
    let n = graph.node_count();
    let mut scores = vec![0.0f64; n];
    if n <= 1 {
        return scores;
    }

    for (v, score) in scores.iter_mut().enumerate() {
        let dist = bfs_distances(graph, v);
        let mut sum_inv = 0.0f64;
        for (u, &d) in dist.iter().enumerate() {
            if u != v {
                if let Some(d) = d {
                    if d > 0 {
                        sum_inv += 1.0 / d as f64;
                    }
                }
            }
        }
        *score = sum_inv / (n - 1) as f64;
    }
    scores
}

/// BFS from a source node, returning distances. `None` = unreachable.
fn bfs_distances<G: Graph>(graph: &G, source: usize) -> Vec<Option<usize>> {
    let n = graph.node_count();
    let mut dist: Vec<Option<usize>> = vec![None; n];
    dist[source] = Some(0);
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);
    while let Some(u) = queue.pop_front() {
        let d = dist[u].unwrap();
        for v in graph.neighbors(u) {
            if v < n && dist[v].is_none() {
                dist[v] = Some(d + 1);
                queue.push_back(v);
            }
        }
    }
    dist
}

/// HITS (Hyperlink-Induced Topic Search) algorithm.
///
/// Computes hub and authority scores for each node.
/// - Authority score of v = sum of hub scores of in-neighbors of v.
/// - Hub score of v = sum of authority scores of out-neighbors of v.
///
/// The `neighbors` function is interpreted as **out-neighbors**.
/// The algorithm constructs an implicit transpose for the authority update.
///
/// Scores are L2-normalized after each iteration. Converges when the L2 change
/// in both vectors is below `tol`, or after `max_iter` iterations.
///
/// Returns `(hub_scores, authority_scores)`.
///
/// Time per iteration: O(V + E). Total: O(max_iter * (V + E)).
pub fn hits<G: Graph>(graph: &G, max_iter: usize, tol: f64) -> (Vec<f64>, Vec<f64>) {
    let n = graph.node_count();
    if n == 0 {
        return (vec![], vec![]);
    }

    // Build transpose (in-neighbors) once.
    let mut in_neighbors: Vec<Vec<usize>> = vec![vec![]; n];
    for u in 0..n {
        for v in graph.neighbors(u) {
            if v < n {
                in_neighbors[v].push(u);
            }
        }
    }

    let init = 1.0 / (n as f64).sqrt();
    let mut hub = vec![init; n];
    let mut auth = vec![init; n];

    for _ in 0..max_iter {
        // Authority update: auth[v] = sum of hub[u] for u in in-neighbors of v.
        let mut new_auth = vec![0.0f64; n];
        for v in 0..n {
            for &u in &in_neighbors[v] {
                new_auth[v] += hub[u];
            }
        }
        l2_normalize(&mut new_auth);

        // Hub update: hub[u] = sum of auth[v] for v in out-neighbors of u.
        let mut new_hub = vec![0.0f64; n];
        for (u, hub_u) in new_hub.iter_mut().enumerate() {
            for v in graph.neighbors(u) {
                if v < n {
                    *hub_u += new_auth[v];
                }
            }
        }
        l2_normalize(&mut new_hub);

        // Check convergence.
        let auth_diff = l2_diff(&auth, &new_auth);
        let hub_diff = l2_diff(&hub, &new_hub);
        auth = new_auth;
        hub = new_hub;
        if auth_diff < tol && hub_diff < tol {
            break;
        }
    }

    (hub, auth)
}

fn l2_normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn l2_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
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
    fn closeness_path_graph_center_highest() {
        // Undirected path: 0 -- 1 -- 2 -- 3 -- 4
        let g = G(vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]]);
        let c = closeness_centrality(&g);
        assert_eq!(c.len(), 5);
        // Node 2 (center) should have the highest closeness.
        for i in [0, 1, 3, 4] {
            assert!(
                c[2] > c[i],
                "center node 2 ({}) should beat node {i} ({})",
                c[2],
                c[i]
            );
        }
    }

    #[test]
    fn closeness_disconnected_node_is_zero() {
        // 0 -- 1, 2 is isolated
        let g = G(vec![vec![1], vec![0], vec![]]);
        let c = closeness_centrality(&g);
        assert_eq!(c[2], 0.0);
        assert!(c[0] > 0.0);
    }

    #[test]
    fn harmonic_disconnected_still_positive() {
        // 0 -- 1, 2 is isolated
        let g = G(vec![vec![1], vec![0], vec![]]);
        let h = harmonic_centrality(&g);
        assert!(h[0] > 0.0);
        assert!(h[1] > 0.0);
        assert_eq!(h[2], 0.0);
    }

    #[test]
    fn harmonic_path_center_highest() {
        let g = G(vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]]);
        let h = harmonic_centrality(&g);
        for i in [0, 1, 3, 4] {
            assert!(h[2] >= h[i], "center should be >= node {i}");
        }
    }

    #[test]
    fn hits_star_graph() {
        // Star: center 0 -> {1,2,3,4} (directed outward)
        let g = G(vec![vec![1, 2, 3, 4], vec![], vec![], vec![], vec![]]);
        let (hub, auth) = hits(&g, 100, 1e-10);
        assert_eq!(hub.len(), 5);
        // Node 0 should be the top hub.
        for i in 1..5 {
            assert!(
                hub[0] > hub[i],
                "node 0 should be top hub, got hub[0]={} hub[{i}]={}",
                hub[0],
                hub[i]
            );
        }
        // Leaves should be top authorities.
        for i in 1..5 {
            assert!(
                auth[i] > auth[0],
                "leaf {i} should have higher authority than center"
            );
        }
    }

    #[test]
    fn hits_empty_graph() {
        let g = G(vec![]);
        let (hub, auth) = hits(&g, 100, 1e-10);
        assert!(hub.is_empty());
        assert!(auth.is_empty());
    }
}
