//! Louvain community detection (Blondel et al., 2008).
//!
//! Greedily optimizes modularity via iterated local moves + graph aggregation.
//! Operates on `GraphRef` (unweighted edges treated as weight 1.0) or
//! `WeightedGraph` for edge-weight-aware variants.

use crate::graph::{GraphRef, WeightedGraph};
use std::collections::HashMap;

/// Compute modularity of a given partition on an unweighted graph.
///
/// Uses the community-level form:
/// Q = sum_c [L_c/m - (sigma_c/(2m))^2]
///
/// where L_c is internal edges in community c (each undirected edge counted
/// once), sigma_c is the sum of degrees, and m is the total edge count.
///
/// Returns 0.0 for graphs with no edges.
pub fn modularity<G: GraphRef>(graph: &G, communities: &[usize]) -> f64 {
    let n = graph.node_count();
    if n == 0 {
        return 0.0;
    }

    let two_m: f64 = (0..n)
        .map(|u| graph.neighbors_ref(u).len() as f64)
        .sum::<f64>();

    if two_m == 0.0 {
        return 0.0;
    }
    let m = two_m / 2.0;

    let num_communities = communities.iter().copied().max().unwrap_or(0) + 1;
    let mut sigma: Vec<f64> = vec![0.0; num_communities];
    let mut twice_l: Vec<f64> = vec![0.0; num_communities];

    for u in 0..n {
        let cu = communities[u];
        let k_u = graph.neighbors_ref(u).len() as f64;
        sigma[cu] += k_u;
        for &v in graph.neighbors_ref(u) {
            if v >= n {
                continue;
            }
            if communities[v] == cu {
                twice_l[cu] += 1.0;
            }
        }
    }

    let mut q = 0.0;
    for c in 0..num_communities {
        let l_c = twice_l[c] / 2.0;
        q += l_c / m - (sigma[c] / two_m).powi(2);
    }
    q
}

/// Louvain community detection with default resolution (1.0) and a seed for reproducibility.
///
/// Returns a community label for each node, contiguous in `0..k`.
pub fn louvain<G: GraphRef>(graph: &G, resolution: f64) -> Vec<usize> {
    louvain_seeded(graph, resolution, 0)
}

/// Louvain community detection with explicit seed for node visit order.
///
/// Two-phase algorithm:
/// 1. Local move phase: greedily reassign nodes to maximize modularity gain.
/// 2. Aggregation phase: collapse communities into super-nodes, repeat.
///
/// ```
/// use graphops::louvain::louvain_seeded;
/// use graphops::GraphRef;
///
/// struct G(Vec<Vec<usize>>);
/// impl GraphRef for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors_ref(&self, n: usize) -> &[usize] { &self.0[n] }
/// }
///
/// // Two triangles connected by one edge.
/// let g = G(vec![
///     vec![1, 2],    vec![0, 2],    vec![0, 1, 3],
///     vec![2, 4, 5], vec![3, 5],    vec![3, 4],
/// ]);
/// let labels = louvain_seeded(&g, 1.0, 42);
/// assert_eq!(labels.len(), 6);
/// ```
pub fn louvain_seeded<G: GraphRef>(graph: &G, resolution: f64, seed: u64) -> Vec<usize> {
    use rand::{rngs::StdRng, SeedableRng};

    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    // Build initial weighted adjacency (unweighted = all 1.0).
    let mut adj: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|u| {
            graph
                .neighbors_ref(u)
                .iter()
                .filter(|&&v| v < n)
                .map(|&v| (v, 1.0))
                .collect()
        })
        .collect();

    // node_to_original[current_node] = set of original nodes it represents.
    // We only track the flat mapping for the final output.
    let mut node_map: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    let mut rng = StdRng::seed_from_u64(seed);

    loop {
        let cn = adj.len();
        let (moved, community) = local_move_phase(&adj, resolution, &mut rng);

        if !moved {
            break;
        }

        // Determine number of communities.
        let num_communities = *community.iter().max().unwrap() + 1;
        if num_communities == cn {
            // No aggregation possible.
            break;
        }

        // Aggregate: build super-node graph.
        let mut super_adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); num_communities];
        for u in 0..cn {
            let cu = community[u];
            for &(v, w) in &adj[u] {
                let cv = community[v];
                if cu != cv {
                    *super_adj[cu].entry(cv).or_insert(0.0) += w;
                } else {
                    // Self-loop weight (internal edges).
                    *super_adj[cu].entry(cu).or_insert(0.0) += w;
                }
            }
        }

        let new_adj: Vec<Vec<(usize, f64)>> = super_adj
            .into_iter()
            .map(|m| m.into_iter().collect())
            .collect();

        // Update node_map: merge original nodes by community.
        let mut new_node_map: Vec<Vec<usize>> = vec![vec![]; num_communities];
        for (u, comm) in community.iter().enumerate() {
            new_node_map[*comm].extend_from_slice(&node_map[u]);
        }

        adj = new_adj;
        node_map = new_node_map;
    }

    // Build final labels from node_map.
    let mut labels = vec![0usize; n];
    for (comm, members) in node_map.iter().enumerate() {
        for &orig in members {
            labels[orig] = comm;
        }
    }

    // Renumber to contiguous 0..k.
    renumber(&mut labels);
    labels
}

/// Louvain community detection on a weighted graph with default seed (0).
pub fn louvain_weighted<G: WeightedGraph>(graph: &G, resolution: f64) -> Vec<usize> {
    louvain_weighted_seeded(graph, resolution, 0)
}

/// Louvain community detection on a weighted graph with explicit seed.
///
/// Uses `graph.edge_weight(u, v)` instead of treating all edges as weight 1.0.
pub fn louvain_weighted_seeded<G: WeightedGraph>(
    graph: &G,
    resolution: f64,
    seed: u64,
) -> Vec<usize> {
    use rand::{rngs::StdRng, SeedableRng};

    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    // Build initial weighted adjacency using actual edge weights.
    let mut adj: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|u| {
            graph
                .neighbors(u)
                .into_iter()
                .filter(|&v| v < n)
                .map(|v| (v, graph.edge_weight(u, v)))
                .collect()
        })
        .collect();

    let mut node_map: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    let mut rng = StdRng::seed_from_u64(seed);

    loop {
        let cn = adj.len();
        let (moved, community) = local_move_phase(&adj, resolution, &mut rng);

        if !moved {
            break;
        }

        let num_communities = *community.iter().max().unwrap() + 1;
        if num_communities == cn {
            break;
        }

        let mut super_adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); num_communities];
        for u in 0..cn {
            let cu = community[u];
            for &(v, w) in &adj[u] {
                let cv = community[v];
                if cu != cv {
                    *super_adj[cu].entry(cv).or_insert(0.0) += w;
                } else {
                    *super_adj[cu].entry(cu).or_insert(0.0) += w;
                }
            }
        }

        let new_adj: Vec<Vec<(usize, f64)>> = super_adj
            .into_iter()
            .map(|m| m.into_iter().collect())
            .collect();

        let mut new_node_map: Vec<Vec<usize>> = vec![vec![]; num_communities];
        for (u, comm) in community.iter().enumerate() {
            new_node_map[*comm].extend_from_slice(&node_map[u]);
        }

        adj = new_adj;
        node_map = new_node_map;
    }

    let mut labels = vec![0usize; n];
    for (comm, members) in node_map.iter().enumerate() {
        for &orig in members {
            labels[orig] = comm;
        }
    }

    renumber(&mut labels);
    labels
}

/// Renumber labels to contiguous 0..k in first-seen order.
fn renumber(labels: &mut [usize]) {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    for l in labels.iter_mut() {
        let id = *map.entry(*l).or_insert_with(|| {
            let cur = next;
            next += 1;
            cur
        });
        *l = id;
    }
}

/// Local move phase: greedily reassign each node to the community that
/// maximizes modularity gain. Returns (any_moved, community_assignment).
fn local_move_phase(
    adj: &[Vec<(usize, f64)>],
    resolution: f64,
    rng: &mut impl rand::Rng,
) -> (bool, Vec<usize>) {
    use rand::seq::SliceRandom;

    let n = adj.len();

    // Degree (sum of edge weights) for each node.
    let k: Vec<f64> = adj
        .iter()
        .map(|edges| edges.iter().map(|(_, w)| w).sum())
        .collect();
    let two_m: f64 = k.iter().sum::<f64>();

    if two_m == 0.0 {
        return (false, (0..n).collect());
    }

    let mut community: Vec<usize> = (0..n).collect();
    // Sum of degrees in each community.
    let mut sigma_tot: Vec<f64> = k.clone();

    let mut order: Vec<usize> = (0..n).collect();
    let mut any_moved = false;

    loop {
        order.shuffle(rng);
        let mut improved = false;

        for &u in &order {
            let cu = community[u];
            let k_u = k[u];

            // Compute weights from u to each neighboring community.
            let mut comm_weights: HashMap<usize, f64> = HashMap::new();
            for &(v, w) in &adj[u] {
                *comm_weights.entry(community[v]).or_insert(0.0) += w;
            }

            // Modularity gain of removing u from cu.
            let k_u_cu = comm_weights.get(&cu).copied().unwrap_or(0.0);
            let sigma_cu_minus = sigma_tot[cu] - k_u;
            let remove_gain = k_u_cu / two_m - resolution * k_u * sigma_cu_minus / (two_m * two_m);

            // Find the best community to move u to.
            let mut best_comm = cu;
            let mut best_gain = 0.0; // net gain must be positive to move

            for (&c, &k_u_c) in &comm_weights {
                if c == cu {
                    continue;
                }
                let sigma_c = sigma_tot[c];
                let add_gain = k_u_c / two_m - resolution * k_u * sigma_c / (two_m * two_m);
                let net = add_gain - remove_gain;
                if net > best_gain || (net == best_gain && c < best_comm) {
                    best_gain = net;
                    best_comm = c;
                }
            }

            if best_comm != cu {
                // Move u from cu to best_comm.
                sigma_tot[cu] -= k_u;
                sigma_tot[best_comm] += k_u;
                community[u] = best_comm;
                improved = true;
                any_moved = true;
            }
        }

        if !improved {
            break;
        }
    }

    // Renumber communities to contiguous 0..k.
    renumber(&mut community);
    (any_moved, community)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::graph::{Graph, GraphRef, WeightedGraph};

    struct VecGraph {
        adj: Vec<Vec<usize>>,
    }

    impl GraphRef for VecGraph {
        fn node_count(&self) -> usize {
            self.adj.len()
        }
        fn neighbors_ref(&self, node: usize) -> &[usize] {
            &self.adj[node]
        }
    }

    /// Weighted graph backed by a dense adjacency matrix (0.0 = no edge).
    struct WeightedVecGraph {
        weights: Vec<Vec<f64>>,
    }

    impl Graph for WeightedVecGraph {
        fn node_count(&self) -> usize {
            self.weights.len()
        }
        fn neighbors(&self, node: usize) -> Vec<usize> {
            self.weights[node]
                .iter()
                .enumerate()
                .filter(|(_, &w)| w > 0.0)
                .map(|(i, _)| i)
                .collect()
        }
    }

    impl WeightedGraph for WeightedVecGraph {
        fn edge_weight(&self, source: usize, target: usize) -> f64 {
            self.weights[source][target]
        }
    }

    fn two_cliques() -> VecGraph {
        // Clique A: {0,1,2,3}, Clique B: {4,5,6,7}, bridge: 3-4
        VecGraph {
            adj: vec![
                vec![1, 2, 3],
                vec![0, 2, 3],
                vec![0, 1, 3],
                vec![0, 1, 2, 4],
                vec![3, 5, 6, 7],
                vec![4, 6, 7],
                vec![4, 5, 7],
                vec![4, 5, 6],
            ],
        }
    }

    #[test]
    fn louvain_two_cliques_finds_two_communities() {
        let g = two_cliques();
        let labels = louvain_seeded(&g, 1.0, 42);
        assert_eq!(labels.len(), 8);
        // Nodes in the same clique should share a label.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);
        // The two cliques should be in different communities.
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn louvain_modularity_positive_for_nontrivial_partition() {
        let g = two_cliques();
        let labels = louvain_seeded(&g, 1.0, 42);
        let q = modularity(&g, &labels);
        assert!(q > 0.0, "modularity = {q}, expected positive");
    }

    #[test]
    fn louvain_beats_random_partition() {
        let g = two_cliques();
        let good = louvain_seeded(&g, 1.0, 42);
        let q_good = modularity(&g, &good);

        // Random partition: even nodes = 0, odd nodes = 1.
        let random: Vec<usize> = (0..8).map(|i| i % 2).collect();
        let q_random = modularity(&g, &random);

        assert!(
            q_good > q_random,
            "louvain Q={q_good} should beat random Q={q_random}"
        );
    }

    #[test]
    fn modularity_all_one_community_is_zero() {
        let g = two_cliques();
        let labels = vec![0; 8];
        let q = modularity(&g, &labels);
        assert!(
            q.abs() < 1e-12,
            "all-one-community modularity = {q}, expected 0"
        );
    }

    #[test]
    fn louvain_empty_graph() {
        let g = VecGraph { adj: vec![] };
        let labels = louvain_seeded(&g, 1.0, 0);
        assert!(labels.is_empty());
    }

    #[test]
    fn louvain_single_node() {
        let g = VecGraph { adj: vec![vec![]] };
        let labels = louvain_seeded(&g, 1.0, 0);
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn louvain_is_deterministic_with_seed() {
        let g = two_cliques();
        let a = louvain_seeded(&g, 1.0, 99);
        let b = louvain_seeded(&g, 1.0, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn louvain_disconnected_components() {
        // Three disconnected edges: {0,1}, {2,3}, {4,5}
        let g = VecGraph {
            adj: vec![vec![1], vec![0], vec![3], vec![2], vec![5], vec![4]],
        };
        let labels = louvain_seeded(&g, 1.0, 0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_eq!(labels[4], labels[5]);
        // At least 3 distinct communities.
        let mut unique: Vec<usize> = labels.clone();
        unique.sort();
        unique.dedup();
        assert!(unique.len() >= 3);
    }

    #[test]
    fn louvain_weighted_two_cliques_finds_two_communities() {
        // Two 4-cliques (weight 1.0) connected by a weak bridge (weight 0.1).
        let n = 8;
        let mut w = vec![vec![0.0f64; n]; n];
        for u in 0..4usize {
            for v in 0..4usize {
                if u != v {
                    w[u][v] = 1.0;
                }
            }
        }
        for u in 4..8usize {
            for v in 4..8usize {
                if u != v {
                    w[u][v] = 1.0;
                }
            }
        }
        w[3][4] = 0.1;
        w[4][3] = 0.1;

        let g = WeightedVecGraph { weights: w };
        let labels = louvain_weighted_seeded(&g, 1.0, 42);
        assert_eq!(labels.len(), 8);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn louvain_weighted_respects_edge_weights() {
        // Star graph with 4 leaves split into two pairs.
        // Pair A (0,1) connected to center (2) with weight 1.0.
        // Pair B (3,4) connected to center (2) with weight 0.01.
        // Also: 0-1 edge (weight 1.0), 3-4 edge (weight 1.0).
        // At high resolution, center should join the heavier pair (A).
        let n = 5;
        let mut w = vec![vec![0.0f64; n]; n];
        w[0][1] = 1.0;
        w[1][0] = 1.0;
        w[0][2] = 1.0;
        w[2][0] = 1.0;
        w[1][2] = 1.0;
        w[2][1] = 1.0;
        w[3][4] = 1.0;
        w[4][3] = 1.0;
        w[3][2] = 0.01;
        w[2][3] = 0.01;
        w[4][2] = 0.01;
        w[2][4] = 0.01;

        let g = WeightedVecGraph { weights: w };
        let labels = louvain_weighted_seeded(&g, 1.0, 42);
        assert_eq!(labels.len(), 5);
        // Nodes 0,1,2 should be in same community; 3,4 in another.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn louvain_weighted_seeded_is_deterministic() {
        let n = 8;
        let mut w = vec![vec![0.0f64; n]; n];
        for u in 0..4usize {
            for v in 0..4usize {
                if u != v {
                    w[u][v] = 1.5;
                }
            }
        }
        for u in 4..8usize {
            for v in 4..8usize {
                if u != v {
                    w[u][v] = 1.5;
                }
            }
        }
        w[3][4] = 0.2;
        w[4][3] = 0.2;
        let g = WeightedVecGraph { weights: w };
        let a = louvain_weighted_seeded(&g, 1.0, 77);
        let b = louvain_weighted_seeded(&g, 1.0, 77);
        assert_eq!(a, b);
    }
}
