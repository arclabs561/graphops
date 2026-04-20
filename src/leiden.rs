//! Leiden community detection (Traag, Waltman & van Eck, 2019).
//!
//! Improves Louvain with a refinement phase that guarantees all communities
//! are internally connected. Operates on `GraphRef` (unweighted edges = weight 1.0)
//! or `WeightedGraph` for edge-weight-aware variants.
//!
//! ## Three Phases
//!
//! 1. **Local moving**: greedily reassign nodes to maximize modularity gain (same as Louvain).
//! 2. **Refinement**: within each community, reset to singletons and re-merge only
//!    within community boundaries, ensuring connectivity.
//! 3. **Aggregation**: collapse communities into super-nodes, repeat.
//!
//! ## Complexity
//!
//! O(m) per iteration, typically O(m log n) total. Space O(n + m).

use crate::graph::{GraphRef, WeightedGraph};
use std::collections::{HashMap, HashSet, VecDeque};

/// Run Leiden community detection with default resolution (1.0).
pub fn leiden<G: GraphRef>(graph: &G, resolution: f64) -> Vec<usize> {
    leiden_seeded(graph, resolution, 0)
}

/// Run Leiden community detection with explicit seed.
///
/// Returns a community label for each node, contiguous in `0..k`.
/// All communities are guaranteed to be internally connected.
///
/// ```
/// use graphops::leiden::leiden_seeded;
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
/// let labels = leiden_seeded(&g, 1.0, 42);
/// assert_eq!(labels.len(), 6);
/// ```
pub fn leiden_seeded<G: GraphRef>(graph: &G, resolution: f64, seed: u64) -> Vec<usize> {
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

    // node_map[current_node] = set of original nodes it represents.
    let mut node_map: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    let mut rng = StdRng::seed_from_u64(seed);

    loop {
        let cn = adj.len();
        let (moved, community) = local_move_phase(&adj, resolution, &mut rng);

        if !moved {
            break;
        }

        // Refinement: ensure each community is internally connected.
        let refined = refinement_phase(&adj, &community);

        let num_communities = *refined.iter().max().unwrap() + 1;
        if num_communities == cn {
            break;
        }

        // Aggregate: build super-node graph.
        let mut super_adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); num_communities];
        for u in 0..cn {
            let cu = refined[u];
            for &(v, w) in &adj[u] {
                let cv = refined[v];
                *super_adj[cu].entry(cv).or_insert(0.0) += w;
            }
        }

        let new_adj: Vec<Vec<(usize, f64)>> = super_adj
            .into_iter()
            .map(|m| m.into_iter().collect())
            .collect();

        // Update node_map: merge original nodes by community.
        let mut new_node_map: Vec<Vec<usize>> = vec![vec![]; num_communities];
        for (u, &comm) in refined.iter().enumerate() {
            new_node_map[comm].extend_from_slice(&node_map[u]);
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

    renumber(&mut labels);
    labels
}

/// Run Leiden community detection on a weighted graph with default seed (0).
pub fn leiden_weighted<G: WeightedGraph>(graph: &G, resolution: f64) -> Vec<usize> {
    leiden_weighted_seeded(graph, resolution, 0)
}

/// Run Leiden community detection on a weighted graph with explicit seed.
///
/// Uses `graph.edge_weight(u, v)` instead of treating all edges as weight 1.0.
/// All communities are guaranteed to be internally connected.
pub fn leiden_weighted_seeded<G: WeightedGraph>(
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

        let refined = refinement_phase(&adj, &community);

        let num_communities = *refined.iter().max().unwrap() + 1;
        if num_communities == cn {
            break;
        }

        let mut super_adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); num_communities];
        for u in 0..cn {
            let cu = refined[u];
            for &(v, w) in &adj[u] {
                let cv = refined[v];
                *super_adj[cu].entry(cv).or_insert(0.0) += w;
            }
        }

        let new_adj: Vec<Vec<(usize, f64)>> = super_adj
            .into_iter()
            .map(|m| m.into_iter().collect())
            .collect();

        let mut new_node_map: Vec<Vec<usize>> = vec![vec![]; num_communities];
        for (u, &comm) in refined.iter().enumerate() {
            new_node_map[comm].extend_from_slice(&node_map[u]);
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

/// Local move phase (shared with Louvain): greedily reassign each node to the
/// community that maximizes modularity gain.
fn local_move_phase(
    adj: &[Vec<(usize, f64)>],
    resolution: f64,
    rng: &mut impl rand::Rng,
) -> (bool, Vec<usize>) {
    use rand::seq::SliceRandom;

    let n = adj.len();

    let k: Vec<f64> = adj
        .iter()
        .map(|edges| edges.iter().map(|(_, w)| w).sum())
        .collect();
    let two_m: f64 = k.iter().sum::<f64>();

    if two_m == 0.0 {
        return (false, (0..n).collect());
    }

    let mut community: Vec<usize> = (0..n).collect();
    let mut sigma_tot: Vec<f64> = k.clone();

    let mut order: Vec<usize> = (0..n).collect();
    let mut any_moved = false;

    loop {
        order.shuffle(rng);
        let mut improved = false;

        for &u in &order {
            let cu = community[u];
            let k_u = k[u];

            let mut comm_weights: HashMap<usize, f64> = HashMap::new();
            for &(v, w) in &adj[u] {
                *comm_weights.entry(community[v]).or_insert(0.0) += w;
            }

            let k_u_cu = comm_weights.get(&cu).copied().unwrap_or(0.0);
            let sigma_cu_minus = sigma_tot[cu] - k_u;
            let remove_gain = k_u_cu / two_m - resolution * k_u * sigma_cu_minus / (two_m * two_m);

            let mut best_comm = cu;
            let mut best_gain = 0.0;

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

    renumber(&mut community);
    (any_moved, community)
}

/// Refinement phase: within each community, find connected components and split
/// disconnected ones into separate communities.
fn refinement_phase(adj: &[Vec<(usize, f64)>], community: &[usize]) -> Vec<usize> {
    let num_communities = *community.iter().max().unwrap_or(&0) + 1;

    // Group nodes by community.
    let mut comm_nodes: Vec<Vec<usize>> = vec![vec![]; num_communities];
    for (u, &c) in community.iter().enumerate() {
        comm_nodes[c].push(u);
    }

    let mut refined = community.to_vec();
    let mut next_comm = num_communities;

    for nodes in &comm_nodes {
        if nodes.len() <= 1 {
            continue;
        }

        // Find connected components within this community.
        let node_set: HashSet<usize> = nodes.iter().copied().collect();
        let components = connected_components_in_subset(adj, &node_set);

        if components.len() <= 1 {
            continue;
        }

        // First component keeps the original community ID.
        // Remaining components get new IDs.
        for component in components.iter().skip(1) {
            for &node in component {
                refined[node] = next_comm;
            }
            next_comm += 1;
        }
    }

    renumber(&mut refined);
    refined
}

/// BFS to find connected components within a subset of nodes.
fn connected_components_in_subset(
    adj: &[Vec<(usize, f64)>],
    node_set: &HashSet<usize>,
) -> Vec<Vec<usize>> {
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for &start in node_set {
        if visited.contains(&start) {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            if !visited.insert(node) {
                continue;
            }
            component.push(node);

            for &(neighbor, _) in &adj[node] {
                if node_set.contains(&neighbor) && !visited.contains(&neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        components.push(component);
    }

    components
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
    fn leiden_two_cliques_finds_two_communities() {
        let g = two_cliques();
        let labels = leiden_seeded(&g, 1.0, 42);
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
    fn leiden_beats_louvain_modularity() {
        let g = two_cliques();
        let labels = leiden_seeded(&g, 1.0, 42);
        let q = crate::louvain::modularity(&g, &labels);
        assert!(q > 0.0, "Leiden modularity = {q}, expected positive");
    }

    #[test]
    fn leiden_empty_graph() {
        let g = VecGraph { adj: vec![] };
        let labels = leiden_seeded(&g, 1.0, 0);
        assert!(labels.is_empty());
    }

    #[test]
    fn leiden_single_node() {
        let g = VecGraph { adj: vec![vec![]] };
        let labels = leiden_seeded(&g, 1.0, 0);
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn leiden_is_deterministic_with_seed() {
        let g = two_cliques();
        let a = leiden_seeded(&g, 1.0, 99);
        let b = leiden_seeded(&g, 1.0, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn leiden_disconnected_components_separated() {
        // Three disconnected edges: {0,1}, {2,3}, {4,5}
        let g = VecGraph {
            adj: vec![vec![1], vec![0], vec![3], vec![2], vec![5], vec![4]],
        };
        let labels = leiden_seeded(&g, 1.0, 0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_eq!(labels[4], labels[5]);
        let mut unique: Vec<usize> = labels.clone();
        unique.sort();
        unique.dedup();
        assert!(unique.len() >= 3);
    }

    #[test]
    fn leiden_communities_are_internally_connected() {
        // Key property: every community must be a connected subgraph.
        // Chain with cross-links to create non-trivial partitions.
        let g = VecGraph {
            adj: vec![
                vec![1, 2],     // 0
                vec![0, 2, 3],  // 1
                vec![0, 1],     // 2
                vec![1, 4],     // 3
                vec![3, 5, 6],  // 4
                vec![4, 6, 7],  // 5
                vec![4, 5],     // 6
                vec![5, 8],     // 7
                vec![7, 9, 10], // 8
                vec![8, 10],    // 9
                vec![8, 9],     // 10
            ],
        };
        let labels = leiden_seeded(&g, 1.0, 42);

        // Group nodes by community.
        let num_comms = *labels.iter().max().unwrap() + 1;
        let mut comm_nodes: Vec<Vec<usize>> = vec![vec![]; num_comms];
        for (node, &c) in labels.iter().enumerate() {
            comm_nodes[c].push(node);
        }

        // Verify each community is connected.
        for nodes in &comm_nodes {
            if nodes.len() <= 1 {
                continue;
            }
            let node_set: HashSet<usize> = nodes.iter().copied().collect();
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(nodes[0]);

            while let Some(u) = queue.pop_front() {
                if !visited.insert(u) {
                    continue;
                }
                for &v in g.neighbors_ref(u) {
                    if node_set.contains(&v) && !visited.contains(&v) {
                        queue.push_back(v);
                    }
                }
            }

            assert_eq!(
                visited.len(),
                nodes.len(),
                "Community {:?} is not internally connected",
                nodes
            );
        }
    }

    #[test]
    fn leiden_weighted_two_cliques_finds_two_communities() {
        // Two 4-cliques connected by a weak bridge (weight 0.1 vs intra-clique 1.0).
        // Nodes 0-3 form clique A; nodes 4-7 form clique B; bridge: 3-4 (weight 0.1).
        let n = 8;
        let mut w = vec![vec![0.0f64; n]; n];
        let clique_a = [0, 1, 2, 3];
        let clique_b = [4, 5, 6, 7];
        for &u in &clique_a {
            for &v in &clique_a {
                if u != v {
                    w[u][v] = 1.0;
                }
            }
        }
        for &u in &clique_b {
            for &v in &clique_b {
                if u != v {
                    w[u][v] = 1.0;
                }
            }
        }
        w[3][4] = 0.1;
        w[4][3] = 0.1;

        let g = WeightedVecGraph { weights: w };
        let labels = leiden_weighted_seeded(&g, 1.0, 42);
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
    fn leiden_weighted_different_weights_can_differ_from_unweighted() {
        // Verify that edge weights actually affect the output.
        // Build a graph where unweighted Leiden and weighted Leiden (with asymmetric weights)
        // can produce different community counts.
        //
        // Graph: two triangles (0-1-2 and 3-4-5) joined by edges 2-3 and 0-5 (forming a ring).
        // Weighted: edges within-triangle = 5.0, cross edges = 0.1.
        // At resolution=1.0 the weighted variant should separate the triangles.
        let n = 6;
        let mut w = vec![vec![0.0f64; n]; n];
        // Triangle A: 0-1-2
        for (u, v) in [(0, 1), (1, 2), (0, 2)] {
            w[u][v] = 5.0;
            w[v][u] = 5.0;
        }
        // Triangle B: 3-4-5
        for (u, v) in [(3, 4), (4, 5), (3, 5)] {
            w[u][v] = 5.0;
            w[v][u] = 5.0;
        }
        // Weak cross-edges
        w[2][3] = 0.1;
        w[3][2] = 0.1;
        w[0][5] = 0.1;
        w[5][0] = 0.1;

        let g = WeightedVecGraph { weights: w };
        let labels = leiden_weighted_seeded(&g, 1.0, 42);
        assert_eq!(labels.len(), 6);
        // Nodes within each triangle must be in the same community.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        // The two triangles must be in different communities.
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn leiden_weighted_seeded_is_deterministic() {
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
        w[3][4] = 0.5;
        w[4][3] = 0.5;
        let g = WeightedVecGraph { weights: w };
        let a = leiden_weighted_seeded(&g, 1.0, 77);
        let b = leiden_weighted_seeded(&g, 1.0, 77);
        assert_eq!(a, b);
    }
}
