//! Graph partitioning / community detection.
//!
//! This module is for algorithms whose **primary output** is a partition of nodes:
//! `labels[i] = community_id`.
//!
//! Design goals (aligned with `graphops`):
//! - **Adapter-first**: operate on `graphops::GraphRef` to avoid per-step allocations.
//! - **Determinism hooks**: algorithms that require randomness accept an explicit seed.
//! - **Minimal deps**: no `petgraph` requirement (though `petgraph` can implement the adapter traits).
//!
//! Invariants:
//! - Output length equals `graph.node_count()`.
//! - Labels are contiguous in `0..k` (renumbered).

use crate::graph::{Graph, GraphRef};

/// Renumber arbitrary labels to `0..k` in first-seen order.
fn renumber(labels: &mut [usize]) -> usize {
    use std::collections::HashMap;
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
    next
}

/// Connected components of an **undirected** graph, using BFS.
///
/// For directed graphs: this computes weakly-connected components if the adapter’s
/// neighbor lists include both in- and out-neighbors; otherwise it follows the adapter.
///
/// ```
/// use graphops::{connected_components, GraphRef};
///
/// struct G(Vec<Vec<usize>>);
/// impl GraphRef for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors_ref(&self, n: usize) -> &[usize] { &self.0[n] }
/// }
///
/// // Two components: {0,1,2} and {3,4}
/// let g = G(vec![vec![1], vec![0, 2], vec![1], vec![4], vec![3]]);
/// let labels = connected_components(&g);
/// assert_eq!(labels[0], labels[1]);
/// assert_ne!(labels[0], labels[3]);
/// ```
pub fn connected_components<G: GraphRef>(graph: &G) -> Vec<usize> {
    let n = graph.node_count();
    let mut labels = vec![usize::MAX; n];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    let mut comp = 0usize;
    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        labels[start] = comp;
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for &v in graph.neighbors_ref(u) {
                if v >= n {
                    // Adapter contract violation; ignore out-of-range neighbor.
                    continue;
                }
                if labels[v] == usize::MAX {
                    labels[v] = comp;
                    queue.push_back(v);
                }
            }
        }
        comp += 1;
    }

    // already contiguous by construction: 0..comp
    labels
}

/// Label propagation community detection (Raghavan et al., 2007).
///
/// Very fast approximate partitioning: iteratively set each node’s label to the most
/// frequent label among its neighbors. This is **seeded** for deterministic tie-breaking.
///
/// Notes:
/// - The algorithm is defined on undirected graphs; on directed graphs behavior depends on adapter.
/// - For isolated nodes, the label remains its own id.
///
/// ```
/// use graphops::{label_propagation, GraphRef};
///
/// struct G(Vec<Vec<usize>>);
/// impl GraphRef for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors_ref(&self, n: usize) -> &[usize] { &self.0[n] }
/// }
///
/// // Two cliques connected by one edge
/// let g = G(vec![
///     vec![1, 2],    vec![0, 2],    vec![0, 1, 3],
///     vec![2, 4, 5], vec![3, 5],    vec![3, 4],
/// ]);
/// let labels = label_propagation(&g, 50, 42);
/// assert_eq!(labels.len(), 6);
/// // Deterministic: same seed -> same result
/// assert_eq!(labels, label_propagation(&g, 50, 42));
/// ```
pub fn label_propagation<G: GraphRef>(graph: &G, max_iters: usize, seed: u64) -> Vec<usize> {
    use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
    use std::collections::HashMap;

    let n = graph.node_count();
    let mut labels: Vec<usize> = (0..n).collect();
    if n == 0 {
        return labels;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..n).collect();

    for _ in 0..max_iters {
        order.shuffle(&mut rng);
        let mut changed = false;

        for &u in &order {
            let neigh = graph.neighbors_ref(u);
            if neigh.is_empty() {
                continue;
            }

            let mut counts: HashMap<usize, usize> = HashMap::new();
            for &v in neigh {
                if v >= n {
                    continue;
                }
                *counts.entry(labels[v]).or_insert(0) += 1;
            }
            if counts.is_empty() {
                continue;
            }

            // Pick the label with highest count; tie-break by smallest label (deterministic).
            let mut best_label = labels[u];
            let mut best_count = 0usize;
            for (&lbl, &c) in &counts {
                if c > best_count || (c == best_count && lbl < best_label) {
                    best_label = lbl;
                    best_count = c;
                }
            }

            if best_label != labels[u] {
                labels[u] = best_label;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    let _k = renumber(&mut labels);
    labels
}

/// Tarjan's algorithm for strongly connected components.
///
/// Returns a component label for each node (nodes in the same SCC share a label).
/// Labels are assigned in reverse topological order of the condensation DAG:
/// component 0 is a sink SCC, the last component is a source SCC.
///
/// The `neighbors` function of the `Graph` trait is interpreted as **out-neighbors**
/// (directed edges). For undirected graphs, each SCC is a connected component.
///
/// Time: O(V + E), single DFS pass.
///
/// ```
/// use graphops::graph::Graph;
/// use graphops::partition::strongly_connected_components;
///
/// struct G(Vec<Vec<usize>>);
/// impl Graph for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors(&self, n: usize) -> Vec<usize> { self.0[n].clone() }
/// }
///
/// // A single 3-cycle: 0 -> 1 -> 2 -> 0
/// let g = G(vec![vec![1], vec![2], vec![0]]);
/// let labels = strongly_connected_components(&g);
/// assert_eq!(labels[0], labels[1]);
/// assert_eq!(labels[1], labels[2]);
/// ```
pub fn strongly_connected_components<G: Graph>(graph: &G) -> Vec<usize> {
    let n = graph.node_count();
    let mut index_counter: usize = 0;
    let mut stack: Vec<usize> = Vec::new();
    let mut on_stack = vec![false; n];
    let mut index: Vec<Option<usize>> = vec![None; n];
    let mut lowlink = vec![0usize; n];
    let mut labels = vec![usize::MAX; n];
    let mut comp = 0usize;

    // Iterative Tarjan to avoid stack overflow on large graphs.
    // Each frame holds (node, neighbor_iterator_position).
    for root in 0..n {
        if index[root].is_some() {
            continue;
        }

        // DFS call stack: (node, next_neighbor_index)
        let mut dfs_stack: Vec<(usize, usize)> = vec![(root, 0)];
        index[root] = Some(index_counter);
        lowlink[root] = index_counter;
        index_counter += 1;
        stack.push(root);
        on_stack[root] = true;

        while let Some(&mut (u, ref mut ni)) = dfs_stack.last_mut() {
            let neighbors = graph.neighbors(u);
            if *ni < neighbors.len() {
                let w = neighbors[*ni];
                *ni += 1;
                if w >= n {
                    continue;
                }
                if index[w].is_none() {
                    // Tree edge: push w
                    index[w] = Some(index_counter);
                    lowlink[w] = index_counter;
                    index_counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    dfs_stack.push((w, 0));
                } else if on_stack[w] {
                    lowlink[u] = lowlink[u].min(index[w].unwrap());
                }
            } else {
                // All neighbors processed for u.
                if lowlink[u] == index[u].unwrap() {
                    // u is root of an SCC.
                    while let Some(v) = stack.pop() {
                        on_stack[v] = false;
                        labels[v] = comp;
                        if v == u {
                            break;
                        }
                    }
                    comp += 1;
                }
                let finished_u = u;
                let finished_lowlink = lowlink[finished_u];
                dfs_stack.pop();
                if let Some(&(parent, _)) = dfs_stack.last() {
                    lowlink[parent] = lowlink[parent].min(finished_lowlink);
                }
            }
        }
    }

    labels
}

/// Topological sort via Kahn's algorithm (BFS-based).
///
/// Returns node indices in topological order, or `None` if the graph contains a cycle.
/// The `neighbors` function is interpreted as **out-neighbors** (directed edges).
///
/// Time: O(V + E).
///
/// ```
/// use graphops::graph::Graph;
/// use graphops::partition::topological_sort;
///
/// struct G(Vec<Vec<usize>>);
/// impl Graph for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors(&self, n: usize) -> Vec<usize> { self.0[n].clone() }
/// }
///
/// // DAG: 0 -> 1 -> 2, 0 -> 2
/// let g = G(vec![vec![1, 2], vec![2], vec![]]);
/// let order = topological_sort(&g).unwrap();
/// assert_eq!(order[0], 0);
/// assert_eq!(order.len(), 3);
/// ```
pub fn topological_sort<G: Graph>(graph: &G) -> Option<Vec<usize>> {
    let n = graph.node_count();
    let mut in_degree = vec![0usize; n];

    for u in 0..n {
        for v in graph.neighbors(u) {
            if v < n {
                in_degree[v] += 1;
            }
        }
    }

    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for (u, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(u);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for v in graph.neighbors(u) {
            if v < n {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
    }

    if order.len() == n {
        Some(order)
    } else {
        None // cycle detected
    }
}

/// Compute the core number of each node via the peeling algorithm.
///
/// The k-core of a graph is the maximal subgraph where every node has degree >= k.
/// The core number of node v is the largest k such that v belongs to the k-core.
///
/// Runs in O(V + E) using bucket-sort ordering.
///
/// ```
/// use graphops::{core_numbers, GraphRef};
///
/// struct G(Vec<Vec<usize>>);
/// impl GraphRef for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors_ref(&self, n: usize) -> &[usize] { &self.0[n] }
/// }
///
/// // Complete graph K4: every node has core number 3.
/// let g = G(vec![vec![1,2,3], vec![0,2,3], vec![0,1,3], vec![0,1,2]]);
/// assert_eq!(core_numbers(&g), vec![3, 3, 3, 3]);
/// ```
pub fn core_numbers<G: GraphRef>(graph: &G) -> Vec<usize> {
    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    let deg: Vec<usize> = (0..n)
        .map(|u| graph.neighbors_ref(u).iter().filter(|&&v| v < n).count())
        .collect();
    let max_deg = deg.iter().copied().max().unwrap_or(0);

    // Bucket sort: bin_start[d] = starting index for degree-d nodes in `vert`.
    let mut bin_count: Vec<usize> = vec![0; max_deg + 2];
    for &d in &deg {
        bin_count[d] += 1;
    }
    let mut bin_start: Vec<usize> = vec![0; max_deg + 2];
    {
        let mut cumul = 0;
        for d in 0..=max_deg {
            bin_start[d] = cumul;
            cumul += bin_count[d];
        }
        bin_start[max_deg + 1] = cumul;
    }

    let mut pos: Vec<usize> = vec![0; n];
    let mut vert: Vec<usize> = vec![0; n];

    // Place nodes into sorted order using a copy of bin_start.
    let mut bin_cur = bin_start.clone();
    for u in 0..n {
        let d = deg[u];
        pos[u] = bin_cur[d];
        vert[bin_cur[d]] = u;
        bin_cur[d] += 1;
    }

    let mut core = deg.clone();

    for i in 0..n {
        let u = vert[i];
        for &v in graph.neighbors_ref(u) {
            if v >= n {
                continue;
            }
            if core[v] > core[u] {
                let dv = core[v];
                let pw = bin_start[dv]; // first position in bucket dv
                let pv = pos[v];
                if pv != pw {
                    let w = vert[pw];
                    vert[pv] = w;
                    vert[pw] = v;
                    pos[w] = pv;
                    pos[v] = pw;
                }
                bin_start[dv] += 1;
                core[v] -= 1;
            }
        }
    }

    core
}

/// Extract the k-core subgraph: returns indices of nodes with core number >= k.
///
/// ```
/// use graphops::{k_core, GraphRef};
///
/// struct G(Vec<Vec<usize>>);
/// impl GraphRef for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors_ref(&self, n: usize) -> &[usize] { &self.0[n] }
/// }
///
/// // Star graph: center=0, leaves=1..4. All core numbers are 1.
/// let g = G(vec![vec![1,2,3,4], vec![0], vec![0], vec![0], vec![0]]);
/// let mut nodes = k_core(&g, 1);
/// nodes.sort();
/// assert_eq!(nodes, vec![0, 1, 2, 3, 4]);
/// assert!(k_core(&g, 2).is_empty());
/// ```
pub fn k_core<G: GraphRef>(graph: &G, k: usize) -> Vec<usize> {
    let cores = core_numbers(graph);
    cores
        .iter()
        .enumerate()
        .filter(|(_, &c)| c >= k)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    impl Graph for VecGraph {
        fn node_count(&self) -> usize {
            self.adj.len()
        }
        fn neighbors(&self, node: usize) -> Vec<usize> {
            self.adj[node].clone()
        }
    }

    #[test]
    fn connected_components_two_components() {
        // 0-1-2 and 3-4
        let g = VecGraph {
            adj: vec![vec![1], vec![0, 2], vec![1], vec![4], vec![3]],
        };
        let labels = connected_components(&g);
        assert_eq!(labels.len(), 5);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn label_propagation_is_deterministic_with_seed() {
        // Two cliques loosely connected.
        let g = VecGraph {
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
        };
        let a = label_propagation(&g, 50, 123);
        let b = label_propagation(&g, 50, 123);
        assert_eq!(a, b);
        assert_eq!(a.len(), GraphRef::node_count(&g));
    }

    // --- SCC tests ---

    #[test]
    fn scc_dag_each_node_own_component() {
        // DAG: 0 -> 1 -> 2 -> 3
        let g = VecGraph {
            adj: vec![vec![1], vec![2], vec![3], vec![]],
        };
        let labels = strongly_connected_components(&g);
        assert_eq!(labels.len(), 4);
        // All labels must be distinct.
        let mut unique: Vec<usize> = labels.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn scc_single_cycle_one_component() {
        // 0 -> 1 -> 2 -> 0
        let g = VecGraph {
            adj: vec![vec![1], vec![2], vec![0]],
        };
        let labels = strongly_connected_components(&g);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn scc_two_cycles_with_bridge() {
        // Cycle A: 0 -> 1 -> 0
        // Cycle B: 2 -> 3 -> 2
        // Bridge: 1 -> 2 (one-way)
        let g = VecGraph {
            adj: vec![vec![1], vec![0, 2], vec![3], vec![2]],
        };
        let labels = strongly_connected_components(&g);
        assert_eq!(labels[0], labels[1]); // same SCC
        assert_eq!(labels[2], labels[3]); // same SCC
        assert_ne!(labels[0], labels[2]); // different SCCs
    }

    #[test]
    fn scc_empty_graph() {
        let g = VecGraph { adj: vec![] };
        let labels = strongly_connected_components(&g);
        assert!(labels.is_empty());
    }

    // --- Topological sort tests ---

    #[test]
    fn topo_sort_dag_valid_ordering() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let g = VecGraph {
            adj: vec![vec![1, 2], vec![3], vec![3], vec![]],
        };
        let order = topological_sort(&g).unwrap();
        assert_eq!(order.len(), 4);
        // Build position map and verify all edges go forward.
        let mut pos = [0usize; 4];
        for (i, &node) in order.iter().enumerate() {
            pos[node] = i;
        }
        for u in 0..4 {
            for &v in &g.adj[u] {
                assert!(pos[u] < pos[v], "edge {u}->{v} violates topo order");
            }
        }
    }

    #[test]
    fn topo_sort_cycle_returns_none() {
        // 0 -> 1 -> 2 -> 0
        let g = VecGraph {
            adj: vec![vec![1], vec![2], vec![0]],
        };
        assert!(topological_sort(&g).is_none());
    }

    #[test]
    fn topo_sort_single_node() {
        let g = VecGraph { adj: vec![vec![]] };
        assert_eq!(topological_sort(&g), Some(vec![0]));
    }

    #[test]
    fn topo_sort_empty_graph() {
        let g = VecGraph { adj: vec![] };
        assert_eq!(topological_sort(&g), Some(vec![]));
    }

    // --- k-core tests ---

    #[test]
    fn core_numbers_complete_k5() {
        // K5: all pairs connected. Core number = 4 for every node.
        let g = VecGraph {
            adj: vec![
                vec![1, 2, 3, 4],
                vec![0, 2, 3, 4],
                vec![0, 1, 3, 4],
                vec![0, 1, 2, 4],
                vec![0, 1, 2, 3],
            ],
        };
        assert_eq!(core_numbers(&g), vec![4, 4, 4, 4, 4]);
    }

    #[test]
    fn core_numbers_star() {
        // Star: center=0, leaves=1..4. Every node has core 1.
        let g = VecGraph {
            adj: vec![vec![1, 2, 3, 4], vec![0], vec![0], vec![0], vec![0]],
        };
        assert_eq!(core_numbers(&g), vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn core_numbers_path() {
        // Path 0-1-2-3. All core 1.
        let g = VecGraph {
            adj: vec![vec![1], vec![0, 2], vec![1, 3], vec![2]],
        };
        assert_eq!(core_numbers(&g), vec![1, 1, 1, 1]);
    }

    #[test]
    fn k_core_zero_returns_all() {
        let g = VecGraph {
            adj: vec![vec![1], vec![0]],
        };
        let mut nodes = k_core(&g, 0);
        nodes.sort();
        assert_eq!(nodes, vec![0, 1]);
    }

    #[test]
    fn core_numbers_empty_graph() {
        let g = VecGraph { adj: vec![] };
        assert_eq!(core_numbers(&g), Vec::<usize>::new());
    }

    #[test]
    fn core_numbers_isolated_nodes() {
        // Three isolated nodes: core number 0.
        let g = VecGraph {
            adj: vec![vec![], vec![], vec![]],
        };
        assert_eq!(core_numbers(&g), vec![0, 0, 0]);
    }
}
