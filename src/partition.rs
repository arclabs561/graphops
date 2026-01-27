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

use crate::graph::GraphRef;

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

    #[test]
    fn connected_components_two_components() {
        // 0-1-2 and 3-4
        let g = VecGraph {
            adj: vec![
                vec![1],
                vec![0, 2],
                vec![1],
                vec![4],
                vec![3],
            ],
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
        assert_eq!(a.len(), g.node_count());
    }
}

