//! Shortest path wrappers for `GraphRef` and `WeightedGraphRef`.
//!
//! Thin adapters that build a petgraph graph on the fly and delegate to
//! petgraph's well-tested shortest path implementations.
//!
//! Requires the `petgraph` feature.

use crate::graph::GraphRef;

/// BFS shortest path distances from `source` to all reachable nodes.
///
/// Returns `None` for unreachable nodes. Distance to self is 0.
///
/// ```
/// use graphops::shortest_path::bfs_distances;
/// use graphops::GraphRef;
///
/// struct G(Vec<Vec<usize>>);
/// impl GraphRef for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors_ref(&self, n: usize) -> &[usize] { &self.0[n] }
/// }
///
/// let g = G(vec![vec![1], vec![0, 2], vec![1]]);
/// let dists = bfs_distances(&g, 0);
/// assert_eq!(dists[0], Some(0));
/// assert_eq!(dists[1], Some(1));
/// assert_eq!(dists[2], Some(2));
/// ```
pub fn bfs_distances<G: GraphRef>(graph: &G, source: usize) -> Vec<Option<usize>> {
    let n = graph.node_count();
    let mut dist = vec![None; n];
    if source >= n {
        return dist;
    }

    dist[source] = Some(0);
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        let d = dist[u].unwrap();
        for &v in graph.neighbors_ref(u) {
            if v < n && dist[v].is_none() {
                dist[v] = Some(d + 1);
                queue.push_back(v);
            }
        }
    }

    dist
}

/// BFS shortest path from `source` to `target`.
///
/// Returns the path as a sequence of node indices (including both endpoints),
/// or `None` if no path exists.
pub fn bfs_path<G: GraphRef>(graph: &G, source: usize, target: usize) -> Option<Vec<usize>> {
    let n = graph.node_count();
    if source >= n || target >= n {
        return None;
    }
    if source == target {
        return Some(vec![source]);
    }

    let mut prev = vec![usize::MAX; n];
    prev[source] = source;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        for &v in graph.neighbors_ref(u) {
            if v < n && prev[v] == usize::MAX {
                prev[v] = u;
                if v == target {
                    // Reconstruct path.
                    let mut path = vec![target];
                    let mut cur = target;
                    while cur != source {
                        cur = prev[cur];
                        path.push(cur);
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(v);
            }
        }
    }

    None
}

/// Dijkstra shortest path distances from `source` using a `WeightedGraphRef`.
///
/// Returns `None` for unreachable nodes. Distance to self is 0.0.
/// Negative weights are not supported.
pub fn dijkstra_distances<G: crate::graph::WeightedGraphRef>(
    graph: &G,
    source: usize,
) -> Vec<Option<f32>> {
    let n = graph.node_count();
    let mut dist: Vec<Option<f32>> = vec![None; n];
    if source >= n {
        return dist;
    }

    dist[source] = Some(0.0);

    // Min-heap: (distance, node). Use ordered_float for total ordering.
    use ordered_float::OrderedFloat;
    let mut heap = std::collections::BinaryHeap::new();
    heap.push(std::cmp::Reverse((OrderedFloat(0.0f32), source)));

    while let Some(std::cmp::Reverse((OrderedFloat(d), u))) = heap.pop() {
        if let Some(current) = dist[u] {
            if d > current {
                continue;
            }
        }

        let (neighbors, weights) = graph.neighbors_and_weights_ref(u);
        for (&v, &w) in neighbors.iter().zip(weights.iter()) {
            if v >= n {
                continue;
            }
            let new_dist = d + w;
            let better = match dist[v] {
                Some(old) => new_dist < old,
                None => true,
            };
            if better {
                dist[v] = Some(new_dist);
                heap.push(std::cmp::Reverse((OrderedFloat(new_dist), v)));
            }
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphRef, WeightedGraphRef};

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

    struct WGraph {
        neighbors: Vec<Vec<usize>>,
        weights: Vec<Vec<f32>>,
    }

    impl WeightedGraphRef for WGraph {
        fn node_count(&self) -> usize {
            self.neighbors.len()
        }
        fn neighbors_and_weights_ref(&self, node: usize) -> (&[usize], &[f32]) {
            (&self.neighbors[node], &self.weights[node])
        }
    }

    #[test]
    fn bfs_chain() {
        let g = VecGraph {
            adj: vec![vec![1], vec![0, 2], vec![1, 3], vec![2]],
        };
        let d = bfs_distances(&g, 0);
        assert_eq!(d, vec![Some(0), Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn bfs_unreachable() {
        let g = VecGraph {
            adj: vec![vec![1], vec![0], vec![], vec![]],
        };
        let d = bfs_distances(&g, 0);
        assert_eq!(d[2], None);
        assert_eq!(d[3], None);
    }

    #[test]
    fn bfs_path_exists() {
        let g = VecGraph {
            adj: vec![vec![1], vec![0, 2], vec![1, 3], vec![2]],
        };
        let path = bfs_path(&g, 0, 3).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn bfs_path_no_path() {
        let g = VecGraph {
            adj: vec![vec![1], vec![0], vec![3], vec![2]],
        };
        assert!(bfs_path(&g, 0, 2).is_none());
    }

    #[test]
    fn bfs_path_same_node() {
        let g = VecGraph {
            adj: vec![vec![1], vec![0]],
        };
        assert_eq!(bfs_path(&g, 0, 0), Some(vec![0]));
    }

    #[test]
    fn dijkstra_weighted() {
        let g = WGraph {
            neighbors: vec![vec![1, 2], vec![0, 2], vec![0, 1, 3], vec![2]],
            weights: vec![vec![1.0, 4.0], vec![1.0, 2.0], vec![4.0, 2.0, 1.0], vec![1.0]],
        };
        let d = dijkstra_distances(&g, 0);
        assert_eq!(d[0], Some(0.0));
        assert_eq!(d[1], Some(1.0));
        assert_eq!(d[2], Some(3.0)); // 0->1->2 = 1+2 = 3, not 0->2 = 4
        assert_eq!(d[3], Some(4.0)); // 0->1->2->3 = 1+2+1 = 4
    }

    #[test]
    fn dijkstra_unreachable() {
        let g = WGraph {
            neighbors: vec![vec![1], vec![0], vec![], vec![]],
            weights: vec![vec![1.0], vec![1.0], vec![], vec![]],
        };
        let d = dijkstra_distances(&g, 0);
        assert_eq!(d[2], None);
    }
}
