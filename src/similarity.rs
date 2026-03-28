//! Node similarity based on neighborhood overlap.
//!
//! Three standard measures:
//! - **Jaccard**: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
//! - **Cosine** (Salton): |N(u) ∩ N(v)| / sqrt(|N(u)| * |N(v)|)
//! - **Overlap**: |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)

use crate::graph::GraphRef;

/// Jaccard similarity between two nodes based on neighbor sets.
///
/// Returns 0.0 if both nodes have no neighbors.
pub fn jaccard<G: GraphRef>(graph: &G, u: usize, v: usize) -> f64 {
    let nu = graph.neighbors_ref(u);
    let nv = graph.neighbors_ref(v);
    let intersection = count_intersection(nu, nv);
    let union = nu.len() + nv.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Cosine (Salton) similarity between two nodes.
///
/// Returns 0.0 if either node has no neighbors.
pub fn cosine<G: GraphRef>(graph: &G, u: usize, v: usize) -> f64 {
    let nu = graph.neighbors_ref(u);
    let nv = graph.neighbors_ref(v);
    let denom = ((nu.len() * nv.len()) as f64).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        count_intersection(nu, nv) as f64 / denom
    }
}

/// Overlap coefficient between two nodes.
///
/// Returns 0.0 if either node has no neighbors.
pub fn overlap<G: GraphRef>(graph: &G, u: usize, v: usize) -> f64 {
    let nu = graph.neighbors_ref(u);
    let nv = graph.neighbors_ref(v);
    let min_size = nu.len().min(nv.len());
    if min_size == 0 {
        0.0
    } else {
        count_intersection(nu, nv) as f64 / min_size as f64
    }
}

/// Compute top-k most similar nodes to `u` by Jaccard similarity.
///
/// Returns pairs of (node, similarity) sorted by descending similarity.
pub fn top_k_similar_jaccard<G: GraphRef>(graph: &G, u: usize, k: usize) -> Vec<(usize, f64)> {
    let n = graph.node_count();
    let mut scores: Vec<(usize, f64)> = (0..n)
        .filter(|&v| v != u)
        .map(|v| (v, jaccard(graph, u, v)))
        .filter(|&(_, s)| s > 0.0)
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores
}

/// Count intersection of two sorted or unsorted neighbor slices.
///
/// Uses a hash set for O(n + m) when slices are large enough.
fn count_intersection(a: &[usize], b: &[usize]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    let (smaller, larger) = if a.len() <= b.len() {
        (a, b)
    } else {
        (b, a)
    };

    // For small sets, brute force. For larger, use hash.
    if smaller.len() <= 16 {
        smaller
            .iter()
            .filter(|x| larger.contains(x))
            .count()
    } else {
        let set: std::collections::HashSet<usize> = smaller.iter().copied().collect();
        larger.iter().filter(|x| set.contains(x)).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphRef;

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
    fn jaccard_identical_neighborhoods() {
        // Triangle: all pairs share both other neighbors.
        let g = VecGraph {
            adj: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
        };
        let j = jaccard(&g, 0, 1);
        // N(0) = {1,2}, N(1) = {0,2}, intersection = {2}, union = {0,1,2}
        assert!((j - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_no_overlap() {
        let g = VecGraph {
            adj: vec![vec![1], vec![0], vec![3], vec![2]],
        };
        assert_eq!(jaccard(&g, 0, 2), 0.0);
    }

    #[test]
    fn cosine_star() {
        // Star: center 0 connected to 1,2,3
        let g = VecGraph {
            adj: vec![vec![1, 2, 3], vec![0], vec![0], vec![0]],
        };
        // N(1) = {0}, N(2) = {0}, intersection = {0}, |N1|=1, |N2|=1
        let c = cosine(&g, 1, 2);
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn overlap_subset() {
        // Node 0 neighbors: {1,2,3}, Node 1 neighbors: {0,2}
        // Intersection: {2}, min_size = 2, overlap = 1/2
        let g = VecGraph {
            adj: vec![vec![1, 2, 3], vec![0, 2], vec![0, 1], vec![0]],
        };
        let o = overlap(&g, 0, 1);
        assert!((o - 0.5).abs() < 1e-10);
    }

    #[test]
    fn top_k_returns_sorted() {
        let g = VecGraph {
            adj: vec![
                vec![1, 2, 3],
                vec![0, 2, 3],
                vec![0, 1],
                vec![0, 1],
            ],
        };
        let top = top_k_similar_jaccard(&g, 0, 2);
        assert!(top.len() <= 2);
        if top.len() >= 2 {
            assert!(top[0].1 >= top[1].1);
        }
    }

    #[test]
    fn isolated_nodes_zero() {
        let g = VecGraph {
            adj: vec![vec![], vec![]],
        };
        assert_eq!(jaccard(&g, 0, 1), 0.0);
        assert_eq!(cosine(&g, 0, 1), 0.0);
        assert_eq!(overlap(&g, 0, 1), 0.0);
    }
}
