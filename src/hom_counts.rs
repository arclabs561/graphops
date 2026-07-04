//! Rooted homomorphism-count profiles: walk and closed-walk counts.
//!
//! A homomorphism count `hom(F, G)` is the number of (not necessarily
//! injective) maps from a pattern `F` into `G`. Rooted at a node, the two
//! tractable families are exactly walks: the rooted-path count
//! `hom(P_k, G, v)` is the number of length-`k` walks starting at `v`, and
//! the rooted-cycle count `hom(C_k, G, v)` is the number of length-`k`
//! closed walks at `v` (the diagonal of the adjacency power `A^k`).
//!
//! Why these matter as *features*: Dell, Grohe & Rattan (ICALP 2018,
//! arXiv:1802.08876) show `k`-WL indistinguishability equals equality of
//! homomorphism counts from all patterns of treewidth ≤ `k` — for `k = 1`
//! (message passing / color refinement), from all *trees*. So closed-walk
//! counts (cycles, treewidth 2) carry structure that message passing alone
//! provably cannot compute, and injecting them as node features raises GNN
//! expressiveness (Barceló et al., NeurIPS 2021; Jin et al., ICML 2024).
//!
//! Counts are pure DP over neighbor lists (`O(len · edges)` for walks,
//! `O(nodes · len · edges)` for closed walks); intended for feature
//! extraction, not for graphs where the latter product is prohibitive.

use crate::graph::Graph;

/// Length-`len` walk counts from each node: `hom(P_len, G, v)` per `v`
/// (the row sums of `A^len`). `len = 0` is all ones; `len = 1` is degree.
pub fn walk_counts<G: Graph>(graph: &G, len: usize) -> Vec<f64> {
    let n = graph.node_count();
    let mut counts = vec![1.0_f64; n];
    for _ in 0..len {
        let mut next = vec![0.0_f64; n];
        for (v, out) in next.iter_mut().enumerate() {
            for u in graph.neighbors(v) {
                if u < n {
                    *out += counts[u];
                }
            }
        }
        counts = next;
    }
    counts
}

/// Length-`len` closed-walk counts at each node: `hom(C_len, G, v)` per `v`
/// (the diagonal of `A^len`). `len = 3` counts rooted triangles (twice, one
/// per orientation); `len < 3` is degenerate on simple graphs and returns
/// the honest DP value rather than special-casing.
pub fn closed_walk_counts<G: Graph>(graph: &G, len: usize) -> Vec<f64> {
    let n = graph.node_count();
    let mut out = vec![0.0_f64; n];
    for (start, slot) in out.iter_mut().enumerate() {
        // x = e_start; len steps of x <- A x; the start entry is (A^len)[s, s].
        let mut x = vec![0.0_f64; n];
        x[start] = 1.0;
        for _ in 0..len {
            let mut next = vec![0.0_f64; n];
            for (v, o) in next.iter_mut().enumerate() {
                for u in graph.neighbors(v) {
                    if u < n {
                        *o += x[u];
                    }
                }
            }
            x = next;
        }
        *slot = x[start];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AdjacencyMatrix;

    fn k3() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ]
    }

    /// Hand-computed on the triangle: walks_1 = degree = 2; walks_2 =
    /// sum of neighbor degrees = 4; closed_3 = one closed triangle walk per
    /// orientation = 2.
    #[test]
    fn triangle_counts_match_hand_computation() {
        let m = k3();
        let g = AdjacencyMatrix(&m);
        assert_eq!(walk_counts(&g, 0), vec![1.0; 3]);
        assert_eq!(walk_counts(&g, 1), vec![2.0; 3]);
        assert_eq!(walk_counts(&g, 2), vec![4.0; 3]);
        assert_eq!(closed_walk_counts(&g, 3), vec![2.0; 3]);
    }

    /// The 4-cycle: eigenvalues of A are {2, 0, -2, 0}, so
    /// trace(A^4) = 2^4 + (−2)^4 = 32, i.e. 8 closed 4-walks per node — and
    /// no closed 3-walks (bipartite).
    #[test]
    fn four_cycle_closed_walks_match_the_spectrum() {
        let m = vec![
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ];
        let g = AdjacencyMatrix(&m);
        assert_eq!(closed_walk_counts(&g, 3), vec![0.0; 4]);
        assert_eq!(closed_walk_counts(&g, 4), vec![8.0; 4]);
    }

    /// The path 0-1-2 distinguishes ends from the middle where degree alone
    /// does at length 1 but closed walks stay zero (no cycles).
    #[test]
    fn path_graph_has_no_closed_odd_walks() {
        let m = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];
        let g = AdjacencyMatrix(&m);
        assert_eq!(walk_counts(&g, 2), vec![2.0, 2.0, 2.0]);
        assert_eq!(closed_walk_counts(&g, 3), vec![0.0; 3]);
    }
}
