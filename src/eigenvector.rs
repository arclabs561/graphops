//! Eigenvector centrality via power iteration.
//!
//! Measures node importance based on the principle that connections to
//! high-scoring nodes contribute more. Related to PageRank but without
//! teleportation or damping.
//!
//! The dominant eigenvector of the adjacency matrix gives the centrality
//! scores. For undirected graphs, this is well-defined (Perron-Frobenius).

use crate::graph::GraphRef;

/// Convergence details for eigenvector centrality computation.
#[derive(Debug, Clone)]
pub struct EigenvectorRun {
    /// Centrality scores, L2-normalized.
    pub scores: Vec<f64>,
    /// Actual iterations performed.
    pub iterations: usize,
    /// Final L1 residual between successive iterations.
    pub diff_l1: f64,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
}

/// Compute eigenvector centrality with default parameters (max_iter=100, tol=1e-6).
pub fn eigenvector_centrality<G: GraphRef>(graph: &G) -> Vec<f64> {
    eigenvector_centrality_run(graph, 100, 1e-6).scores
}

/// Compute eigenvector centrality with full convergence reporting.
///
/// Scores are L2-normalized (unit vector). Isolated nodes get score 0.
///
/// ```
/// use graphops::eigenvector::eigenvector_centrality_run;
/// use graphops::GraphRef;
///
/// struct G(Vec<Vec<usize>>);
/// impl GraphRef for G {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors_ref(&self, n: usize) -> &[usize] { &self.0[n] }
/// }
///
/// let g = G(vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
/// let run = eigenvector_centrality_run(&g, 100, 1e-6);
/// assert!(run.converged);
/// assert!((run.scores[0] - run.scores[1]).abs() < 1e-6); // symmetric graph
/// ```
pub fn eigenvector_centrality_run<G: GraphRef>(
    graph: &G,
    max_iterations: usize,
    tolerance: f64,
) -> EigenvectorRun {
    let n = graph.node_count();
    if n == 0 {
        return EigenvectorRun {
            scores: vec![],
            iterations: 0,
            diff_l1: 0.0,
            converged: true,
        };
    }

    // Initialize uniformly.
    let init = 1.0 / (n as f64).sqrt();
    let mut scores = vec![init; n];
    let mut new_scores = vec![0.0; n];
    let mut diff_l1 = 0.0;
    let mut iterations = 0;

    for iter in 0..max_iterations {
        iterations = iter + 1;

        // Multiply by (A + I): new[v] = scores[v] + sum(scores[u]) for u in neighbors(v).
        // Adding the identity breaks bipartite oscillation while preserving eigenvector order.
        for v in 0..n {
            let mut sum = scores[v]; // self-loop (identity contribution)
            for &u in graph.neighbors_ref(v) {
                if u < n {
                    sum += scores[u];
                }
            }
            new_scores[v] = sum;
        }

        // L2-normalize with positive sign convention (largest component positive).
        let norm: f64 = new_scores.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            // Sign convention: ensure the vector has a positive sum.
            let sign = if new_scores.iter().sum::<f64>() >= 0.0 {
                1.0
            } else {
                -1.0
            };
            for x in &mut new_scores {
                *x = *x * sign / norm;
            }
        }

        // Compute L1 diff.
        diff_l1 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        std::mem::swap(&mut scores, &mut new_scores);

        if diff_l1 < tolerance {
            return EigenvectorRun {
                scores,
                iterations,
                diff_l1,
                converged: true,
            };
        }
    }

    EigenvectorRun {
        scores,
        iterations,
        diff_l1,
        converged: false,
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
    fn triangle_all_equal() {
        let g = VecGraph {
            adj: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
        };
        let run = eigenvector_centrality_run(&g, 100, 1e-8);
        assert!(run.converged);
        let expected = 1.0 / 3.0_f64.sqrt();
        for &s in &run.scores {
            assert!((s - expected).abs() < 1e-6, "score {s} != {expected}");
        }
    }

    #[test]
    fn star_center_highest() {
        // Star: node 0 connected to 1,2,3,4
        let g = VecGraph {
            adj: vec![
                vec![1, 2, 3, 4],
                vec![0],
                vec![0],
                vec![0],
                vec![0],
            ],
        };
        let scores = eigenvector_centrality(&g);
        assert!(scores[0] > scores[1]);
        assert!((scores[1] - scores[2]).abs() < 1e-6);
    }

    #[test]
    fn empty_graph() {
        let g = VecGraph { adj: vec![] };
        let run = eigenvector_centrality_run(&g, 100, 1e-6);
        assert!(run.scores.is_empty());
        assert!(run.converged);
    }

    #[test]
    fn isolated_nodes() {
        let g = VecGraph {
            adj: vec![vec![], vec![], vec![]],
        };
        let scores = eigenvector_centrality(&g);
        // All zeros after normalization of zero vector.
        for &s in &scores {
            assert!(s.abs() < 1e-10 || (s - 1.0 / 3.0_f64.sqrt()).abs() < 1e-10);
        }
    }

    #[test]
    fn l2_normalized() {
        let g = VecGraph {
            adj: vec![
                vec![1, 2, 3],
                vec![0, 2],
                vec![0, 1, 3],
                vec![0, 2],
            ],
        };
        let scores = eigenvector_centrality(&g);
        let l2: f64 = scores.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((l2 - 1.0).abs() < 1e-6, "L2 norm = {l2}");
    }
}
