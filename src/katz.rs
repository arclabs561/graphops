//! Katz centrality via power iteration.
//!
//! Katz centrality scores every node based on the number of walks it receives
//! from all other nodes, discounted geometrically by walk length.  The update
//! rule is
//!
//! \\[ x \leftarrow \alpha A^T x + \beta \mathbf{1} \\]
//!
//! converging to \\( x = (I - \alpha A^T)^{-1} \beta \mathbf{1} \\) when
//! \\( \alpha < 1/\rho(A) \\) (\\(\rho(A)\\) is the spectral radius).
//!
//! Unlike PageRank, Katz scores are not normalized to a probability
//! distribution; the constant \\(\beta\\) controls the baseline score every
//! node accumulates regardless of connectivity.

use crate::graph::GraphRef;
use crate::{Error, Result};

/// Hyperparameters for Katz centrality.
///
/// Defaults: `alpha = 0.1`, `beta = 1.0`, `max_iterations = 100`,
/// `tolerance = 1e-6`.  Convergence is guaranteed when
/// `alpha < 1 / spectral_radius(A)`; the default `0.1` is safe for most
/// graphs encountered in practice.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KatzConfig {
    /// Attenuation factor `alpha > 0`.  Must satisfy `alpha < 1/ρ(A)` for the
    /// iteration to converge.  Smaller values converge faster and emphasise
    /// direct neighbours; larger values give more weight to long-range walks.
    pub alpha: f64,
    /// Baseline score added to every node on each iteration.  The default
    /// `1.0` gives every node a constant "intrinsic" score independent of
    /// connectivity.
    pub beta: f64,
    /// Maximum number of power-iteration steps.  The run returns with
    /// `converged = false` when the L1 delta is still above `tolerance`.
    pub max_iterations: usize,
    /// Convergence threshold on the L1 norm of the score delta between
    /// consecutive iterations.
    pub tolerance: f64,
}

impl Default for KatzConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            beta: 1.0,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl KatzConfig {
    /// Validate hyperparameters; returns `Error::InvalidParameter` on failure.
    pub fn validate(&self) -> Result<()> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(Error::InvalidParameter(
                "alpha must be finite and > 0".to_string(),
            ));
        }
        if self.alpha >= 1.0 {
            return Err(Error::InvalidParameter(
                "alpha must be < 1 (required for convergence; tighter bound \
                 alpha < 1/spectral_radius(A) may be needed for large graphs)"
                    .to_string(),
            ));
        }
        if !self.beta.is_finite() {
            return Err(Error::InvalidParameter("beta must be finite".to_string()));
        }
        if self.max_iterations == 0 {
            return Err(Error::InvalidParameter(
                "max_iterations must be > 0".to_string(),
            ));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(Error::InvalidParameter(
                "tolerance must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Result of a Katz centrality run: scores plus convergence diagnostics.
///
/// The `*_run` variant (`katz_centrality_run`) returns this struct; the shorter
/// `katz_centrality` discards the diagnostics and returns only `scores`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KatzRun {
    /// Katz centrality score per node, indexed by node id.  All values are
    /// non-negative; the scale depends on `beta` and the graph structure.
    pub scores: Vec<f64>,
    /// Number of power-iteration steps executed.
    pub iterations: usize,
    /// L1 norm of the score delta at the final iteration.
    pub diff_l1: f64,
    /// Whether the run converged (`diff_l1 < config.tolerance`) before
    /// hitting `config.max_iterations`.
    pub converged: bool,
}

/// Compute Katz centrality scores for all nodes using default parameters.
///
/// **Currently no in-tree consumer.** Added in graphops 0.3 to round out the
/// centrality offering alongside Newman betweenness; the queued sheaf
/// community-detection workflow consumes Leiden/Louvain (partition assignment),
/// not centrality.  Use it if you have your own use case.
///
/// # Examples
///
/// ```
/// use graphops::{katz_centrality, KatzConfig};
/// use graphops::graph::GraphRef;
///
/// struct VecGraph(Vec<Vec<usize>>);
/// impl GraphRef for VecGraph {
///     fn node_count(&self) -> usize { self.0.len() }
///     fn neighbors_ref(&self, n: usize) -> &[usize] { &self.0[n] }
/// }
///
/// // 4-node path: 0--1--2--3 (undirected)
/// let g = VecGraph(vec![vec![1], vec![0, 2], vec![1, 3], vec![2]]);
/// let scores = katz_centrality(&g, KatzConfig::default());
/// assert_eq!(scores.len(), 4);
/// assert!(scores.iter().all(|&s| s >= 0.0));
/// ```
pub fn katz_centrality<G: GraphRef>(graph: &G, config: KatzConfig) -> Vec<f64> {
    katz_centrality_run(graph, config).scores
}

/// Katz centrality with full convergence diagnostics.
pub fn katz_centrality_run<G: GraphRef>(graph: &G, config: KatzConfig) -> KatzRun {
    let n = graph.node_count();
    if n == 0 {
        return KatzRun {
            scores: Vec::new(),
            iterations: 0,
            diff_l1: 0.0,
            converged: true,
        };
    }

    // Initialize scores to beta (the baseline each node accumulates on
    // iteration 0 before any edge propagation).
    let mut scores = vec![config.beta; n];
    let mut new_scores = vec![0.0_f64; n];

    // Precompute neighbor lists once so we don't allocate per iteration.
    // GraphRef gives zero-copy &[usize] slices.
    let neighbors: Vec<&[usize]> = (0..n).map(|u| graph.neighbors_ref(u)).collect();

    let mut iters = 0usize;
    let mut last_diff = f64::INFINITY;
    let mut converged = false;

    for _ in 0..config.max_iterations {
        iters += 1;

        // x_new[v] = beta + alpha * sum_{u: v in neighbors(u)} x[u]
        // Equivalently, for each edge u->v (or undirected edge {u,v}),
        // propagate alpha * scores[u] into new_scores[v].
        new_scores.fill(config.beta);
        for u in 0..n {
            let contrib = config.alpha * scores[u];
            for &v in neighbors[u] {
                if v < n {
                    new_scores[v] += contrib;
                }
            }
        }

        // L1 residual = Σ|old - new|. SIMD-accelerated via innr when the
        // `simd` feature is enabled; portable fallback uses the iterator path.
        #[cfg(feature = "simd")]
        let diff: f64 = innr::dense_f64::l1_distance_f64(&scores, &new_scores);
        #[cfg(not(feature = "simd"))]
        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();

        last_diff = diff;
        std::mem::swap(&mut scores, &mut new_scores);
        if diff < config.tolerance {
            converged = true;
            break;
        }
    }

    KatzRun {
        scores,
        iterations: iters,
        diff_l1: last_diff,
        converged,
    }
}

/// Checked Katz centrality.
///
/// Validates `config` before running.  Returns `Err(Error::InvalidParameter)`
/// for out-of-range parameters (e.g. `alpha >= 1`).
pub fn katz_centrality_checked<G: GraphRef>(graph: &G, config: KatzConfig) -> Result<Vec<f64>> {
    config.validate()?;
    Ok(katz_centrality(graph, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphRef;
    use proptest::prelude::*;

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

    /// Undirected 4-node path: 0--1--2--3.
    /// Inner nodes (1 and 2) receive more walks than the endpoints.
    #[test]
    fn path_4_nodes_center_outscores_endpoints() {
        let g = VecGraph {
            adj: vec![
                vec![1],    // 0 -- 1
                vec![0, 2], // 1 -- 0, 2
                vec![1, 3], // 2 -- 1, 3
                vec![2],    // 3 -- 2
            ],
        };
        let scores = katz_centrality(&g, KatzConfig::default());
        assert_eq!(scores.len(), 4);
        assert!(
            scores[1] > scores[0],
            "scores[1]={} should exceed scores[0]={}",
            scores[1],
            scores[0]
        );
        assert!(
            scores[2] > scores[3],
            "scores[2]={} should exceed scores[3]={}",
            scores[2],
            scores[3]
        );
    }

    /// 3-node clique: every node has the same neighbourhood, so all scores
    /// must converge to the same value.
    #[test]
    fn triangle_clique_all_equal() {
        let g = VecGraph {
            adj: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
        };
        let run = katz_centrality_run(&g, KatzConfig::default());
        assert!(run.converged, "triangle should converge");
        let s0 = run.scores[0];
        for (i, &s) in run.scores.iter().enumerate() {
            assert!(
                (s - s0).abs() < 1e-6,
                "scores[{i}]={s} diverges from scores[0]={s0}"
            );
        }
    }

    /// alpha >= 1 must be rejected by the checked variant.
    #[test]
    fn alpha_geq_one_rejected_by_checked() {
        let g = VecGraph {
            adj: vec![vec![1], vec![0]],
        };
        let cfg = KatzConfig {
            alpha: 1.0,
            ..KatzConfig::default()
        };
        assert!(
            katz_centrality_checked(&g, cfg).is_err(),
            "alpha=1.0 should return Err"
        );
        let cfg_large = KatzConfig {
            alpha: 2.5,
            ..KatzConfig::default()
        };
        assert!(
            katz_centrality_checked(&g, cfg_large).is_err(),
            "alpha=2.5 should return Err"
        );
    }

    proptest! {
        /// For any small random graph with valid config, all scores must be
        /// finite and non-negative.
        #[test]
        fn proptest_scores_finite_nonneg(
            n in 1usize..10,
            edges in proptest::collection::vec((0usize..10, 0usize..10), 0..30),
        ) {
            let mut adj = vec![vec![]; n];
            for (u, v) in edges {
                if u < n && v < n && u != v {
                    adj[u].push(v);
                    adj[v].push(u); // undirected
                }
            }
            // Deduplicate to avoid double-counting contributions.
            for row in &mut adj {
                row.sort_unstable();
                row.dedup();
            }
            let g = VecGraph { adj };
            let scores = katz_centrality(&g, KatzConfig::default());
            prop_assert_eq!(scores.len(), n);
            for &s in &scores {
                prop_assert!(s.is_finite(), "score is not finite: {s}");
                prop_assert!(s >= 0.0, "score is negative: {s}");
            }
        }
    }
}
