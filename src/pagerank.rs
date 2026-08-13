//! PageRank centrality (power iteration on a random-surfer Markov chain).

use crate::graph::{Graph, GraphRef, WeightedGraph};
use crate::{Error, Result};

/// Result of a PageRank run: scores plus convergence diagnostics.
///
/// The `*_run` variants (`pagerank_run`, `pagerank_weighted_run`,
/// `personalized_pagerank_run`) return this struct; the shorter variants
/// (`pagerank`, etc.) discard the diagnostics and return only `scores`.
///
/// Use the diagnostic fields when tuning `max_iterations` / `tolerance` or
/// when bench-marking convergence across graphs of different sizes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PageRankRun {
    /// PageRank score per node, indexed by node id. Sums to approximately
    /// `1.0` modulo floating-point and dangling-node redistribution.
    pub scores: Vec<f64>,
    /// Number of power-iteration steps executed. Equal to
    /// `config.max_iterations` when `converged == false`.
    pub iterations: usize,
    /// L1 norm of the score delta at the final iteration. Compare against
    /// `config.tolerance` to see how close we were to converging.
    pub diff_l1: f64,
    /// Whether the run converged (`diff_l1 < config.tolerance`) before hitting
    /// `config.max_iterations`.
    pub converged: bool,
}

/// PageRank hyperparameters.
///
/// Defaults match the Brin–Page 1998 paper: `damping = 0.85`,
/// `max_iterations = 100`, `tolerance = 1e-6`. For high-precision downstream
/// work (personalized PageRank as a similarity metric) drop tolerance to
/// `1e-9` and bump iterations to `500`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PageRankConfig {
    /// Damping factor `alpha` in `[0, 1]`: the probability of following an
    /// edge at each step (vs teleporting uniformly). Classical value is `0.85`.
    pub damping: f64,
    /// Maximum power-iteration steps before giving up. The run returns
    /// with `converged = false` if the L1 delta is still above `tolerance`.
    pub max_iterations: usize,
    /// Convergence threshold on the L1 norm of the score delta between
    /// consecutive iterations. Lower = tighter convergence, more iterations.
    pub tolerance: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl PageRankConfig {
    /// Validate hyperparameters; returns `Error::InvalidParameter` on failure.
    pub fn validate(&self) -> Result<()> {
        if !self.damping.is_finite() {
            return Err(Error::InvalidParameter(
                "damping must be finite".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.damping) {
            return Err(Error::InvalidParameter(
                "damping must be in [0,1]".to_string(),
            ));
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

/// Checked PageRank centrality.
///
/// This validates `config` and rejects obviously-invalid numeric settings.
pub fn pagerank_checked<G: Graph>(graph: &G, config: PageRankConfig) -> Result<Vec<f64>> {
    config.validate()?;
    Ok(pagerank(graph, config))
}

/// Compute PageRank scores for all nodes.
///
/// # Examples
///
/// ```
/// use graphops::{pagerank, AdjacencyMatrix, PageRankConfig};
///
/// // 3-node directed graph: 0->1, 1->2, 2->0
/// let adj = vec![
///     vec![0.0, 1.0, 0.0],
///     vec![0.0, 0.0, 1.0],
///     vec![1.0, 0.0, 0.0],
/// ];
/// let scores = pagerank(&AdjacencyMatrix(&adj), PageRankConfig::default());
/// assert_eq!(scores.len(), 3);
/// let total: f64 = scores.iter().sum();
/// assert!((total - 1.0).abs() < 1e-6);
/// ```
pub fn pagerank<G: Graph>(graph: &G, config: PageRankConfig) -> Vec<f64> {
    pagerank_run(graph, config).scores
}

/// PageRank with convergence reporting.
///
/// `iterations` is the number of update steps performed.
/// `diff_l1` is the final \(L_1\) residual (sum of absolute deltas).
pub fn pagerank_run<G: Graph>(graph: &G, config: PageRankConfig) -> PageRankRun {
    let n = graph.node_count();
    // Precompute neighbor lists once to avoid per-iteration allocation.
    let neighbors: Vec<Vec<usize>> = (0..n).map(|u| graph.neighbors(u)).collect();
    pagerank_ref_run(&BorrowedAdjacency(&neighbors), config)
}

struct BorrowedAdjacency<'a>(&'a [Vec<usize>]);

impl GraphRef for BorrowedAdjacency<'_> {
    fn node_count(&self) -> usize {
        self.0.len()
    }

    fn neighbors_ref(&self, node: usize) -> &[usize] {
        &self.0[node]
    }
}

/// Compute PageRank over borrowed adjacency lists.
///
/// Unlike [`pagerank`], this path does not copy edge lists before iteration.
/// Prefer it for CSR and adjacency-list representations that can return stable
/// neighbor slices.
pub fn pagerank_ref<G: GraphRef>(graph: &G, config: PageRankConfig) -> Vec<f64> {
    pagerank_ref_run(graph, config).scores
}

/// PageRank with convergence reporting over borrowed adjacency lists.
pub fn pagerank_ref_run<G: GraphRef>(graph: &G, config: PageRankConfig) -> PageRankRun {
    let n = graph.node_count();
    if n == 0 {
        return PageRankRun {
            scores: Vec::new(),
            iterations: 0,
            diff_l1: 0.0,
            converged: true,
        };
    }
    let n_f64 = n as f64;
    let mut scores = vec![1.0 / n_f64; n];
    let mut new_scores = vec![0.0; n];
    let out_degrees: Vec<usize> = (0..n).map(|u| graph.neighbors_ref(u).len()).collect();

    let mut iters = 0usize;
    let mut last_diff = f64::INFINITY;
    let mut converged = false;
    for _ in 0..config.max_iterations {
        iters += 1;
        let dangling_sum: f64 = out_degrees
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(i, _)| scores[i])
            .sum();
        let dangling_contrib = config.damping * dangling_sum / n_f64;
        let teleport = (1.0 - config.damping) / n_f64;
        new_scores.fill(teleport + dangling_contrib);

        for u in 0..n {
            let deg = out_degrees[u];
            if deg > 0 {
                let share = config.damping * scores[u] / deg as f64;
                for &v in graph.neighbors_ref(u) {
                    new_scores[v] += share;
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
    PageRankRun {
        scores,
        iterations: iters,
        diff_l1: last_diff,
        converged,
    }
}

/// Checked PageRank over borrowed adjacency lists.
pub fn pagerank_ref_checked<G: GraphRef>(graph: &G, config: PageRankConfig) -> Result<Vec<f64>> {
    config.validate()?;
    Ok(pagerank_ref(graph, config))
}

/// Checked PageRank with diagnostics over borrowed adjacency lists.
pub fn pagerank_ref_checked_run<G: GraphRef>(
    graph: &G,
    config: PageRankConfig,
) -> Result<PageRankRun> {
    config.validate()?;
    Ok(pagerank_ref_run(graph, config))
}

/// Weighted PageRank centrality.
///
/// Edges are treated as having non-negative weights, and a node's outgoing mass is split
/// proportionally to outgoing edge weights.
///
/// This matches the Markov chain transition:
/// \[
///   P(u \to v) = \frac{w(u,v)}{\sum_x w(u,x)}
/// \]
///
/// Note: `pagerank_weighted` clamps negative weights to 0.0 for robustness. If you need strict
/// validation (no negatives/NaNs), use `pagerank_weighted_checked`.
pub fn pagerank_weighted<G: WeightedGraph>(graph: &G, config: PageRankConfig) -> Vec<f64> {
    pagerank_weighted_run(graph, config).scores
}

/// Like [`pagerank_weighted`] but returns full [`PageRankRun`] diagnostics.
///
/// Prefer this when you need to know whether the run converged
/// (`result.converged`) or how close it got (`result.diff_l1`). The scalar
/// `pagerank_weighted` wrapper discards everything but `scores`.
pub fn pagerank_weighted_run<G: WeightedGraph>(graph: &G, config: PageRankConfig) -> PageRankRun {
    let n = graph.node_count();
    if n == 0 {
        return PageRankRun {
            scores: Vec::new(),
            iterations: 0,
            diff_l1: 0.0,
            converged: true,
        };
    }

    let n_f64 = n as f64;
    let mut scores = vec![1.0 / n_f64; n];
    let mut new_scores = vec![0.0; n];

    // Precompute outgoing neighbors once (Graph) and outgoing weight sums (WeightedGraph).
    let neighbors: Vec<Vec<usize>> = (0..n).map(|u| graph.neighbors(u)).collect();
    let out_wsum: Vec<f64> = (0..n)
        .map(|u| {
            neighbors[u]
                .iter()
                .map(|&v| graph.edge_weight(u, v).max(0.0))
                .sum()
        })
        .collect();

    let mut iters = 0usize;
    let mut last_diff = f64::INFINITY;
    let mut converged = false;
    for _ in 0..config.max_iterations {
        iters += 1;
        let dangling_sum: f64 = out_wsum
            .iter()
            .enumerate()
            .filter(|(_, &ws)| ws == 0.0)
            .map(|(i, _)| scores[i])
            .sum();

        let dangling_contrib = config.damping * dangling_sum / n_f64;
        let teleport = (1.0 - config.damping) / n_f64;
        new_scores.fill(teleport + dangling_contrib);

        for u in 0..n {
            let ws = out_wsum[u];
            if ws > 0.0 {
                // distribute along outgoing edges proportional to weight
                for &v in &neighbors[u] {
                    let w = graph.edge_weight(u, v).max(0.0);
                    if w > 0.0 {
                        new_scores[v] += config.damping * scores[u] * (w / ws);
                    }
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

    PageRankRun {
        scores,
        iterations: iters,
        diff_l1: last_diff,
        converged,
    }
}

/// Checked weighted PageRank.
///
/// Rejects NaN/negative edge weights (instead of silently clamping them to 0.0).
pub fn pagerank_weighted_checked<G: WeightedGraph>(
    graph: &G,
    config: PageRankConfig,
) -> Result<Vec<f64>> {
    config.validate()?;
    validate_weights(graph)?;
    Ok(pagerank_weighted(graph, config))
}

fn validate_weights<G: WeightedGraph>(graph: &G) -> Result<()> {
    let n = graph.node_count();
    for u in 0..n {
        for v in graph.neighbors(u) {
            let w = graph.edge_weight(u, v);
            if !w.is_finite() {
                return Err(Error::InvalidParameter(
                    "edge weights must be finite".to_string(),
                ));
            }
            if w < 0.0 {
                return Err(Error::InvalidParameter(
                    "edge weights must be non-negative".to_string(),
                ));
            }
        }
    }
    Ok(())
}

/// Validated PageRank with full diagnostics. Errors on invalid `config` and
/// returns a [`PageRankRun`] on success. Equivalent to calling
/// [`PageRankConfig::validate`] then [`pagerank_run`].
pub fn pagerank_checked_run<G: Graph>(graph: &G, config: PageRankConfig) -> Result<PageRankRun> {
    config.validate()?;
    Ok(pagerank_run(graph, config))
}

/// Validated weighted PageRank with full diagnostics. Errors on invalid config,
/// NaN weights, or negative weights. Returns a [`PageRankRun`] on success.
pub fn pagerank_weighted_checked_run<G: WeightedGraph>(
    graph: &G,
    config: PageRankConfig,
) -> Result<PageRankRun> {
    config.validate()?;
    validate_weights(graph)?;
    Ok(pagerank_weighted_run(graph, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AdjacencyMatrix;
    use proptest::prelude::*;
    use std::cell::Cell;

    struct RefGraph(Vec<Vec<usize>>);

    impl GraphRef for RefGraph {
        fn node_count(&self) -> usize {
            self.0.len()
        }

        fn neighbors_ref(&self, node: usize) -> &[usize] {
            &self.0[node]
        }
    }

    #[test]
    fn pagerank_ref_matches_owned_path() {
        let adjacency = vec![vec![1, 2], vec![2], vec![0], vec![]];
        let borrowed = RefGraph(adjacency.clone());

        struct Owned(Vec<Vec<usize>>);
        impl Graph for Owned {
            fn node_count(&self) -> usize {
                self.0.len()
            }
            fn neighbors(&self, node: usize) -> Vec<usize> {
                self.0[node].clone()
            }
        }

        let config = PageRankConfig {
            tolerance: 1e-12,
            ..PageRankConfig::default()
        };
        let owned = pagerank_run(&Owned(adjacency), config);
        let by_ref = pagerank_ref_run(&borrowed, config);
        assert_eq!(owned.iterations, by_ref.iterations);
        assert_eq!(owned.converged, by_ref.converged);
        assert_eq!(owned.scores, by_ref.scores);
        assert_eq!(owned.diff_l1, by_ref.diff_l1);
    }

    #[test]
    fn weighted_checked_run_executes_solver_once() {
        struct CountedWeighted {
            adjacency: Vec<Vec<usize>>,
            neighbor_calls: Cell<usize>,
        }
        impl Graph for CountedWeighted {
            fn node_count(&self) -> usize {
                self.adjacency.len()
            }
            fn neighbors(&self, node: usize) -> Vec<usize> {
                self.neighbor_calls.set(self.neighbor_calls.get() + 1);
                self.adjacency[node].clone()
            }
        }
        impl WeightedGraph for CountedWeighted {
            fn edge_weight(&self, _source: usize, _target: usize) -> f64 {
                1.0
            }
        }

        let graph = CountedWeighted {
            adjacency: vec![vec![1], vec![0]],
            neighbor_calls: Cell::new(0),
        };
        pagerank_weighted_checked_run(&graph, PageRankConfig::default()).unwrap();

        // One graph scan validates weights and one builds the solver's cached
        // adjacency. A duplicate solver execution would add a third scan.
        assert_eq!(graph.neighbor_calls.get(), 2 * graph.node_count());
    }

    #[test]
    fn test_pagerank_weighted_sums_to_one() {
        // 3 nodes, weighted edges:
        // 0 -> 1 (2.0), 0 -> 2 (1.0)
        // 1 -> 2 (1.0)
        // 2 -> (dangling)
        let adj = vec![
            vec![0.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ];
        let g = AdjacencyMatrix(&adj);
        let scores = pagerank_weighted(&g, PageRankConfig::default());
        let total: f64 = scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "sum={total}");
    }

    #[test]
    fn test_pagerank_weight_biases_toward_heavier_edge() {
        // 0 links to 1 twice as strongly as to 2, so 1 should rank >= 2.
        let adj = vec![
            vec![0.0, 2.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let g = AdjacencyMatrix(&adj);
        let scores = pagerank_weighted(&g, PageRankConfig::default());
        assert!(
            scores[1] >= scores[2],
            "scores[1]={} scores[2]={}",
            scores[1],
            scores[2]
        );
    }

    proptest! {
        #[test]
        fn prop_pagerank_unweighted_sums_to_one(n in 1usize..10, edges in proptest::collection::vec((0usize..10, 0usize..10), 0..40)) {
            // Build unweighted adjacency matrix with 0/1 entries.
            let mut adj = vec![vec![0.0_f64; n]; n];
            for (u,v) in edges {
                if u < n && v < n && u != v {
                    adj[u][v] = 1.0;
                }
            }
            let g = AdjacencyMatrix(&adj);
            let scores = pagerank_checked(&g, PageRankConfig::default()).unwrap();
            prop_assert_eq!(scores.len(), n);
            let sum: f64 = scores.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-6, "sum={}", sum);
            prop_assert!(scores.iter().all(|x| *x >= -1e-12));
        }

        #[test]
        fn prop_pagerank_weighted_checked_rejects_negative(n in 2usize..10, u in 0usize..10, v in 0usize..10) {
            #[derive(Debug)]
            struct TestWeighted {
                neighbors: Vec<Vec<usize>>,
                w: Vec<Vec<f64>>,
            }
            impl crate::graph::Graph for TestWeighted {
                fn node_count(&self) -> usize { self.neighbors.len() }
                fn neighbors(&self, node: usize) -> Vec<usize> { self.neighbors[node].clone() }
            }
            impl crate::graph::WeightedGraph for TestWeighted {
                fn edge_weight(&self, source: usize, target: usize) -> f64 { self.w[source][target] }
            }

            let u = u % n;
            let mut v = v % n;
            if v == u { v = (v + 1) % n; }

            let mut neighbors = vec![Vec::<usize>::new(); n];
            let mut w = vec![vec![0.0_f64; n]; n];
            neighbors[u].push(v);
            w[u][v] = -1.0;

            let g = TestWeighted { neighbors, w };
            let res = pagerank_weighted_checked(&g, PageRankConfig::default());
            prop_assert!(res.is_err());
        }
    }
}
