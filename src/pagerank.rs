//! PageRank centrality.

use crate::graph::{Graph, WeightedGraph};
use crate::{Error, Result};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PageRankRun {
    pub scores: Vec<f64>,
    pub iterations: usize,
    pub diff_l1: f64,
    pub converged: bool,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PageRankConfig {
    pub damping: f64,
    pub max_iterations: usize,
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

pub fn pagerank<G: Graph>(graph: &G, config: PageRankConfig) -> Vec<f64> {
    pagerank_run(graph, config).scores
}

/// PageRank with convergence reporting.
///
/// `iterations` is the number of update steps performed.
/// `diff_l1` is the final \(L_1\) residual (sum of absolute deltas).
pub fn pagerank_run<G: Graph>(graph: &G, config: PageRankConfig) -> PageRankRun {
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
    let out_degrees: Vec<usize> = (0..n).map(|i| graph.out_degree(i)).collect();

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
                for v in graph.neighbors(u) {
                    new_scores[v] += share;
                }
            }
        }

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
    Ok(pagerank_weighted(graph, config))
}

pub fn pagerank_checked_run<G: Graph>(graph: &G, config: PageRankConfig) -> Result<PageRankRun> {
    config.validate()?;
    Ok(pagerank_run(graph, config))
}

pub fn pagerank_weighted_checked_run<G: WeightedGraph>(
    graph: &G,
    config: PageRankConfig,
) -> Result<PageRankRun> {
    // Validate config + weights by calling the existing checked wrapper first.
    // Then re-run with convergence reporting (no additional validation needed).
    let _ = pagerank_weighted_checked(graph, config)?;
    Ok(pagerank_weighted_run(graph, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AdjacencyMatrix;
    use proptest::prelude::*;

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
            let n = n;
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

            let n = n;
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
