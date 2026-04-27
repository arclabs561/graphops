//! Betweenness centrality.
//!
//! Two algorithms are provided:
//!
//! - **Brandes** (feature-gated behind `petgraph`): exact betweenness for directed, unweighted
//!   `petgraph::Graph`/`DiGraph`.
//! - **Newman** (no feature gate): random-walk ("current-flow") betweenness centrality for
//!   undirected graphs, based on the Laplacian linear system from Newman (2005).
//!
//! Public invariants:
//! - Output vectors are indexed by node id `0..n-1`.
//! - Disconnected graphs are allowed; unreachable pairs contribute 0 to the score.

// ── Brandes (petgraph-gated) ─────────────────────────────────────────────────

#[cfg(feature = "petgraph")]
use petgraph::prelude::*;

/// Betweenness centrality (Brandes) for directed, unweighted graphs.
///
/// Returns one score per `NodeIndex`, ordered by index.
#[cfg(feature = "petgraph")]
pub fn betweenness_centrality<N, E, Ix>(graph: &petgraph::Graph<N, E, Directed, Ix>) -> Vec<f64>
where
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();
    if n <= 2 {
        return vec![0.0; n];
    }

    let mut betweenness = vec![0.0; n];

    for s in graph.node_indices() {
        let mut stack: Vec<NodeIndex<Ix>> = Vec::new();
        let mut pred: Vec<Vec<NodeIndex<Ix>>> = vec![vec![]; n];
        let mut sigma = vec![0.0f64; n];
        let mut dist: Vec<i32> = vec![-1; n];

        sigma[s.index()] = 1.0;
        dist[s.index()] = 0;

        let mut queue: std::collections::VecDeque<NodeIndex<Ix>> =
            std::collections::VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for w in graph.neighbors_directed(v, Direction::Outgoing) {
                if dist[w.index()] < 0 {
                    dist[w.index()] = dist[v.index()] + 1;
                    queue.push_back(w);
                }
                if dist[w.index()] == dist[v.index()] + 1 {
                    sigma[w.index()] += sigma[v.index()];
                    pred[w.index()].push(v);
                }
            }
        }

        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w.index()] {
                // sigma[w] can be 0 for disconnected nodes; guard division.
                let sigma_w = sigma[w.index()];
                if sigma_w > 0.0 {
                    delta[v.index()] += (sigma[v.index()] / sigma_w) * (1.0 + delta[w.index()]);
                }
            }
            if w != s {
                betweenness[w.index()] += delta[w.index()];
            }
        }
    }

    // Directed normalization to [0,1] for connected-ish graphs.
    let norm = 1.0 / ((n - 1) * (n - 2)) as f64;
    for b in &mut betweenness {
        *b *= norm;
    }
    betweenness
}

// ── Newman random-walk betweenness ───────────────────────────────────────────

use crate::graph::GraphRef;
use crate::{Error, Result};

/// Hyperparameters for Newman random-walk betweenness centrality.
///
/// Defaults: `n_sources = usize::MAX` (all sources, exact), `seed = 42`,
/// `max_iter = 200`, `tolerance = 1e-6`.
///
/// When `n_sources < n`, the algorithm samples that many source nodes
/// uniformly at random (seeded by `seed`) and extrapolates the result, giving
/// an approximation in `O(n_sources * n)` instead of `O(n^2)`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NewmanBetweennessConfig {
    /// Number of source nodes to use for the sum over source–sink pairs.
    ///
    /// Set to `usize::MAX` (the default) to enumerate all `n*(n-1)/2` pairs
    /// exactly.  Set to a smaller value for an `O(n_sources * n)` approximation.
    pub n_sources: usize,
    /// RNG seed used to draw the source sample when `n_sources < n`.
    pub seed: u64,
    /// Maximum Jacobi iterations per linear solve.  Each solve advances the
    /// current-flow potential on a single source–sink pair.
    pub max_iter: usize,
    /// L1 convergence threshold for the Jacobi residual.  The solver stops
    /// early when the residual falls below this value.
    pub tolerance: f64,
}

impl Default for NewmanBetweennessConfig {
    fn default() -> Self {
        Self {
            n_sources: usize::MAX,
            seed: 42,
            max_iter: 200,
            tolerance: 1e-6,
        }
    }
}

impl NewmanBetweennessConfig {
    /// Validate hyperparameters; returns `Error::InvalidParameter` on failure.
    pub fn validate(&self) -> Result<()> {
        if self.n_sources == 0 {
            return Err(Error::InvalidParameter("n_sources must be > 0".to_string()));
        }
        if self.max_iter == 0 {
            return Err(Error::InvalidParameter("max_iter must be > 0".to_string()));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(Error::InvalidParameter(
                "tolerance must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Result of a Newman betweenness run: scores plus convergence diagnostics.
///
/// The `*_run` variant (`newman_betweenness_run`) returns this struct; the
/// shorter `newman_betweenness` discards the diagnostics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NewmanBetweennessRun {
    /// Newman random-walk betweenness score per node, indexed by node id.
    /// Scores are non-negative; the exact scale depends on whether all
    /// source–sink pairs are enumerated or a sample is used.
    pub scores: Vec<f64>,
    /// Total Jacobi iterations across all linear solves.
    pub iterations: usize,
    /// `false` if any individual linear solve hit `max_iter` without satisfying
    /// the L1 tolerance.
    pub converged: bool,
}

/// Compute Newman random-walk betweenness centrality using default parameters.
///
/// This is the "current-flow" betweenness from Newman (2005): for each
/// source–sink pair `(s, t)` the algorithm treats the graph as a resistor
/// network and computes the fraction of the unit current that flows through
/// each node.  Summing over all pairs gives a betweenness score that captures
/// long-range connectivity, unlike the shortest-path version.
///
/// # Performance and scaling
///
/// The exact form (default `n_sources = usize::MAX`) iterates over every
/// `(s, t)` pair and runs one Jacobi linear-solve per pair, costing
/// approximately `O(n_pairs * max_iter * avg_degree) = O(n^2 * max_iter * d)`
/// total work.  Practical guidance:
///
/// - **`n <= ~500`**: exact mode is fine.  Sub-second on small graphs.
/// - **`n` in `~500..10_000`**: set `n_sources` to a fixed sample
///   (e.g. `((n as f64).sqrt() as usize).max(64)`) — this drops the cost to
///   `O(n_sources * n * max_iter * d)`.  Scores are unbiased estimates of the
///   exact betweenness up to a constant factor.
/// - **`n >= 10_000`**: exact mode is impractical (hours).  Always sample.
///
/// `_run` returns convergence diagnostics so callers can detect cases where
/// the Jacobi solver hit `max_iter` without converging (signals a poorly-
/// conditioned source–sink pair, often near disconnected components).
///
/// **Currently no in-tree consumer.** This was added alongside Katz centrality
/// in graphops 0.4 to round out the centrality offering; the queued sheaf
/// community-detection workflow uses Leiden/Louvain (partition assignment),
/// not centrality (node ranking).  Use it if you have your own use case.
///
/// # Mathematical background
///
/// For each pair `(s, t)` the potential vector \\(v\\) satisfies
///
/// \\[ L \\, v = b \\]
///
/// where \\(L = D - A\\) is the graph Laplacian (\\(D\\) = degree matrix,
/// \\(A\\) = adjacency matrix) and \\(b_i = +1\\) at the source, \\(-1\\) at
/// the sink, \\(0\\) elsewhere.  Because \\(L\\) is singular (rank \\(n-1\\)),
/// one node is pinned to potential \\(0\\) and its row/column dropped.
///
/// The Jacobi update for node \\(i\\) (pinned node excluded) is
///
/// \\[ v^{(k+1)}_i = \\frac{b_i + \\sum_{j \\in N(i)} v^{(k)}_j}{d_i} \\]
///
/// The edge current on \\((i,j)\\) is \\(|v_i - v_j|\\), and the contribution
/// to node \\(k\\)'s betweenness is
/// \\(\\tfrac{1}{2} \\sum_{j \\in N(k)} |v_k - v_j|\\).
///
/// After summing over all pairs the result is divided by
/// \\(\\tfrac{(n-1)(n-2)}{2}\\) to normalize into \\([0, 1]\\).
///
/// # Examples
///
/// ```
/// use graphops::{newman_betweenness, NewmanBetweennessConfig};
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
/// let scores = newman_betweenness(&g, NewmanBetweennessConfig::default());
/// assert!(scores[1] > scores[0]);
/// assert!(scores[2] > scores[3]);
/// ```
pub fn newman_betweenness<G: GraphRef>(graph: &G, config: NewmanBetweennessConfig) -> Vec<f64> {
    newman_betweenness_run(graph, config).scores
}

/// Newman betweenness with full convergence diagnostics.
pub fn newman_betweenness_run<G: GraphRef>(
    graph: &G,
    config: NewmanBetweennessConfig,
) -> NewmanBetweennessRun {
    let n = graph.node_count();

    // Trivial cases: need at least 3 nodes for a non-zero result.
    if n < 3 {
        return NewmanBetweennessRun {
            scores: vec![0.0; n],
            iterations: 0,
            converged: true,
        };
    }

    // Precompute neighbor lists and degrees once.
    let neighbors: Vec<&[usize]> = (0..n).map(|u| graph.neighbors_ref(u)).collect();
    let degree: Vec<usize> = (0..n).map(|u| neighbors[u].len()).collect();

    // Determine which source nodes to use.
    // When n_sources >= n we enumerate all n nodes as sources (exact).
    let sources: Vec<usize> = if config.n_sources >= n {
        (0..n).collect()
    } else {
        // Reservoir sampling (Algorithm R) — same pattern as sample_start_nodes_reservoir.
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        reservoir_sample_nodes(n, config.n_sources, &mut rng)
    };

    let n_sources_actual = sources.len();
    let mut scores = vec![0.0f64; n];
    let mut total_iters = 0usize;
    let mut all_converged = true;

    // Pinned node: node n-1 (arbitrary; its row/column is dropped from the
    // linear system to make it full-rank).
    let pinned = n - 1;

    // Reusable work buffers.
    let mut v = vec![0.0f64; n]; // current potential
    let mut v_new = vec![0.0f64; n]; // next-iteration potential

    for &s in &sources {
        for t in 0..n {
            if t == s {
                continue;
            }

            // Set up RHS: b[s] = +1, b[t] = -1, b[_] = 0.
            // Jacobi for node i (i != pinned):
            //   v[i] = (b[i] + sum_{j in N(i)} v[j]) / degree[i]
            // Pinned node stays at 0 throughout.

            // Warm-start from zero each solve.
            v.fill(0.0);

            let mut solve_iters = 0usize;
            let mut solve_converged = false;

            for _ in 0..config.max_iter {
                solve_iters += 1;
                v_new.fill(0.0);

                for i in 0..n {
                    if i == pinned || degree[i] == 0 {
                        continue;
                    }
                    // b[i]: source injects +1, sink drains -1.
                    let b_i: f64 = if i == s {
                        1.0
                    } else if i == t {
                        -1.0
                    } else {
                        0.0
                    };
                    let neighbor_sum: f64 = neighbors[i].iter().map(|&j| v[j]).sum();
                    v_new[i] = (b_i + neighbor_sum) / degree[i] as f64;
                }
                // Pinned node remains 0.

                // L1 residual over non-pinned nodes.
                #[cfg(feature = "simd")]
                let residual: f64 = innr::dense_f64::l1_distance_f64(&v, &v_new);
                #[cfg(not(feature = "simd"))]
                let residual: f64 = v
                    .iter()
                    .zip(v_new.iter())
                    .map(|(old, new)| (old - new).abs())
                    .sum();

                std::mem::swap(&mut v, &mut v_new);

                if residual < config.tolerance {
                    solve_converged = true;
                    break;
                }
            }

            total_iters += solve_iters;
            if !solve_converged {
                all_converged = false;
            }

            // Accumulate betweenness: for each node k, add (1/2) * sum_{j in N(k)} |v[k] - v[j]|.
            for k in 0..n {
                let mut flow_k = 0.0f64;
                for &j in neighbors[k] {
                    flow_k += (v[k] - v[j]).abs();
                }
                scores[k] += 0.5 * flow_k;
            }
        }
    }

    // Normalize.  For the exact case (all sources), the denominator is the
    // number of unordered pairs: (n-1)*(n-2)/2.  Each source iterates over
    // n-1 sinks, contributing to each pair once, so the raw sum is twice the
    // unordered-pair sum.  Dividing by (n-1)*(n-2) yields [0,1].
    //
    // For the sampled case we extrapolate proportionally: the sample covers
    // n_sources_actual*(n-1) ordered source→sink pairs out of n*(n-1) total,
    // so we scale by n / n_sources_actual (then apply the same (n-1)*(n-2)
    // normalization).
    let scale = if config.n_sources >= n {
        // Exact: directed pairs per source = n-1; unordered-pair norm = (n-1)*(n-2).
        1.0 / ((n - 1) * (n - 2)) as f64
    } else {
        // Approximate: extrapolate to full n then normalize.
        (n as f64 / n_sources_actual as f64) / ((n - 1) * (n - 2)) as f64
    };

    for s in &mut scores {
        *s *= scale;
    }

    NewmanBetweennessRun {
        scores,
        iterations: total_iters,
        converged: all_converged,
    }
}

/// Checked Newman betweenness centrality.
///
/// Validates `config` before running.  Returns `Err(Error::InvalidParameter)`
/// for out-of-range parameters (e.g. `n_sources == 0`).
pub fn newman_betweenness_checked<G: GraphRef>(
    graph: &G,
    config: NewmanBetweennessConfig,
) -> Result<Vec<f64>> {
    config.validate()?;
    Ok(newman_betweenness(graph, config))
}

/// Reservoir sampling (Algorithm R) of `k` distinct nodes from `0..n`.
fn reservoir_sample_nodes<R: rand::Rng>(n: usize, k: usize, rng: &mut R) -> Vec<usize> {
    let k = k.min(n);
    let mut reservoir: Vec<usize> = (0..k).collect();
    for i in k..n {
        let j = rng.random_range(0..=i);
        if j < k {
            reservoir[j] = i;
        }
    }
    reservoir
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphRef;

    // Helper: undirected adjacency-list graph for Newman tests.
    struct VG(Vec<Vec<usize>>);

    impl GraphRef for VG {
        fn node_count(&self) -> usize {
            self.0.len()
        }
        fn neighbors_ref(&self, node: usize) -> &[usize] {
            &self.0[node]
        }
    }

    // ── Brandes tests (petgraph-gated) ────────────────────────────────────

    #[cfg(feature = "petgraph")]
    #[test]
    fn line_graph_middle_is_highest() {
        use petgraph::prelude::*;

        // 0 -> 1 -> 2 -> 3
        let mut g: DiGraph<(), ()> = DiGraph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        g.add_edge(a, b, ());
        g.add_edge(b, c, ());
        g.add_edge(c, d, ());

        let bc = betweenness_centrality(&g);
        // endpoints should be 0; middle nodes should be > 0.
        assert_eq!(bc[a.index()], 0.0);
        assert_eq!(bc[d.index()], 0.0);
        assert!(bc[b.index()] > 0.0, "b={}", bc[b.index()]);
        assert!(bc[c.index()] > 0.0, "c={}", bc[c.index()]);
    }

    // ── Newman tests ──────────────────────────────────────────────────────

    /// 4-node path 0--1--2--3: inner nodes carry more current than endpoints.
    #[test]
    fn newman_line_4_nodes_inner_outscores_outer() {
        let g = VG(vec![
            vec![1],    // 0
            vec![0, 2], // 1
            vec![1, 3], // 2
            vec![2],    // 3
        ]);
        let scores = newman_betweenness(&g, NewmanBetweennessConfig::default());
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

    /// Star graph with center 0 and leaves {1,2,3,4,5}: center should dominate.
    #[test]
    fn newman_star_5_nodes_center_highest() {
        // 6 nodes: 0 (center), 1-5 (leaves)
        let g = VG(vec![
            vec![1, 2, 3, 4, 5], // 0 (center)
            vec![0],             // 1
            vec![0],             // 2
            vec![0],             // 3
            vec![0],             // 4
            vec![0],             // 5
        ]);
        let scores = newman_betweenness(&g, NewmanBetweennessConfig::default());
        assert_eq!(scores.len(), 6);
        for i in 1..6 {
            assert!(
                scores[0] > scores[i],
                "center scores[0]={} should exceed leaf scores[{}]={}",
                scores[0],
                i,
                scores[i]
            );
        }
    }

    /// `n_sources = 0` must be rejected by the checked variant.
    #[test]
    fn newman_n_sources_zero_rejected_by_checked() {
        let g = VG(vec![vec![1], vec![0], vec![1, 3], vec![2]]);
        let cfg = NewmanBetweennessConfig {
            n_sources: 0,
            ..NewmanBetweennessConfig::default()
        };
        assert!(
            newman_betweenness_checked(&g, cfg).is_err(),
            "n_sources=0 should return Err"
        );
    }

    /// For any small random undirected graph with valid config, all scores
    /// must be finite and non-negative.
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn proptest_scores_finite_nonneg(
                n in 1usize..10,
                edges in proptest::collection::vec((0usize..10, 0usize..10), 0..30),
            ) {
                let mut adj = vec![vec![]; n];
                for (u, v) in edges {
                    if u < n && v < n && u != v {
                        adj[u].push(v);
                        adj[v].push(u);
                    }
                }
                for row in &mut adj {
                    row.sort_unstable();
                    row.dedup();
                }
                let g = VG(adj);
                let scores = newman_betweenness(&g, NewmanBetweennessConfig::default());
                prop_assert_eq!(scores.len(), n);
                for &s in &scores {
                    prop_assert!(s.is_finite(), "score is not finite: {s}");
                    prop_assert!(s >= 0.0, "score is negative: {s}");
                }
            }
        }
    }
}
