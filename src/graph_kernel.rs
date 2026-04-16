//! Graph kernels: similarity measures between labeled graphs.
//!
//! These kernels measure structural similarity between graphs, connecting
//! RKHS methods to graph-structured data.
//!
//! Graphs are represented as adjacency lists (`&[Vec<usize>]`) with
//! optional node labels (`&[u64]`). Node indices are 0-based.

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

/// Weisfeiler-Leman subtree kernel (1-WL).
///
/// Counts matching subtree patterns at each depth up to `max_depth`.
/// At each iteration, node labels are refined by hashing the multiset
/// of neighbor labels, capturing increasingly large subtree structure.
///
/// The kernel value is the inner product of the aggregated label histograms
/// across all depths.
///
/// # Arguments
///
/// * `adj1` - Adjacency list for graph 1
/// * `labels1` - Initial node labels for graph 1
/// * `adj2` - Adjacency list for graph 2
/// * `labels2` - Initial node labels for graph 2
/// * `max_depth` - Number of WL refinement iterations
///
/// # Example
///
/// ```rust
/// use graphops::wl_subtree_kernel;
///
/// // Triangle graph
/// let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
/// let labels = vec![1, 1, 1];
///
/// let k = wl_subtree_kernel(&adj, &labels, &adj, &labels, 2);
/// assert!(k > 0.0);
/// ```
pub fn wl_subtree_kernel(
    adj1: &[Vec<usize>],
    labels1: &[u64],
    adj2: &[Vec<usize>],
    labels2: &[u64],
    max_depth: usize,
) -> f64 {
    let n1 = adj1.len();
    let n2 = adj2.len();

    if n1 == 0 && n2 == 0 {
        return 0.0;
    }

    let mut current1: Vec<u64> = labels1.to_vec();
    let mut current2: Vec<u64> = labels2.to_vec();

    let mut next_id: u64 = {
        // Start IDs after the max initial label to avoid collisions
        let max_label = labels1
            .iter()
            .chain(labels2.iter())
            .copied()
            .max()
            .unwrap_or(0);
        max_label + 1
    };

    // Signature -> compressed label mapping, shared across both graphs
    // so that identical structural signatures get the same label.
    let mut signature_map: HashMap<(u64, Vec<u64>), u64> = HashMap::new();

    let mut total = 0.0;

    // Count matching labels at depth 0
    total += histogram_dot(&current1, &current2);

    for _depth in 0..max_depth {
        let new1 = refine_labels(adj1, &current1, &mut signature_map, &mut next_id);
        let new2 = refine_labels(adj2, &current2, &mut signature_map, &mut next_id);

        current1 = new1;
        current2 = new2;

        total += histogram_dot(&current1, &current2);
    }

    total
}

/// Refine node labels by hashing each node's label with its sorted neighbor labels.
///
/// The `signature_map` must be shared across both graphs within a kernel call
/// so that identical structural patterns receive the same compressed label.
fn refine_labels(
    adj: &[Vec<usize>],
    labels: &[u64],
    signature_map: &mut HashMap<(u64, Vec<u64>), u64>,
    next_id: &mut u64,
) -> Vec<u64> {
    let n = adj.len();
    let mut new_labels = Vec::with_capacity(n);

    for i in 0..n {
        let mut neighbor_labels: Vec<u64> = adj[i].iter().map(|&j| labels[j]).collect();
        neighbor_labels.sort_unstable();

        let sig = (labels[i], neighbor_labels);

        let new_label = *signature_map.entry(sig).or_insert_with(|| {
            let id = *next_id;
            *next_id += 1;
            id
        });

        new_labels.push(new_label);
    }

    new_labels
}

/// Inner product of two label histograms: sum over all labels of count_1(l) * count_2(l).
fn histogram_dot(labels1: &[u64], labels2: &[u64]) -> f64 {
    let mut counts1: HashMap<u64, u64> = HashMap::new();
    for &l in labels1 {
        *counts1.entry(l).or_insert(0) += 1;
    }

    let mut counts2: HashMap<u64, u64> = HashMap::new();
    for &l in labels2 {
        *counts2.entry(l).or_insert(0) += 1;
    }

    let mut dot = 0u64;
    for (&label, &c1) in &counts1 {
        if let Some(&c2) = counts2.get(&label) {
            dot += c1 * c2;
        }
    }

    dot as f64
}

/// Random walk kernel via direct product graph.
///
/// Counts matching random walks between two graphs with geometric decay.
/// `k(G1, G2) = sum_{l=0}^{max_len} lambda^l * (matching walks of length l)`
///
/// A matching walk of length `l` is a sequence of node pairs
/// `(u0,v0), (u1,v1), ..., (ul,vl)` where each `(ui, ui+1)` is an edge
/// in G1 and `(vi, vi+1)` is an edge in G2.
///
/// # Arguments
///
/// * `adj1` - Adjacency list for graph 1
/// * `adj2` - Adjacency list for graph 2
/// * `max_len` - Maximum walk length
/// * `lambda` - Decay factor per step (should be < 1 / max(degree) for convergence)
///
/// # Example
///
/// ```rust
/// use graphops::random_walk_kernel;
///
/// // Two edges
/// let adj1 = vec![vec![1], vec![0]];
/// let adj2 = vec![vec![1], vec![0]];
///
/// let k = random_walk_kernel(&adj1, &adj2, 3, 0.1);
/// assert!(k > 0.0);
/// ```
pub fn random_walk_kernel(
    adj1: &[Vec<usize>],
    adj2: &[Vec<usize>],
    max_len: usize,
    lambda: f64,
) -> f64 {
    let n1 = adj1.len();
    let n2 = adj2.len();

    if n1 == 0 || n2 == 0 {
        return 0.0;
    }

    let np = n1 * n2; // product graph size

    // Build product graph adjacency.
    // Product node (i, j) has index i * n2 + j.
    // Edge (i,j) -> (i',j') exists iff (i,i') in E1 and (j,j') in E2.
    let mut prod_adj: Vec<Vec<usize>> = vec![Vec::new(); np];
    for (i, neighbors_i) in adj1.iter().enumerate() {
        for &i_prime in neighbors_i {
            for (j, neighbors_j) in adj2.iter().enumerate() {
                for &j_prime in neighbors_j {
                    let from = i * n2 + j;
                    let to = i_prime * n2 + j_prime;
                    prod_adj[from].push(to);
                }
            }
        }
    }

    // Count walks using iterative BFS-style propagation.
    // walk_count[node] = number of walks ending at this node at current length.
    let mut walk_count = vec![1.0_f64; np]; // length 0: one walk per product node
    let mut total: f64 = walk_count.iter().sum::<f64>(); // lambda^0 * count

    for l in 1..=max_len {
        let mut next_count = vec![0.0_f64; np];
        for (node, neighbors) in prod_adj.iter().enumerate() {
            for &neighbor in neighbors {
                next_count[neighbor] += walk_count[node];
            }
        }
        let factor = lambda.powi(l as i32);
        total += factor * next_count.iter().sum::<f64>();
        walk_count = next_count;
    }

    total
}

/// Sliced Wasserstein graph kernel.
///
/// Compares graphs by computing the Sliced Wasserstein distance between
/// their node feature distributions, then exponentiating:
///
/// `k(G1, G2) = exp(-lambda * SW(features_G1, features_G2))`
///
/// Each graph is represented as a set of node feature vectors (an empirical
/// distribution in feature space). The SW distance projects both distributions
/// onto random directions and averages the 1D Wasserstein distances.
///
/// For graphs without explicit node features, use [`structural_node_features`]
/// to compute degree, clustering coefficient, and average neighbor degree.
///
/// # Arguments
///
/// * `features1` - Node features of graph 1 (n1 vectors of dimension d)
/// * `features2` - Node features of graph 2 (n2 vectors of dimension d)
/// * `num_projections` - Number of random projection directions
/// * `lambda` - Bandwidth parameter (larger = more sensitive to distance)
/// * `seed` - RNG seed for reproducible projections
///
/// # Example
///
/// ```rust
/// use graphops::{sliced_wasserstein_graph_kernel, structural_node_features};
///
/// // Triangle vs path
/// let tri = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
/// let path = vec![vec![1], vec![0, 2], vec![1]];
///
/// let f1 = structural_node_features(&tri);
/// let f2 = structural_node_features(&path);
///
/// let k = sliced_wasserstein_graph_kernel(&f1, &f2, 50, 1.0, 42);
/// assert!(k > 0.0 && k <= 1.0);
/// ```
pub fn sliced_wasserstein_graph_kernel(
    features1: &[Vec<f64>],
    features2: &[Vec<f64>],
    num_projections: usize,
    lambda: f64,
    seed: u64,
) -> f64 {
    if features1.is_empty() || features2.is_empty() {
        return 0.0;
    }

    let d = features1[0].len();
    assert!(
        features2[0].len() == d,
        "feature dimensions must match: {} vs {}",
        d,
        features2[0].len()
    );

    let sw = sliced_wasserstein_distance(features1, features2, d, num_projections, seed);
    (-lambda * sw).exp()
}

/// Compute Sliced Wasserstein distance between two empirical distributions.
///
/// Projects both point sets onto `num_projections` random unit vectors,
/// then averages the 1D Wasserstein-1 distances of the projections.
fn sliced_wasserstein_distance(
    features1: &[Vec<f64>],
    features2: &[Vec<f64>],
    d: usize,
    num_projections: usize,
    seed: u64,
) -> f64 {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut total_dist = 0.0;

    for _ in 0..num_projections {
        // Sample random direction on S^{d-1} via Gaussian + normalize
        let direction = random_unit_vector(d, &mut rng);

        // Project both sets
        let mut proj1: Vec<f64> = features1.iter().map(|f| dot(f, &direction)).collect();
        let mut proj2: Vec<f64> = features2.iter().map(|f| dot(f, &direction)).collect();

        proj1.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        proj2.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        total_dist += wasserstein_1d(&proj1, &proj2);
    }

    total_dist / num_projections as f64
}

/// 1D Wasserstein-1 distance between two sorted empirical distributions.
///
/// Uses the quantile function representation: W1 = integral |F^{-1} - G^{-1}|.
/// For discrete distributions of possibly different sizes n and m, we evaluate
/// at n*m uniformly spaced quantile points (equivalently, merge the two CDFs).
fn wasserstein_1d(sorted1: &[f64], sorted2: &[f64]) -> f64 {
    let n = sorted1.len();
    let m = sorted2.len();

    if n == m {
        // Equal sizes: W1 = (1/n) * sum |x_i - y_i| on sorted values
        let mut sum = 0.0;
        for i in 0..n {
            sum += (sorted1[i] - sorted2[i]).abs();
        }
        return sum / n as f64;
    }

    // Unequal sizes: integrate |F^{-1}(t) - G^{-1}(t)| over [0,1].
    // Merge the breakpoints of both quantile functions.
    // F^{-1}(t) = sorted1[floor(t * n)] for t in [0,1).
    // We integrate piecewise between consecutive breakpoints.
    let n_f = n as f64;
    let m_f = m as f64;

    // Breakpoints are at k/n for k=1..n and k/m for k=1..m
    let mut breaks: Vec<f64> = Vec::with_capacity(n + m + 2);
    breaks.push(0.0);
    for k in 1..=n {
        breaks.push(k as f64 / n_f);
    }
    for k in 1..=m {
        breaks.push(k as f64 / m_f);
    }
    breaks.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    breaks.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    let mut integral = 0.0;
    for i in 0..breaks.len() - 1 {
        let t = (breaks[i] + breaks[i + 1]) * 0.5; // midpoint
        let width = breaks[i + 1] - breaks[i];

        // Quantile function: F^{-1}(t) = sorted[floor(t * n)], clamped
        let idx1 = ((t * n_f).floor() as usize).min(n - 1);
        let idx2 = ((t * m_f).floor() as usize).min(m - 1);

        integral += (sorted1[idx1] - sorted2[idx2]).abs() * width;
    }

    integral
}

/// Sample a random unit vector on S^{d-1} using Gaussian projection.
fn random_unit_vector(d: usize, rng: &mut SmallRng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..d).map(|_| sample_normal(rng)).collect();
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Box-Muller normal sample.
fn sample_normal(rng: &mut SmallRng) -> f64 {
    let u1: f64 = rng.random::<f64>();
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Inner product of two slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute structural node features from an adjacency list.
///
/// Returns per-node feature vectors: `[degree, clustering_coefficient, avg_neighbor_degree]`.
///
/// These features capture local graph topology and can be used with
/// [`sliced_wasserstein_graph_kernel`] for graphs without explicit node attributes.
///
/// # Arguments
///
/// * `adj` - Adjacency list (undirected; `adj[i]` lists neighbors of node i)
///
/// # Example
///
/// ```rust
/// use graphops::structural_node_features;
///
/// // Triangle: every node has degree 2, clustering coeff 1.0
/// let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
/// let features = structural_node_features(&adj);
/// assert_eq!(features.len(), 3);
/// assert!((features[0][0] - 2.0).abs() < 1e-10); // degree
/// assert!((features[0][1] - 1.0).abs() < 1e-10); // clustering
/// ```
pub fn structural_node_features(adj: &[Vec<usize>]) -> Vec<Vec<f64>> {
    let n = adj.len();
    let mut features = Vec::with_capacity(n);

    for i in 0..n {
        let degree = adj[i].len() as f64;

        // Local clustering coefficient
        let cc = if adj[i].len() < 2 {
            0.0
        } else {
            let neighbors = &adj[i];
            let mut triangles = 0u64;
            for (a_idx, &a) in neighbors.iter().enumerate() {
                for &b in &neighbors[a_idx + 1..] {
                    if adj[a].contains(&b) {
                        triangles += 1;
                    }
                }
            }
            let possible = (neighbors.len() * (neighbors.len() - 1)) / 2;
            triangles as f64 / possible as f64
        };

        // Average neighbor degree
        let avg_neighbor_deg = if adj[i].is_empty() {
            0.0
        } else {
            let sum: f64 = adj[i].iter().map(|&j| adj[j].len() as f64).sum();
            sum / adj[i].len() as f64
        };

        features.push(vec![degree, cc, avg_neighbor_deg]);
    }

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // WL Subtree Kernel
    // =========================================================================

    #[test]
    fn wl_identical_graphs() {
        // Triangle
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let labels = vec![1, 1, 1];

        let k_same = wl_subtree_kernel(&adj, &labels, &adj, &labels, 3);

        // Path graph (different structure)
        let adj2 = vec![vec![1], vec![0, 2], vec![1]];
        let labels2 = vec![1, 1, 1];

        let k_diff = wl_subtree_kernel(&adj, &labels, &adj2, &labels2, 3);

        assert!(
            k_same > k_diff,
            "identical graphs should have higher kernel than different, got same={k_same} diff={k_diff}"
        );
    }

    #[test]
    fn wl_symmetric() {
        let adj1 = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let labels1 = vec![1, 2, 1];

        let adj2 = vec![vec![1], vec![0, 2], vec![1]];
        let labels2 = vec![1, 1, 2];

        let k_12 = wl_subtree_kernel(&adj1, &labels1, &adj2, &labels2, 2);
        let k_21 = wl_subtree_kernel(&adj2, &labels2, &adj1, &labels1, 2);

        assert!(
            (k_12 - k_21).abs() < 1e-10,
            "WL kernel should be symmetric, got {k_12} vs {k_21}"
        );
    }

    #[test]
    fn wl_empty_graphs() {
        let adj: Vec<Vec<usize>> = vec![];
        let labels: Vec<u64> = vec![];

        let k = wl_subtree_kernel(&adj, &labels, &adj, &labels, 3);
        assert!((k - 0.0).abs() < 1e-10);
    }

    #[test]
    fn wl_different_labels_matter() {
        let adj = vec![vec![1], vec![0]];
        let labels_a = vec![1, 1];
        let labels_b = vec![1, 2];

        let k_same = wl_subtree_kernel(&adj, &labels_a, &adj, &labels_a, 1);
        let k_diff = wl_subtree_kernel(&adj, &labels_a, &adj, &labels_b, 1);

        assert!(
            k_same > k_diff,
            "same labels should give higher kernel than different labels"
        );
    }

    // =========================================================================
    // Random Walk Kernel
    // =========================================================================

    #[test]
    fn rw_self_geq_other() {
        // Triangle
        let adj1 = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        // Path
        let adj2 = vec![vec![1], vec![0, 2], vec![1]];

        let k_self = random_walk_kernel(&adj1, &adj1, 3, 0.1);
        let k_other = random_walk_kernel(&adj1, &adj2, 3, 0.1);

        assert!(
            k_self >= k_other,
            "self-kernel should be >= cross-kernel, got self={k_self} other={k_other}"
        );
    }

    #[test]
    fn rw_symmetric() {
        let adj1 = vec![vec![1, 2], vec![0], vec![0]];
        let adj2 = vec![vec![1], vec![0, 2], vec![1]];

        let k_12 = random_walk_kernel(&adj1, &adj2, 3, 0.1);
        let k_21 = random_walk_kernel(&adj2, &adj1, 3, 0.1);

        assert!(
            (k_12 - k_21).abs() < 1e-10,
            "random walk kernel should be symmetric, got {k_12} vs {k_21}"
        );
    }

    #[test]
    fn rw_empty_graph() {
        let adj1: Vec<Vec<usize>> = vec![];
        let adj2 = vec![vec![1], vec![0]];

        let k = random_walk_kernel(&adj1, &adj2, 3, 0.1);
        assert!((k - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rw_disconnected_nodes() {
        // Two isolated nodes (no edges)
        let adj = vec![vec![], vec![]];

        let k = random_walk_kernel(&adj, &adj, 3, 0.1);
        // Only length-0 walks exist: 2*2 = 4 product nodes, each with 1 walk
        assert!(
            (k - 4.0).abs() < 1e-10,
            "isolated nodes should have kernel = n1*n2, got {k}"
        );
    }

    // =========================================================================
    // Sliced Wasserstein Graph Kernel
    // =========================================================================

    #[test]
    fn sw_identical_graphs_kernel_one() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let k = sliced_wasserstein_graph_kernel(&features, &features, 100, 1.0, 42);
        assert!(
            (k - 1.0).abs() < 1e-10,
            "kernel of identical features should be 1.0, got {k}"
        );
    }

    #[test]
    fn sw_symmetric() {
        let f1 = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
        let f2 = vec![vec![5.0, 5.0], vec![6.0, 7.0], vec![8.0, 9.0]];

        let k_12 = sliced_wasserstein_graph_kernel(&f1, &f2, 100, 1.0, 42);
        let k_21 = sliced_wasserstein_graph_kernel(&f2, &f1, 100, 1.0, 42);

        assert!(
            (k_12 - k_21).abs() < 1e-10,
            "SW kernel should be symmetric, got {k_12} vs {k_21}"
        );
    }

    #[test]
    fn sw_in_unit_interval() {
        let f1 = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let f2 = vec![vec![10.0, 10.0], vec![11.0, 11.0]];

        let k = sliced_wasserstein_graph_kernel(&f1, &f2, 100, 1.0, 42);
        assert!(k > 0.0, "kernel should be positive, got {k}");
        assert!(k <= 1.0, "kernel should be <= 1, got {k}");
    }

    #[test]
    fn sw_similar_closer_than_dissimilar() {
        // Base features
        let f_base = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        // Small perturbation
        let f_similar = vec![vec![0.1, 0.05], vec![1.05, 0.95], vec![2.1, 1.9]];
        // Far away
        let f_dissimilar = vec![vec![50.0, 50.0], vec![51.0, 51.0], vec![52.0, 52.0]];

        let k_sim = sliced_wasserstein_graph_kernel(&f_base, &f_similar, 200, 1.0, 42);
        let k_dis = sliced_wasserstein_graph_kernel(&f_base, &f_dissimilar, 200, 1.0, 42);

        assert!(
            k_sim > k_dis,
            "similar features should give higher kernel: sim={k_sim} dis={k_dis}"
        );
    }

    #[test]
    fn sw_empty_returns_zero() {
        let empty: Vec<Vec<f64>> = vec![];
        let f = vec![vec![1.0, 2.0]];

        assert!((sliced_wasserstein_graph_kernel(&empty, &f, 50, 1.0, 0) - 0.0).abs() < 1e-15);
        assert!((sliced_wasserstein_graph_kernel(&f, &empty, 50, 1.0, 0) - 0.0).abs() < 1e-15);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn sw_gram_matrix_psd() {
        // Three small graphs with structural features
        let g1 = vec![vec![1, 2], vec![0, 2], vec![0, 1]]; // triangle
        let g2 = vec![vec![1], vec![0, 2], vec![1]]; // path
        let g3 = vec![vec![1, 2, 3], vec![0], vec![0], vec![0]]; // star

        let f1 = structural_node_features(&g1);
        let f2 = structural_node_features(&g2);
        let f3 = structural_node_features(&g3);

        let graphs = [&f1, &f2, &f3];
        let n = graphs.len();

        // Build Gram matrix
        let mut gram = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                gram[i][j] = sliced_wasserstein_graph_kernel(graphs[i], graphs[j], 200, 1.0, 42);
            }
        }

        // Check symmetric
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (gram[i][j] - gram[j][i]).abs() < 1e-10,
                    "Gram matrix not symmetric at ({i},{j})"
                );
            }
        }

        // Check PSD: all eigenvalues >= 0.
        // For 3x3, use characteristic polynomial or just check
        // all principal minors are non-negative (Sylvester's criterion).
        // Minor 1: gram[0][0] >= 0
        assert!(gram[0][0] >= -1e-10, "1x1 minor negative");
        // Minor 2: gram[0][0]*gram[1][1] - gram[0][1]^2 >= 0
        let m2 = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
        assert!(m2 >= -1e-10, "2x2 minor negative: {m2}");
        // Minor 3: determinant
        let det = gram[0][0] * (gram[1][1] * gram[2][2] - gram[1][2] * gram[2][1])
            - gram[0][1] * (gram[1][0] * gram[2][2] - gram[1][2] * gram[2][0])
            + gram[0][2] * (gram[1][0] * gram[2][1] - gram[1][1] * gram[2][0]);
        assert!(det >= -1e-10, "3x3 determinant negative: {det}");
    }

    // =========================================================================
    // Structural Node Features
    // =========================================================================

    #[test]
    fn structural_features_triangle() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let features = structural_node_features(&adj);

        assert_eq!(features.len(), 3);
        for f in &features {
            assert!((f[0] - 2.0).abs() < 1e-10, "degree should be 2");
            assert!((f[1] - 1.0).abs() < 1e-10, "clustering coeff should be 1.0");
            assert!(
                (f[2] - 2.0).abs() < 1e-10,
                "avg neighbor degree should be 2"
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn structural_features_star() {
        // Hub node 0 connected to leaves 1, 2, 3
        let adj = vec![vec![1, 2, 3], vec![0], vec![0], vec![0]];
        let features = structural_node_features(&adj);

        // Hub: degree 3, cc 0 (no edges between leaves), avg neighbor deg 1
        assert!((features[0][0] - 3.0).abs() < 1e-10, "hub degree");
        assert!((features[0][1] - 0.0).abs() < 1e-10, "hub cc");
        assert!((features[0][2] - 1.0).abs() < 1e-10, "hub avg neighbor deg");

        // Leaf: degree 1, cc 0, avg neighbor deg 3
        for i in 1..4 {
            assert!((features[i][0] - 1.0).abs() < 1e-10, "leaf degree");
            assert!((features[i][1] - 0.0).abs() < 1e-10, "leaf cc");
            assert!(
                (features[i][2] - 3.0).abs() < 1e-10,
                "leaf avg neighbor deg"
            );
        }
    }

    #[test]
    fn structural_features_isolated() {
        let adj: Vec<Vec<usize>> = vec![vec![], vec![]];
        let features = structural_node_features(&adj);

        for f in &features {
            assert!((f[0] - 0.0).abs() < 1e-10, "isolated degree");
            assert!((f[1] - 0.0).abs() < 1e-10, "isolated cc");
            assert!((f[2] - 0.0).abs() < 1e-10, "isolated avg neighbor deg");
        }
    }
}
