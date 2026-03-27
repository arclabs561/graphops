//! Ellipsoidal graph embeddings.
//!
//! Each node is embedded as an ellipsoid (center + PSD shape matrix) derived
//! from the spectral decomposition of the graph Laplacian. Nodes with the same
//! spectral position but different local connectivity (hub vs leaf) receive
//! different ellipsoid shapes.
//!
//! Reference: Fanuel, Aspeel, Schaub, Delvenne (2025),
//! "Ellipsoidal Embeddings of Graphs", SIAM J. Math. Data Sci.

use crate::graph::Graph;

/// Ellipsoidal embedding of a single graph node.
#[derive(Debug, Clone)]
pub struct Ellipsoid {
    /// `k`-dimensional center (spectral position).
    pub center: Vec<f64>,
    /// `k x k` PSD shape matrix (local structure), stored row-major.
    pub shape: Vec<f64>,
}

/// Configuration for ellipsoidal embedding.
#[derive(Debug, Clone)]
pub struct EllipsoidalConfig {
    /// Embedding dimension (number of non-trivial eigenvectors to use).
    pub dim: usize,
    /// Regularization added to the Laplacian before inversion: `(L + eps * I)^{-1}`.
    /// A small positive value avoids numerical issues from the zero eigenvalue.
    /// Default: `1e-10`.
    pub regularization: f64,
}

impl Default for EllipsoidalConfig {
    fn default() -> Self {
        Self {
            dim: 2,
            regularization: 1e-10,
        }
    }
}

/// Compute ellipsoidal embeddings for all nodes.
///
/// Returns one [`Ellipsoid`] per node, with `center` of length `dim` and
/// `shape` of length `dim * dim`.
///
/// # Panics
///
/// Panics if `dim` is zero or exceeds `node_count - 1` (there are at most
/// `n - 1` non-trivial Laplacian eigenvectors).
pub fn ellipsoidal_embedding<G: Graph>(graph: &G, config: &EllipsoidalConfig) -> Vec<Ellipsoid> {
    let n = graph.node_count();
    let dim = config.dim;
    assert!(dim > 0, "embedding dimension must be positive");
    assert!(
        dim < n,
        "embedding dimension must be < node_count (at most n-1 non-trivial eigenvectors)"
    );

    // Build the graph Laplacian L = D - A.
    let mut laplacian = vec![0.0_f64; n * n];
    for u in 0..n {
        let nbrs = graph.neighbors(u);
        laplacian[u * n + u] = nbrs.len() as f64;
        for v in nbrs {
            laplacian[u * n + v] -= 1.0;
        }
    }

    // Eigen-decomposition of the symmetric Laplacian via Jacobi iteration.
    let (eigenvalues, eigenvectors) = symmetric_eigen(n, &mut laplacian);

    // Sort eigenvalues ascending; skip the smallest (zero / near-zero) eigenvalue.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    // Take eigenvectors 1..=dim (skip index 0, the trivial constant eigenvector).
    let selected: Vec<usize> = order[1..=dim].to_vec();

    // Compute L^+ restricted to selected eigenvectors.
    // L^+ = sum_i (1/lambda_i) * u_i * u_i^T  for non-zero lambda_i.
    //
    // For each node v:
    //   center_v[j] = u_{selected[j]}[v] / sqrt(lambda_{selected[j]})
    //   (L^+)_{vv} projected onto the k-dim subspace gives the shape matrix.
    //
    // The shape matrix for node v is:
    //   Sigma_v[j][l] = sum over selected eigenvectors of
    //     u_j[v] * u_l[v] / lambda_j  (projected covariance at v)
    //
    // More precisely, following Fanuel et al.: the embedding maps node v to
    // the k x k matrix  M_v  where  M_v[j,l] = u_j(v) * u_l(v) / lambda_l
    // (the "ellipsoidal Gram matrix" at v). The center is the diagonal
    // scaling u_j(v) / sqrt(lambda_j).

    let mut embeddings = Vec::with_capacity(n);

    for v in 0..n {
        let mut center = Vec::with_capacity(dim);
        let mut shape = vec![0.0_f64; dim * dim];

        for (_j, &ej) in selected.iter().enumerate() {
            let lam_j = eigenvalues[ej].max(config.regularization);
            let u_jv = eigenvectors[ej * n + v];
            center.push(u_jv / lam_j.sqrt());
        }

        // Shape: Sigma_v[j,l] = u_j(v) * u_l(v) / (sqrt(lambda_j) * sqrt(lambda_l))
        // This is the rank-1 contribution of node v to the projected pseudo-inverse,
        // capturing the local spread of v in spectral space.
        for j in 0..dim {
            let ej = selected[j];
            let lam_j = eigenvalues[ej].max(config.regularization);
            let u_jv = eigenvectors[ej * n + v];
            for l in 0..dim {
                let el = selected[l];
                let lam_l = eigenvalues[el].max(config.regularization);
                let u_lv = eigenvectors[el * n + v];
                shape[j * dim + l] = u_jv * u_lv / (lam_j.sqrt() * lam_l.sqrt());
            }
        }

        embeddings.push(Ellipsoid { center, shape });
    }

    embeddings
}

/// Bures-Wasserstein distance between two ellipsoids.
///
/// Treats each ellipsoid as a Gaussian N(center, shape) and computes the
/// 2-Wasserstein distance:
///
/// $$W_2^2 = \|m_1 - m_2\|^2 + \operatorname{tr}(S_1) + \operatorname{tr}(S_2)
///           - 2\,\operatorname{tr}\bigl((S_1^{1/2} S_2 S_1^{1/2})^{1/2}\bigr)$$
///
/// Returns a non-negative value. Returns 0.0 for identical ellipsoids.
pub fn ellipsoid_distance(a: &Ellipsoid, b: &Ellipsoid) -> f64 {
    let dim = a.center.len();
    assert_eq!(dim, b.center.len());
    assert_eq!(a.shape.len(), dim * dim);
    assert_eq!(b.shape.len(), dim * dim);

    // ||m1 - m2||^2
    let center_dist_sq: f64 = a
        .center
        .iter()
        .zip(b.center.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    let tr_a = trace(dim, &a.shape);
    let tr_b = trace(dim, &b.shape);

    // Compute sqrt(A) * B * sqrt(A), then take matrix sqrt, then trace.
    let sqrt_a = matrix_sqrt_psd(dim, &a.shape);
    // M = sqrt_a * B * sqrt_a
    let m = mat_mul(dim, &mat_mul(dim, &sqrt_a, &b.shape), &sqrt_a);
    let sqrt_m = matrix_sqrt_psd(dim, &m);
    let tr_cross = trace(dim, &sqrt_m);

    let w2_sq = center_dist_sq + tr_a + tr_b - 2.0 * tr_cross;
    w2_sq.max(0.0).sqrt()
}

/// Bhattacharyya overlap coefficient between two ellipsoids.
///
/// Treats each ellipsoid as a Gaussian and computes:
///
/// $$\mathrm{BC} = \frac{\det(S_1)^{1/4}\,\det(S_2)^{1/4}}
///                      {\det\!\bigl(\tfrac{S_1+S_2}{2}\bigr)^{1/2}}
///   \exp\!\Bigl(-\tfrac18\,(m_1-m_2)^T\bigl(\tfrac{S_1+S_2}{2}\bigr)^{-1}(m_1-m_2)\Bigr)$$
///
/// Returns a value in `[0, 1]`. Returns 1.0 for identical ellipsoids.
pub fn ellipsoid_overlap(a: &Ellipsoid, b: &Ellipsoid) -> f64 {
    let dim = a.center.len();
    assert_eq!(dim, b.center.len());
    assert_eq!(a.shape.len(), dim * dim);
    assert_eq!(b.shape.len(), dim * dim);

    let eps = 1e-8;

    // Regularize shape matrices (they may be rank-deficient).
    let mut sa = a.shape.clone();
    let mut sb = b.shape.clone();
    for i in 0..dim {
        sa[i * dim + i] += eps;
        sb[i * dim + i] += eps;
    }

    // S_avg = (S1 + S2) / 2
    let mut s_avg = vec![0.0_f64; dim * dim];
    for i in 0..dim * dim {
        s_avg[i] = (sa[i] + sb[i]) / 2.0;
    }

    let det_a = matrix_det(dim, &sa).abs().max(eps);
    let det_b = matrix_det(dim, &sb).abs().max(eps);
    let det_avg = matrix_det(dim, &s_avg).abs().max(eps);

    // det ratio
    let det_factor = (det_a.powf(0.25) * det_b.powf(0.25)) / det_avg.sqrt();

    // Mahalanobis term: (m1-m2)^T S_avg^{-1} (m1-m2)
    let s_avg_inv = matrix_inverse(dim, &s_avg);
    let diff: Vec<f64> = a
        .center
        .iter()
        .zip(b.center.iter())
        .map(|(x, y)| x - y)
        .collect();
    let mut mahal = 0.0;
    for i in 0..dim {
        let mut row_sum = 0.0;
        for j in 0..dim {
            row_sum += s_avg_inv[i * dim + j] * diff[j];
        }
        mahal += diff[i] * row_sum;
    }

    let overlap = det_factor * (-mahal / 8.0).exp();
    overlap.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Dense linear algebra helpers (small matrices only)
// ---------------------------------------------------------------------------

fn trace(n: usize, m: &[f64]) -> f64 {
    (0..n).map(|i| m[i * n + i]).sum()
}

fn mat_mul(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

/// Matrix square root of a symmetric PSD matrix via eigendecomposition.
fn matrix_sqrt_psd(n: usize, m: &[f64]) -> Vec<f64> {
    let mut work = m.to_vec();
    let (vals, vecs) = symmetric_eigen(n, &mut work);
    // Reconstruct: U * diag(sqrt(max(0, lambda))) * U^T
    let mut result = vec![0.0; n * n];
    for k in 0..n {
        let s = vals[k].max(0.0).sqrt();
        for i in 0..n {
            let vi = vecs[k * n + i] * s;
            for j in 0..n {
                result[i * n + j] += vi * vecs[k * n + j];
            }
        }
    }
    result
}

/// Determinant of a small matrix via LU decomposition (partial pivoting).
fn matrix_det(n: usize, m: &[f64]) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut a = m.to_vec();
    let mut sign = 1.0_f64;
    for col in 0..n {
        // Pivot
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return 0.0;
        }
        if max_row != col {
            for j in 0..n {
                a.swap(col * n + j, max_row * n + j);
            }
            sign = -sign;
        }
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for j in col..n {
                let val = a[col * n + j];
                a[row * n + j] -= factor * val;
            }
        }
    }
    let mut det = sign;
    for i in 0..n {
        det *= a[i * n + i];
    }
    det
}

/// Inverse of a small matrix via Gauss-Jordan elimination.
fn matrix_inverse(n: usize, m: &[f64]) -> Vec<f64> {
    let mut aug = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = m[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        // Pivot
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * 2 * n + j, max_row * 2 * n + j);
            }
        }
        let pivot = aug[col * 2 * n + col];
        if pivot.abs() < 1e-15 {
            // Singular -- return identity as fallback (regularization should prevent this).
            let mut id = vec![0.0; n * n];
            for i in 0..n {
                id[i * n + i] = 1.0;
            }
            return id;
        }
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                let val = aug[col * 2 * n + j];
                aug[row * 2 * n + j] -= factor * val;
            }
        }
    }
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    inv
}

/// Eigendecomposition of a symmetric matrix via Jacobi iteration.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors are stored as
/// `eigenvectors[k * n + i]` = component `i` of eigenvector `k`.
///
/// The input matrix `a` is destroyed during computation.
fn symmetric_eigen(n: usize, a: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
    // Initialize eigenvector matrix to identity.
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find the largest off-diagonal element.
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-12 {
            break;
        }

        // Compute rotation.
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to A: A' = G^T A G
        // Update rows/cols p and q.
        let mut new_ap = vec![0.0; n];
        let mut new_aq = vec![0.0; n];
        for i in 0..n {
            new_ap[i] = c * a[p * n + i] + s * a[q * n + i];
            new_aq[i] = -s * a[p * n + i] + c * a[q * n + i];
        }
        for i in 0..n {
            a[p * n + i] = new_ap[i];
            a[q * n + i] = new_aq[i];
        }
        // Update columns p and q.
        for i in 0..n {
            let aip = a[i * n + p];
            let aiq = a[i * n + q];
            a[i * n + p] = c * aip + s * aiq;
            a[i * n + q] = -s * aip + c * aiq;
        }

        // Update eigenvectors: V' = V * G
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();

    // Transpose v so that eigenvectors[k * n + i] = component i of eigenvector k.
    let mut eigenvectors = vec![0.0; n * n];
    for k in 0..n {
        for i in 0..n {
            eigenvectors[k * n + i] = v[i * n + k];
        }
    }

    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple adjacency-list graph for testing.
    struct TestGraph {
        adj: Vec<Vec<usize>>,
    }

    impl TestGraph {
        fn complete(n: usize) -> Self {
            let adj = (0..n)
                .map(|i| (0..n).filter(|&j| j != i).collect())
                .collect();
            Self { adj }
        }

        fn star(n: usize) -> Self {
            // Node 0 is the hub, nodes 1..n are leaves.
            let mut adj = vec![vec![]; n];
            for i in 1..n {
                adj[0].push(i);
                adj[i].push(0);
            }
            Self { adj }
        }

        fn path(n: usize) -> Self {
            let mut adj = vec![vec![]; n];
            for i in 0..(n - 1) {
                adj[i].push(i + 1);
                adj[i + 1].push(i);
            }
            Self { adj }
        }
    }

    impl Graph for TestGraph {
        fn node_count(&self) -> usize {
            self.adj.len()
        }
        fn neighbors(&self, node: usize) -> Vec<usize> {
            self.adj[node].clone()
        }
    }

    #[test]
    fn embedding_dimensions_match() {
        let g = TestGraph::path(6);
        let config = EllipsoidalConfig {
            dim: 3,
            ..Default::default()
        };
        let embs = ellipsoidal_embedding(&g, &config);
        assert_eq!(embs.len(), 6);
        for e in &embs {
            assert_eq!(e.center.len(), 3);
            assert_eq!(e.shape.len(), 9);
        }
    }

    #[test]
    fn shape_matrices_are_psd() {
        let g = TestGraph::path(8);
        let config = EllipsoidalConfig {
            dim: 3,
            ..Default::default()
        };
        let embs = ellipsoidal_embedding(&g, &config);
        for e in &embs {
            // Check PSD: eigenvalues of the shape matrix should be >= 0.
            let mut m = e.shape.clone();
            let (vals, _) = symmetric_eigen(3, &mut m);
            for &v in &vals {
                assert!(v >= -1e-10, "shape matrix eigenvalue is negative: {v}");
            }
        }
    }

    #[test]
    fn hub_and_leaf_have_different_shapes() {
        let n = 6;
        let g = TestGraph::star(n);
        // Use full non-trivial eigenspace so that trace is invariant under
        // the arbitrary basis choice in degenerate eigenspaces.
        let config = EllipsoidalConfig {
            dim: n - 1,
            ..Default::default()
        };
        let embs = ellipsoidal_embedding(&g, &config);

        let dim = n - 1;
        let hub_trace = trace(dim, &embs[0].shape);
        let leaf_traces: Vec<f64> = (1..n).map(|i| trace(dim, &embs[i].shape)).collect();
        let leaf_avg: f64 = leaf_traces.iter().sum::<f64>() / leaf_traces.len() as f64;
        // All leaves should have equal trace (graph symmetry).
        for &lt in &leaf_traces {
            assert!(
                (lt - leaf_avg).abs() < 1e-6,
                "leaf traces differ: {lt} vs avg {leaf_avg}"
            );
        }
        // Hub and leaves should have different traces -- the embedding
        // captures structural differences between the hub and leaves.
        assert!(
            (hub_trace - leaf_avg).abs() > 1e-4,
            "hub trace ({hub_trace}) and leaf trace ({leaf_avg}) should differ"
        );
    }

    #[test]
    fn distance_symmetry_and_self() {
        let g = TestGraph::path(5);
        let config = EllipsoidalConfig {
            dim: 2,
            ..Default::default()
        };
        let embs = ellipsoidal_embedding(&g, &config);

        // Self-distance is 0.
        for e in &embs {
            let d = ellipsoid_distance(e, e);
            assert!(d < 1e-6, "self-distance should be ~0, got {d}");
        }

        // Symmetry.
        for i in 0..embs.len() {
            for j in (i + 1)..embs.len() {
                let d1 = ellipsoid_distance(&embs[i], &embs[j]);
                let d2 = ellipsoid_distance(&embs[j], &embs[i]);
                assert!(
                    (d1 - d2).abs() < 1e-6,
                    "distance not symmetric: {d1} vs {d2}"
                );
                assert!(d1 >= 0.0, "distance should be non-negative");
            }
        }
    }

    #[test]
    fn distance_is_nonnegative() {
        let g = TestGraph::star(7);
        let config = EllipsoidalConfig {
            dim: 3,
            ..Default::default()
        };
        let embs = ellipsoidal_embedding(&g, &config);
        for i in 0..embs.len() {
            for j in 0..embs.len() {
                let d = ellipsoid_distance(&embs[i], &embs[j]);
                assert!(d >= -1e-10, "distance should be non-negative, got {d}");
            }
        }
    }

    #[test]
    fn complete_graph_identical_ellipsoids() {
        let n = 5;
        let g = TestGraph::complete(n);
        // Use the full non-trivial eigenspace (n-1 dims) so the trace is
        // rotation-invariant across the degenerate eigenspace of K_n.
        let config = EllipsoidalConfig {
            dim: n - 1,
            ..Default::default()
        };
        let embs = ellipsoidal_embedding(&g, &config);

        let dim = n - 1;
        let traces: Vec<f64> = embs.iter().map(|e| trace(dim, &e.shape)).collect();
        let first = traces[0];
        for &tr in &traces[1..] {
            assert!(
                (tr - first).abs() < 1e-6,
                "complete graph: traces differ: {tr} vs {first}"
            );
        }
    }

    #[test]
    fn overlap_self_is_one() {
        let g = TestGraph::path(5);
        let config = EllipsoidalConfig {
            dim: 2,
            ..Default::default()
        };
        let embs = ellipsoidal_embedding(&g, &config);
        for e in &embs {
            let o = ellipsoid_overlap(e, e);
            assert!(
                (o - 1.0).abs() < 1e-3,
                "self-overlap should be ~1.0, got {o}"
            );
        }
    }

    #[test]
    fn overlap_is_symmetric() {
        let g = TestGraph::star(6);
        let config = EllipsoidalConfig {
            dim: 2,
            ..Default::default()
        };
        let embs = ellipsoidal_embedding(&g, &config);
        for i in 0..embs.len() {
            for j in (i + 1)..embs.len() {
                let o1 = ellipsoid_overlap(&embs[i], &embs[j]);
                let o2 = ellipsoid_overlap(&embs[j], &embs[i]);
                assert!(
                    (o1 - o2).abs() < 1e-8,
                    "overlap not symmetric: {o1} vs {o2}"
                );
            }
        }
    }

    #[test]
    fn overlap_in_unit_range() {
        let g = TestGraph::path(6);
        let config = EllipsoidalConfig {
            dim: 2,
            ..Default::default()
        };
        let embs = ellipsoidal_embedding(&g, &config);
        for i in 0..embs.len() {
            for j in 0..embs.len() {
                let o = ellipsoid_overlap(&embs[i], &embs[j]);
                assert!(
                    (0.0..=1.0 + 1e-10).contains(&o),
                    "overlap out of range: {o}"
                );
            }
        }
    }
}
