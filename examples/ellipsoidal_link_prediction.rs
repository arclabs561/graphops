//! Ellipsoidal embeddings for link prediction.
//!
//! Embeds graph nodes as ellipsoids (center + shape matrix), then predicts
//! missing edges using ellipsoid overlap as a link score. Higher overlap
//! between two node ellipsoids suggests a more likely connection.
//!
//! This demonstrates the connection between graphops::ellipsoidal (unsupervised
//! spectral embedding) and subsume-style region embeddings (supervised KG
//! embedding): both use geometric regions for relational reasoning.

use graphops::ellipsoidal::{ellipsoid_distance, ellipsoid_overlap, ellipsoidal_embedding, EllipsoidalConfig};
use graphops::Graph;

struct VecGraph {
    adj: Vec<Vec<usize>>,
}

impl Graph for VecGraph {
    fn node_count(&self) -> usize { self.adj.len() }
    fn neighbors(&self, node: usize) -> Vec<usize> { self.adj[node].clone() }
}

fn main() {
    // Karate club-like graph: two dense clusters with a few bridges.
    let adj = vec![
        vec![1, 2, 3, 4],     // 0: cluster A hub
        vec![0, 2, 3],        // 1: cluster A
        vec![0, 1, 3],        // 2: cluster A
        vec![0, 1, 2, 7],     // 3: cluster A, bridge to B
        vec![0, 5],           // 4: cluster A, bridge to B
        vec![4, 6, 7, 8],     // 5: cluster B
        vec![5, 7, 8],        // 6: cluster B
        vec![3, 5, 6, 8],     // 7: cluster B hub
        vec![5, 6, 7],        // 8: cluster B
    ];
    let g = VecGraph { adj };

    let config = EllipsoidalConfig { dim: 3, ..Default::default() };
    let embeddings = ellipsoidal_embedding(&g, &config);

    println!("Ellipsoidal link prediction");
    println!("  {} nodes, dim={}", g.node_count(), config.dim);
    println!();

    // Show embeddings.
    for (i, e) in embeddings.iter().enumerate() {
        println!(
            "  node {i}: center=[{:.3}, {:.3}, {:.3}]",
            e.center[0], e.center[1], e.center[2],
        );
    }
    println!();

    // Compute overlap scores for all non-edges as link predictions.
    let n = Graph::node_count(&g);
    let mut predictions: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            // Skip existing edges.
            if g.adj[i].contains(&j) {
                continue;
            }
            let overlap = ellipsoid_overlap(&embeddings[i], &embeddings[j])
                .unwrap_or(0.0);
            predictions.push((i, j, overlap));
        }
    }

    // Sort by overlap (descending = most likely link).
    predictions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!("Top predicted links (by ellipsoid overlap):");
    for &(i, j, score) in predictions.iter().take(5) {
        let dist = ellipsoid_distance(&embeddings[i], &embeddings[j]).unwrap_or(f64::NAN);
        println!("  ({i}, {j}): overlap={score:.4}, distance={dist:.4}");
    }
    println!();

    // Compute intra-cluster vs inter-cluster distances.
    let cluster_a = [0, 1, 2, 3];
    let cluster_b = [5, 6, 7, 8];

    let intra_a: f64 = pairwise_mean_distance(&embeddings, &cluster_a);
    let intra_b: f64 = pairwise_mean_distance(&embeddings, &cluster_b);
    let inter: f64 = cross_mean_distance(&embeddings, &cluster_a, &cluster_b);

    println!("Cluster analysis:");
    println!("  Cluster A intra-distance: {intra_a:.4}");
    println!("  Cluster B intra-distance: {intra_b:.4}");
    println!("  Inter-cluster distance:   {inter:.4}");
    if inter > intra_a && inter > intra_b {
        println!("  Inter > intra (clusters well-separated in embedding)");
    }
}

fn pairwise_mean_distance(
    embeddings: &[graphops::ellipsoidal::Ellipsoid],
    nodes: &[usize],
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for (i, &a) in nodes.iter().enumerate() {
        for &b in &nodes[i + 1..] {
            sum += ellipsoid_distance(&embeddings[a], &embeddings[b]).unwrap_or(0.0);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn cross_mean_distance(
    embeddings: &[graphops::ellipsoidal::Ellipsoid],
    a: &[usize],
    b: &[usize],
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for &i in a {
        for &j in b {
            sum += ellipsoid_distance(&embeddings[i], &embeddings[j]).unwrap_or(0.0);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}
