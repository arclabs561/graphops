//! PageRank on a small directed graph.
//!
//! Graph (4 nodes):
//!   0 -> 1, 0 -> 2
//!   1 -> 2
//!   2 -> 0
//!   3 -> 2

use graphops::graph::AdjacencyMatrix;
use graphops::{pagerank, PageRankConfig};

fn main() {
    // Adjacency matrix: adj[i][j] > 0 means edge i -> j.
    let adj = vec![
        vec![0.0, 1.0, 1.0, 0.0], // 0 -> 1, 0 -> 2
        vec![0.0, 0.0, 1.0, 0.0], // 1 -> 2
        vec![1.0, 0.0, 0.0, 0.0], // 2 -> 0
        vec![0.0, 0.0, 1.0, 0.0], // 3 -> 2
    ];
    let graph = AdjacencyMatrix(&adj);

    let config = PageRankConfig {
        damping: 0.85,
        max_iterations: 100,
        tolerance: 1e-8,
    };

    let scores = pagerank(&graph, config);

    println!("PageRank scores (damping={}):", config.damping);
    let labels = ["A", "B", "C", "D"];
    for (i, score) in scores.iter().enumerate() {
        println!("  node {} ({}): {:.6}", i, labels[i], score);
    }

    let sum: f64 = scores.iter().sum();
    println!("\nSum of scores: {:.6} (should be ~1.0)", sum);

    // Node 2 (C) should rank highest: it receives edges from 0, 1, and 3.
    let max_node = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!(
        "Highest-ranked node: {} ({}) with score {:.6}",
        max_node.0, labels[max_node.0], max_node.1
    );
}
