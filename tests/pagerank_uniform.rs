//! PageRank on graphs whose stationary distribution is exactly uniform.
//!
//! A directed cycle and a complete graph are both regular, so PageRank returns
//! 1/n for every node regardless of the damping and dangling convention. Any
//! deviation is a bug.

// Index loops are the clearest form for dense adjacency-matrix construction.
#![allow(clippy::needless_range_loop)]

use graphops::graph::AdjacencyMatrix;
use graphops::{pagerank, PageRankConfig};

fn config() -> PageRankConfig {
    PageRankConfig {
        damping: 0.85,
        max_iterations: 200,
        tolerance: 1e-10,
    }
}

fn directed_cycle(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    for i in 0..n {
        adj[i][(i + 1) % n] = 1.0;
    }
    adj
}

fn complete(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                adj[i][j] = 1.0;
            }
        }
    }
    adj
}

fn assert_uniform(name: &str, adj: &[Vec<f64>]) {
    let scores = pagerank(&AdjacencyMatrix(adj), config());
    assert_eq!(scores.len(), adj.len(), "{name}: score length");
    let u = 1.0 / scores.len() as f64;
    let dev = scores.iter().map(|&s| (s - u).abs()).fold(0.0, f64::max);
    let sum: f64 = scores.iter().sum();
    assert!(dev < 1e-6, "{name}: max |score - 1/n| = {dev:.2e}");
    assert!((sum - 1.0).abs() < 1e-6, "{name}: sum = {sum}");
}

#[test]
fn directed_cycle_is_uniform() {
    assert_uniform("directed cycle n=64", &directed_cycle(64));
}

#[test]
fn complete_graph_is_uniform() {
    assert_uniform("complete graph n=30", &complete(30));
}
