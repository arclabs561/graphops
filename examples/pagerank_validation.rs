//! Validate graphops PageRank against exact references, plus a real graph.
//!
//! Two graphs have a provably uniform stationary distribution regardless of the
//! damping/dangling convention: a directed cycle and a complete graph. PageRank
//! on either must return 1/n for every node; deviation is a bug. Then it runs on
//! the real Cora citation network (fetch first) as a sanity + scale check.
//!
//! ```sh
//! ./scripts/fetch_cora_cites.sh   # optional; the exact checks run without it
//! cargo run --release --example pagerank_validation
//! ```

// Index loops are the clearest form for dense adjacency-matrix construction.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;

use graphops::graph::AdjacencyMatrix;
use graphops::{pagerank, PageRankConfig};

fn config() -> PageRankConfig {
    PageRankConfig {
        damping: 0.85,
        max_iterations: 200,
        tolerance: 1e-10,
    }
}

/// Max absolute deviation of `scores` from the uniform distribution 1/n.
fn max_dev_from_uniform(scores: &[f64]) -> f64 {
    let u = 1.0 / scores.len() as f64;
    scores.iter().map(|&s| (s - u).abs()).fold(0.0, f64::max)
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

fn main() -> ExitCode {
    let mut ok = true;

    println!("=== exact references (uniform stationary distribution) ===");
    for (name, adj) in [
        ("directed cycle n=64", directed_cycle(64)),
        ("complete graph n=30", complete(30)),
    ] {
        let scores = pagerank(&AdjacencyMatrix(&adj), config());
        let dev = max_dev_from_uniform(&scores);
        let sum: f64 = scores.iter().sum();
        let pass = dev < 1e-6 && (sum - 1.0).abs() < 1e-6;
        ok &= pass;
        println!(
            "  {name}: max|score - 1/n| = {dev:.2e}  sum = {sum:.6}  [{}]",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cora");
    let cites = dir.join("cora.cites");
    if cites.exists() {
        println!("\n=== real graph: Cora citation network ===");
        let content = std::fs::read_to_string(&cites).unwrap();
        let mut id_to_idx: HashMap<String, usize> = HashMap::new();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            let mut it = line.split_whitespace();
            let (a, b) = (it.next().unwrap(), it.next().unwrap());
            let mut idx = |k: &str| -> usize {
                let n = id_to_idx.len();
                *id_to_idx.entry(k.to_string()).or_insert(n)
            };
            // cora.cites is "cited citing"; PageRank flows along citing -> cited.
            let (cited, citing) = (idx(a), idx(b));
            edges.push((citing, cited));
        }
        let n = id_to_idx.len();
        let mut adj = vec![vec![0.0; n]; n];
        for (s, d) in edges {
            adj[s][d] = 1.0;
        }
        let scores = pagerank(&AdjacencyMatrix(&adj), config());
        let sum: f64 = scores.iter().sum();
        let all_pos = scores.iter().all(|&s| s > 0.0);
        let mut ranked: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!(
            "  nodes = {n}  sum = {sum:.6}  all positive = {all_pos}  top-3 scores = {:.4?}",
            ranked.iter().take(3).map(|(_, s)| *s).collect::<Vec<_>>()
        );
        ok &= (sum - 1.0).abs() < 1e-4 && all_pos;
    } else {
        println!(
            "\n(Cora not fetched; run ./scripts/fetch_cora_cites.sh for the real-graph check)"
        );
    }

    if ok {
        ExitCode::SUCCESS
    } else {
        eprintln!("\nVALIDATION FAILED");
        ExitCode::FAILURE
    }
}
