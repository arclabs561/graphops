//! PageRank on the real Cora citation network: a scale and sanity eval.
//!
//! Reports node count, score sum, positivity, and the top-3 scores. The exact
//! checks (uniform PageRank on a directed cycle and a complete graph) live in
//! `tests/pagerank_uniform.rs` and run under `cargo test`.
//!
//! Cora is not checked in. Without it this prints a skip message and exits 0.
//!
//! ```sh
//! ./scripts/fetch_cora_cites.sh
//! cargo run --release --example pagerank_validation
//! ```

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

fn main() -> ExitCode {
    let cites = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cora/cora.cites");
    if !cites.exists() {
        println!(
            "skip: {} not found; run ./scripts/fetch_cora_cites.sh first",
            cites.display()
        );
        return ExitCode::SUCCESS;
    }

    let content = match std::fs::read_to_string(&cites) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cannot read {}: {e}", cites.display());
            return ExitCode::FAILURE;
        }
    };
    let mut id_to_idx: HashMap<String, usize> = HashMap::new();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (lineno, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(a), Some(b)) = (it.next(), it.next()) else {
            eprintln!("{}:{}: expected two ids", cites.display(), lineno + 1);
            return ExitCode::FAILURE;
        };
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
    println!("Cora citation network");
    println!(
        "  nodes = {n}  sum = {sum:.6}  all positive = {all_pos}  top-3 scores = {:.4?}",
        ranked.iter().take(3).map(|(_, s)| *s).collect::<Vec<_>>()
    );

    // A probability vector with teleportation is strictly positive and sums to 1.
    if (sum - 1.0).abs() < 1e-4 && all_pos {
        ExitCode::SUCCESS
    } else {
        eprintln!("PageRank on Cora is not a positive probability vector");
        ExitCode::FAILURE
    }
}
