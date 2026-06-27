//! Rosetta correctness fixtures: graphops PageRank asserted against networkx.
//!
//! Reference values in `fixtures/rosetta/graphops_pagerank.json` come from
//! `gen_graphops.py` (their provenance). graphops and networkx use the same
//! random-surfer model (uniform teleport, dangling mass redistributed
//! uniformly), so the unique PageRank fixed point agrees once both are
//! converged tightly. The two stop rules differ (graphops: L1 delta < tol;
//! networkx: L1 delta < n*tol), so both run far tighter than the 1e-9
//! comparison to remove that as a variable.
//!
//! Covered: pagerank (unweighted, with a dangling node) and pagerank_weighted.
//! Deferred: Brandes betweenness (petgraph-feature-gated) and connected
//! components (no single graphops function maps cleanly to networkx).
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_graphops.py`.

use graphops::{pagerank, pagerank_weighted, AdjacencyMatrix, PageRankConfig};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/rosetta/graphops_pagerank.json");

#[derive(Deserialize)]
struct Fixture {
    params: Params,
    adj_unweighted: Vec<Vec<f64>>,
    adj_weighted: Vec<Vec<f64>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Params {
    damping: f64,
    graphops_tolerance: f64,
    max_iterations: usize,
}

#[derive(Deserialize)]
struct Expected {
    pagerank: Vec<f64>,
    pagerank_weighted: Vec<f64>,
}

fn vec_close(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-9 * (1.0 + w.abs());
        let diff = (g - w).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: graphops={g} networkx={w} diff={diff} tol={tol}"
        );
    }
}

#[test]
fn rosetta_pagerank_matches_networkx() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let config = PageRankConfig {
        damping: fx.params.damping,
        max_iterations: fx.params.max_iterations,
        tolerance: fx.params.graphops_tolerance,
    };

    let scores = pagerank(&AdjacencyMatrix(&fx.adj_unweighted), config);
    vec_close(&scores, &fx.expected.pagerank, "pagerank");

    let wscores = pagerank_weighted(&AdjacencyMatrix(&fx.adj_weighted), config);
    vec_close(
        &wscores,
        &fx.expected.pagerank_weighted,
        "pagerank_weighted",
    );
}
