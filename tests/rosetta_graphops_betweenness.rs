// The whole file is gated on `petgraph` (`betweenness_centrality` lives behind
// it); with the feature off it compiles to nothing.
#![cfg(feature = "petgraph")]
//! Rosetta correctness fixtures: graphops Brandes betweenness centrality
//! asserted against networkx.
//!
//! Reference values in `fixtures/rosetta/graphops_betweenness.json` come from
//! `gen_graphops_betweenness.py` (their provenance). graphops and networkx use
//! the same directed Brandes convention: shortest-path counting with the sigma
//! split over multiple shortest paths, endpoints excluded, and directed
//! normalization 1 / ((n-1)(n-2)). networkx is called with `normalized=True,
//! endpoints=False` (its defaults).
//!
//! EXACT tolerance class: unweighted integer-hop Brandes is exact rational
//! arithmetic on both sides, so the only gap is f64 rounding; comparison is 1e-9.
//!
//! Covered: path (single shortest path per pair), diamond (two equal-length
//! shortest paths, so the sigma split is exercised), and a fixed-seed random DAG.
//! This is the deferred Brandes half of `rosetta_graphops.rs`.
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_graphops_betweenness.py`.

use graphops::betweenness_centrality;
use petgraph::prelude::*;
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/rosetta/graphops_betweenness.json");

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    n: usize,
    edges: Vec<[usize; 2]>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    betweenness: Vec<f64>,
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

fn build_digraph(n: usize, edges: &[[usize; 2]]) -> DiGraph<(), ()> {
    let mut g: DiGraph<(), ()> = DiGraph::new();
    for _ in 0..n {
        g.add_node(());
    }
    for &[u, v] in edges {
        g.add_edge(NodeIndex::new(u), NodeIndex::new(v), ());
    }
    g
}

#[test]
fn rosetta_betweenness_matches_networkx() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    assert!(!fx.cases.is_empty(), "fixture has no cases");

    for case in &fx.cases {
        let g = build_digraph(case.n, &case.edges);
        let bc = betweenness_centrality(&g);
        vec_close(&bc, &case.expected.betweenness, &case.name);
    }
}
