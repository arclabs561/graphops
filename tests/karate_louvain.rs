//! Louvain on Zachary's karate club against networkx.
//!
//! Edges are `networkx.karate_club_graph()` (78 edges, unweighted). Over seeds
//! 0..20, `networkx.algorithms.community.louvain_communities` reaches
//! modularity 0.3952 to 0.4198, and 0.4198 is the known optimum for this
//! graph. Modularity is computed here from its definition, independently of
//! `graphops::modularity`.

use graphops::{
    louvain_connected_seeded, louvain_seeded, louvain_weighted_seeded, AdjacencyMatrix, GraphRef,
};

const KARATE_EDGES: [(usize, usize); 78] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
];

const N: usize = 34;

struct Adj(Vec<Vec<usize>>);

impl GraphRef for Adj {
    fn node_count(&self) -> usize {
        self.0.len()
    }
    fn neighbors_ref(&self, node: usize) -> &[usize] {
        &self.0[node]
    }
}

fn karate() -> Adj {
    let mut adj = vec![Vec::new(); N];
    for &(u, v) in &KARATE_EDGES {
        adj[u].push(v);
        adj[v].push(u);
    }
    Adj(adj)
}

/// Newman-Girvan modularity: Q = sum_c [L_c / m - (d_c / 2m)^2].
fn modularity(labels: &[usize]) -> f64 {
    let m = KARATE_EDGES.len() as f64;
    let n_comm = labels.iter().max().map_or(0, |&c| c + 1);
    let mut internal = vec![0.0; n_comm];
    let mut degree = vec![0.0; n_comm];
    for &(u, v) in &KARATE_EDGES {
        let (cu, cv) = (labels[u], labels[v]);
        degree[cu] += 1.0;
        degree[cv] += 1.0;
        if cu == cv {
            internal[cu] += 1.0;
        }
    }
    (0..n_comm)
        .map(|c| internal[c] / m - (degree[c] / (2.0 * m)).powi(2))
        .sum()
}

fn assert_in_networkx_range(name: &str, labels: &[usize]) {
    assert_eq!(labels.len(), N);
    let q = modularity(labels);
    assert!(
        (0.38..=0.4198 + 1e-4).contains(&q),
        "{name}: modularity {q:.4}, networkx range 0.3952..0.4198, labels {labels:?}"
    );
}

#[test]
fn louvain_karate_modularity_is_in_networkx_range() {
    let g = karate();
    for seed in 0..10 {
        assert_in_networkx_range(
            &format!("louvain seed {seed}"),
            &louvain_seeded(&g, 1.0, seed),
        );
        assert_in_networkx_range(
            &format!("louvain_connected seed {seed}"),
            &louvain_connected_seeded(&g, 1.0, seed),
        );
    }
}

#[test]
fn weighted_louvain_with_unit_weights_matches_networkx_range() {
    let mut w = vec![vec![0.0; N]; N];
    for &(u, v) in &KARATE_EDGES {
        w[u][v] = 1.0;
        w[v][u] = 1.0;
    }
    for seed in 0..10 {
        assert_in_networkx_range(
            &format!("louvain_weighted seed {seed}"),
            &louvain_weighted_seeded(&AdjacencyMatrix(&w), 1.0, seed),
        );
    }
}
