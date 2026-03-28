# graphops

[![crates.io](https://img.shields.io/crates/v/graphops.svg)](https://crates.io/crates/graphops)
[![Documentation](https://docs.rs/graphops/badge.svg)](https://docs.rs/graphops)
[![CI](https://github.com/arclabs561/graphops/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/graphops/actions/workflows/ci.yml)

Graph algorithms and node embeddings.

```toml
[dependencies]
graphops = "0.1.0"
```

## PageRank

```rust
use graphops::{pagerank, PageRankConfig};
use graphops::AdjacencyMatrix;

// Adjacency matrix: edge weights (0.0 = no edge)
let adj = vec![
    vec![0.0, 1.0, 1.0],
    vec![0.0, 0.0, 1.0],
    vec![1.0, 0.0, 0.0],
];

let scores = pagerank(&AdjacencyMatrix(&adj), PageRankConfig::default());
assert_eq!(scores.len(), 3);
```

Weighted PageRank and convergence diagnostics are available via `pagerank_weighted` and `pagerank_run`.

## Personalized PageRank (PPR)

Seed-biased ranking from a set of source nodes:

```rust
use graphops::{personalized_pagerank, PageRankConfig};
use graphops::AdjacencyMatrix;

let adj = vec![
    vec![0.0, 1.0, 0.0],
    vec![0.0, 0.0, 1.0],
    vec![1.0, 0.0, 0.0],
];

// Personalization vector: bias toward node 0
let pv = vec![1.0, 0.0, 0.0];
let scores = personalized_pagerank(&AdjacencyMatrix(&adj), PageRankConfig::default(), &pv);
```

## Random walks

Uniform and biased (node2vec-style) random walks, with optional parallelism:

```rust
use graphops::random_walk::{generate_walks, WalkConfig};
use graphops::AdjacencyMatrix;

let adj = vec![
    vec![0.0, 1.0, 1.0],
    vec![1.0, 0.0, 1.0],
    vec![1.0, 1.0, 0.0],
];

let config = WalkConfig {
    length: 10,
    walks_per_node: 5,
    seed: 42,
    ..WalkConfig::default()
};

let walks = generate_walks(&AdjacencyMatrix(&adj), config);
// walks: Vec<Vec<usize>> -- each walk is a sequence of node indices
```

For node2vec-style biased walks (with return parameter p and in-out parameter q), use `generate_biased_walks`. Parallel variants (`_parallel` suffix) are available with the `parallel` feature.

## Reachability

Count how many nodes each node can reach (forward) and be reached from (backward):

```rust
use graphops::reachability::reachability_counts_edges;

let edges = vec![(0, 1), (1, 2), (0, 2)];
let (forward, backward) = reachability_counts_edges(3, &edges);
// forward[0] = 2 (node 0 reaches nodes 1 and 2)
```

## Partitioning

Connected components and label propagation community detection:

```rust
use graphops::partition::{connected_components, label_propagation};
use graphops::AdjacencyMatrix;

let adj = vec![
    vec![0.0, 1.0, 0.0],
    vec![1.0, 0.0, 0.0],
    vec![0.0, 0.0, 0.0], // isolated node
];

let components = connected_components(&AdjacencyMatrix(&adj));
// components: [0, 0, 1] -- two components

let communities = label_propagation(&AdjacencyMatrix(&adj), 100, 42);
```

## Betweenness centrality

Requires the `petgraph` feature:

```rust
use graphops::betweenness::betweenness_centrality;
use petgraph::prelude::*;

let mut g: DiGraph<(), ()> = DiGraph::new();
let a = g.add_node(());
let b = g.add_node(());
let c = g.add_node(());
g.add_edge(a, b, ());
g.add_edge(b, c, ());

let scores = betweenness_centrality(&g);
// scores[1] is highest (node b is on the only a->c path)
```

## Examples

[**pagerank.rs**](examples/pagerank.rs) -- PageRank on a 4-node directed graph with labeled output. Demonstrates the adapter pattern: define an adjacency matrix, pass it to `pagerank`, and inspect ranked scores. Shows how link structure determines authority (node C, the most linked-to, ranks highest).

```bash
cargo run --example pagerank
```

## Feature flags

| Feature | What it adds |
|---------|-------------|
| `petgraph` | petgraph adapters + betweenness centrality |
| `parallel` | Parallel walk generation (via rayon) |
| `serde` | Serialize/deserialize for graph adapters |

## License

MIT OR Apache-2.0
