# graphops

Graph operators and centralities as a small Rust crate.

Implemented operators include:

- PageRank and Personalized PageRank
- random walks and biased walks (node2vec-style)
- reachability counts
- connected components / label propagation
- top-k helpers
- (feature-gated) betweenness centrality via `petgraph`

## Usage

```toml
[dependencies]
graphops = "0.1.0"
```

Example:

```rust
use graphops::{pagerank, PageRankConfig};
use petgraph::prelude::*;

let mut g: DiGraph<(), f64> = DiGraph::new();
let a = g.add_node(());
let b = g.add_node(());
g.add_edge(a, b, 1.0);

let scores = pagerank(&g, PageRankConfig::default());
assert_eq!(scores.len(), g.node_count());
```

## Features

- `serde`: enable serde on some graph adapters (when available).
- `parallel`: enable parallel walk generation.
- `petgraph`: enable `petgraph` adapters + betweenness helper.

## Development

```bash
cargo test
```
