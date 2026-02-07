# graphops

Graph operators and centralities as a small Rust crate.

## Usage

```toml
[dependencies]
graphops = { version = "0.1.0", features = ["petgraph"] }
petgraph = "0.6"
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

## Implemented Operators

- PageRank and Personalized PageRank
- Random walks and biased walks (node2vec-style)
- Reachability counts
- Connected components / label propagation
- Top-k helpers
- (feature-gated) Betweenness centrality via `petgraph`

## Features

- `serde`: enable serde on some graph adapters.
- `parallel`: enable parallel walk generation.
- `petgraph`: enable `petgraph` adapters + betweenness helper.

## Development

```bash
cargo test
```
