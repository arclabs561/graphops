# graphops

Graph algorithms and node embeddings.

## Install

```toml
[dependencies]
graphops = "0.5.0"
```

## PageRank

```rust
use graphops::{pagerank, AdjacencyMatrix, PageRankConfig};

// adj[i][j] > 0 means an edge from i to j.
let adj = vec![
    vec![0.0, 1.0, 1.0],
    vec![0.0, 0.0, 1.0],
    vec![1.0, 0.0, 0.0],
];

let scores = pagerank(&AdjacencyMatrix(&adj), PageRankConfig::default());
assert_eq!(scores.len(), 3);
assert!((scores.iter().sum::<f64>() - 1.0).abs() < 1e-9);
```

See the [API documentation](https://docs.rs/graphops) for the available
algorithms and [`examples/`](examples/README.md) for runnable programs.

## Features

| Feature | Enables |
|---|---|
| `petgraph` | `petgraph` adapters and betweenness centrality |
| `parallel` | Parallel random-walk generation with Rayon |
| `serde` | Serialization support |
| `simd` | SIMD-accelerated PageRank convergence reductions through `innr` |

## Limitation

`graphops` provides in-memory algorithms. It does not provide graph storage,
queries, transactions, or a database service.

## License

MIT OR Apache-2.0
