# graphops

Graph algorithms and node embeddings.

## Install

```toml
[dependencies]
graphops = "0.5.1"
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

PageRank, weighted PageRank and Brandes betweenness are checked against
networkx outputs in `tests/rosetta_graphops*.rs`.

## Personalized PageRank

```rust
use graphops::{personalized_pagerank, AdjacencyMatrix, PageRankConfig};

let adj = vec![
    vec![0.0, 1.0, 0.0],
    vec![0.0, 0.0, 1.0],
    vec![1.0, 0.0, 0.0],
];

// Restart at node 0.
let ppr = personalized_pagerank(&AdjacencyMatrix(&adj), PageRankConfig::default(), &[1.0, 0.0, 0.0]);
assert!(ppr[0] > ppr[1] && ppr[1] > ppr[2]);
```

## node2vec walks

```rust
use graphops::{generate_biased_walks, AdjacencyMatrix, WalkConfig};

let adj = vec![
    vec![0.0, 1.0, 1.0],
    vec![1.0, 0.0, 1.0],
    vec![1.0, 1.0, 0.0],
];
let config = WalkConfig { length: 5, walks_per_node: 2, p: 1.0, q: 0.5, seed: 7 };

let walks = generate_biased_walks(&AdjacencyMatrix(&adj), config);
assert_eq!(walks.len(), 3 * 2);
assert!(walks.iter().all(|w| w.len() == 5));
// Same seed, same walks.
assert_eq!(walks, generate_biased_walks(&AdjacencyMatrix(&adj), config));
```

## Community detection

```rust
use graphops::{louvain_seeded, GraphRef};

struct Adj(Vec<Vec<usize>>);

impl GraphRef for Adj {
    fn node_count(&self) -> usize {
        self.0.len()
    }
    fn neighbors_ref(&self, node: usize) -> &[usize] {
        &self.0[node]
    }
}

// Two triangles joined by the edge 2-3.
let g = Adj(vec![vec![1, 2], vec![0, 2], vec![0, 1, 3], vec![2, 4, 5], vec![3, 5], vec![3, 4]]);
let labels = louvain_seeded(&g, 1.0, 42);
assert_eq!(labels[0], labels[1]);
assert_eq!(labels[0], labels[2]);
assert_eq!(labels[3], labels[5]);
assert_ne!(labels[0], labels[3]);
```

`louvain_connected` additionally splits disconnected communities, so every
community it returns is connected (`leiden` is an older name for it; it does not
implement Leiden's refinement phase).

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
