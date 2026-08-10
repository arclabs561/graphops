# graphops

[![crates.io](https://img.shields.io/crates/v/graphops.svg)](https://crates.io/crates/graphops)
[![Documentation](https://docs.rs/graphops/badge.svg)](https://docs.rs/graphops)

Graph algorithms and node embeddings.

`graphops` operates on adjacency matrices or
[petgraph](https://crates.io/crates/petgraph) graphs.

```toml
[dependencies]
graphops = "0.5.0"
```

## Capabilities

| Family | Algorithms | Reference |
|---|---|---|
| Ranking and centrality | PageRank, personalized PageRank, HITS, Katz, eigenvector, closeness, harmonic, and betweenness | [`centrality`](https://docs.rs/graphops/latest/graphops/centrality/), [`pagerank`](examples/pagerank.rs) |
| Communities and topology | Components, SCCs, topological sort, k-core, label propagation, Louvain, and Leiden | [`partition`](https://docs.rs/graphops/latest/graphops/partition/), [`community_detection`](examples/community_detection.rs) |
| Traversal and paths | Reachability, BFS, and Dijkstra | [`reachability`](https://docs.rs/graphops/latest/graphops/reachability/), [`shortest_path`](https://docs.rs/graphops/latest/graphops/shortest_path/) |
| Walks and embeddings | Uniform and node2vec walks; ellipsoidal embeddings | [`random_walk`](https://docs.rs/graphops/latest/graphops/random_walk/), [`random_walks`](examples/random_walks.rs), [`ellipsoidal_link_prediction`](examples/ellipsoidal_link_prediction.rs) |
| Structural features | Triangles, clustering coefficients, walk counts, similarity, and top-k helpers | [`triangle`](https://docs.rs/graphops/latest/graphops/triangle/), [`hom_counts`](https://docs.rs/graphops/latest/graphops/hom_counts/) |
| Graph kernels | Weisfeiler-Lehman subtree, random-walk, and sliced Wasserstein kernels | [`graph_kernel`](https://docs.rs/graphops/latest/graphops/graph_kernel/) |

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

Personalized PageRank (seed-biased ranking) is available via `personalized_pagerank`.

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

Node2vec-style bias uses the return parameter `p` and in-out parameter `q`.
Parallel walk generation is available with the `parallel` feature.

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

See [`examples/README.md`](examples/README.md) for what to inspect in each example.

| Example | What it covers |
|---------|---------------|
| `pagerank` | PageRank on a small directed graph |
| `community_detection` | Louvain and Leiden community detection on a two-cluster graph |
| `random_walks` | Uniform and node2vec-style biased random walks |
| `ellipsoidal_link_prediction` | Ellipsoidal embeddings for link prediction |

```bash
cargo run --example pagerank
cargo run --example community_detection
cargo run --example random_walks
cargo run --example ellipsoidal_link_prediction
```

## Feature flags

Optional features: `petgraph` (petgraph adapters + betweenness centrality), `parallel` (rayon walk generation), `serde`.

## Limitations

- Inputs must expose dense node identifiers in `0..node_count()` through the
  graph traits; outputs use the same order. Directed and weighted semantics are
  documented per algorithm.
- Algorithms with randomized visit or sampling order accept a seed. Reuse the
  seed when comparing runs.
- This crate provides in-memory algorithm primitives. It does not provide graph
  storage, queries, transactions, or a database service.

## License

MIT OR Apache-2.0
