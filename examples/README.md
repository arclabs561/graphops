# graphops examples

## Where to start

| I want to... | Run |
|---|---|
| Rank nodes in a directed graph | `pagerank` |
| Split a graph into communities | `community_detection` |
| Generate walks for embedding pipelines | `random_walks` |
| Score plausible missing edges with regions | `ellipsoidal_link_prediction` |
| Expand graph neighborhoods only as needed | `lazy_frontier` |

```sh
cargo run --example pagerank
cargo run --example community_detection
cargo run --example random_walks
cargo run --example ellipsoidal_link_prediction
cargo run --example lazy_frontier
```

## What to inspect

- `pagerank` prints a normalized score vector and the highest-ranked node. Node C ranks highest because it receives edges from three nodes.
- `community_detection` compares Louvain and Leiden on two dense clusters joined by a bridge.
- `random_walks` prints uniform walks and node2vec-style walks with `q < 1`, which biases walks outward from the previous node.
- `ellipsoidal_link_prediction` embeds nodes as ellipsoids, ranks non-edges by overlap, and compares intra-cluster vs. inter-cluster distance.
- `lazy_frontier` compares eager depth expansion with a priority frontier that stops after enough evidence is found.
