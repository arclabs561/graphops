# graphops examples

## Where to start

| I want to... | Run |
|---|---|
| Rank nodes in a directed graph | `pagerank` |
| Split a graph into communities | `community_detection` |
| Generate walks for embedding pipelines | `random_walks` |
| Score plausible missing edges with regions | `ellipsoidal_link_prediction` |
| Expand a graph lazily from a query node | `lazy_frontier` |

```sh
cargo run --example pagerank
cargo run --example community_detection
cargo run --example random_walks
cargo run --example ellipsoidal_link_prediction
cargo run --example lazy_frontier
```

## What to inspect

- `pagerank` prints a normalized score vector and the highest-ranked node.
- `community_detection` compares Louvain with a variant that splits disconnected communities. It does not implement Leiden's refinement phase.
- `random_walks` prints uniform walks and node2vec-style walks with `q < 1`, which biases walks outward from the previous node.
- `ellipsoidal_link_prediction` embeds nodes as ellipsoids, ranks non-edges by overlap, and compares intra-cluster vs. inter-cluster distance.
- `lazy_frontier` expands the most promising frontier edge first and stops once the evidence threshold is met, then compares how many nodes it visited with an eager expansion to the same depth.

## Real-data eval

| Data | Example | What to read from the output |
|---|---|---|
| Cora citation network (2708 nodes) | `pagerank_validation` | node count, score sum, positivity, and the top-3 scores |

Cora is not checked in. Without it the example prints a skip message and exits 0.

```sh
./scripts/fetch_cora_cites.sh
cargo run --release --example pagerank_validation
```

## Correctness checks

Checks against a known answer live in `tests/`, not here, so `cargo test` runs
them. For example, `tests/pagerank_uniform.rs` checks that PageRank is exactly
uniform on a directed cycle and a complete graph.
