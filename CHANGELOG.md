# Changelog

## [Unreleased]

### Added

- `WeightedGraph::stored_weights(node)`, a defaulted method that yields every
  stored outgoing weight of a node. The default reads `edge_weight` for each
  entry of `neighbors`; `AdjacencyMatrix` overrides it to return the full row.

### Fixed

- `pagerank_weighted_checked` and `pagerank_weighted_checked_run` now reject
  negative and NaN weights in an `AdjacencyMatrix`. The validator walked
  `neighbors`, which keeps only `w > 0.0`, so those entries were never seen
  and the call succeeded. Other algorithms are unchanged.

- `clustering_coefficients` and `global_clustering_coefficient` ignore
  self-loops, as networkx does. A self-loop counted toward the degree, so a
  triangle with a self-loop on node 0 gave C(0) = 1/3 instead of 1.

## [0.5.3] - 2026-10-30

### Fixed

- `newman_betweenness` solves each source-sink pair by conjugate gradient
  instead of Jacobi iteration. Jacobi needed O(n^2) sweeps on a path, so the
  default 200 iterations stopped early on a 40-node path and returned
  unconverged scores. `max_iter` now caps CG iterations.
- `newman_betweenness` pins one node per connected component instead of only
  node n - 1, and pairs in different components contribute 0. Two disjoint
  identical triangles previously scored 0.333, 0.383 and 0.433 instead of
  equal values.

## [0.5.2] - 2026-10-26

### Fixed

- Louvain and connected Louvain no longer count a super-node's own self-loop
  as a link to its community. After the first aggregation no node could move,
  so results stopped at the level-1 partition; on Zachary's karate club,
  modularity rises from 0.335 (7 communities) to the networkx range
  (0.415 to 0.420, 4 communities).
- The weighted petgraph adapter sums the weights of parallel edges instead of
  counting the first one several times, and Louvain, connected Louvain and
  weighted PageRank count each neighbour pair once.
- Sorting uses `total_cmp`, so a NaN score or feature no longer panics in
  `sliced_wasserstein_graph_kernel`.

## [0.5.1] - 2026-08-13

### Added

- Borrowed-adjacency PageRank and personalized PageRank entry points for
  `GraphRef` implementations, including checked and diagnostic variants.

## [0.5.0] - 2026-07-03

### Added

- `hom_counts` module: rooted walk and closed-walk counts per node — the
  tractable homomorphism-count families (rooted paths and cycles), grounded
  in the Dell-Grohe-Rattan treewidth/WL correspondence, intended as
  expressiveness-raising GNN node features.

## [0.4.3]

### Changed

- Eigenvector L1 and current-flow-centrality L2 convergence residuals now
  route through `innr::dense_f64` under the `simd` feature, matching
  pagerank/katz/betweenness. With innr 0.6.1 these f64 reductions dispatch
  to AVX-512/AVX2/NEON; portable fallback unchanged when `simd` is off.

## [0.4.1]

### Documented

- `newman_betweenness` now documents its scaling behavior up front: exact
  mode is `O(n^2 * max_iter * d)` and recommends sampling for `n >= 500`
  (`n_sources = sqrt(n)` rule of thumb) and is impractical above `n = 10000`.
  Previously a user could call the function with default config on a large
  graph and silently wait for hours.
- `katz_centrality` and `newman_betweenness` both note that no in-tree
  consumer currently uses them (sheaf's path uses the community-detection
  routines, not centrality). They are speculative additions, not
  consumer-driven, and exist for callers with their own use cases.

## [0.4.0]

### Added

- `newman_betweenness` (and `_run` / `_checked` variants) — Newman (2005)
  random-walk betweenness centrality. Each source-sink pair contributes via a
  Jacobi-solved current-flow on the graph Laplacian; node scores aggregate
  edge currents incident to each node. `NewmanBetweennessConfig` defaults:
  `n_sources = usize::MAX` (exact, all sources), `seed = 42`,
  `max_iter = 200`, `tolerance = 1e-6`. Set `n_sources` smaller for an
  approximation that scales to larger graphs.
- `betweenness` module is now unconditionally compiled; the existing
  petgraph-based Brandes impl is internally feature-gated. Newman has no
  petgraph dependency.

## [0.3.0]

### Added

- `katz_centrality` (and `_run` / `_checked` variants) — Katz centrality via
  power iteration with the same `simd` L1-residual gate as PageRank.
  `KatzConfig` defaults: `alpha = 0.1`, `beta = 1.0`, `max_iterations = 100`,
  `tolerance = 1e-6`. The `_checked` form rejects `alpha >= 1`; tighter
  bounds (`alpha < 1/spectral_radius(A)`) may be needed for large graphs.

## [0.2.2]

### Changed

- PageRank L1 residual computation uses `innr::dense_f64::l1_distance_f64`
  under the `simd` feature, accelerating the per-iteration convergence check
  on AVX2/NEON targets. Behavior is identical to the scalar path; the gate is
  purely a perf optimization.

## [0.2.1]

### Changed

- Real prose for trait, config, and error doc comments (45 stubs replaced).
- Two unresolved cross-crate rustdoc links escaped to plain backticks.

## [0.2.0]

### Added

- `PetgraphRef` adapter for petgraph integration.
- `graph_kernel` module: Weisfeiler-Lehman, random-walk, shortest-path kernels.
- `louvain_seeded` for community detection with explicit seed.
- Doc and clippy cleanup pass.
