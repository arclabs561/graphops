# Changelog

## [0.4.1]

### Documented

- `newman_betweenness` now documents its scaling behavior up front: exact
  mode is `O(n^2 * max_iter * d)` and recommends sampling for `n >= 500`
  (`n_sources = sqrt(n)` rule of thumb) and is impractical above `n = 10000`.
  Previously a user could call the function with default config on a large
  graph and silently wait for hours.
- `katz_centrality` and `newman_betweenness` both note that no in-tree
  consumer currently uses them (sheaf's community-detection path uses
  Leiden/Louvain, not centrality). They are speculative additions, not
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
