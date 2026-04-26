# Changelog

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
