#![warn(missing_docs)]
//! `graphops`: graph operators (walk/diffusion + related centralities).
//!
//! This crate is the canonical home for operator implementations in this workspace.
//! Call sites should prefer `graphops::*` (and not refer to historical names).
//!
//! Public invariants (must not drift):
//! - **Node order**: outputs are indexed by node id \(0..n-1\) consistent with the input graph’s
//!   adapter semantics (e.g. `petgraph::NodeIndex::index()` when using the `petgraph` feature).
//! - **Determinism**: deterministic operators are deterministic given identical inputs + configs.
//! - **No silent normalization**: normalization behavior is explicit in the API/docs (e.g.
//!   personalization vectors in PPR).
//!
//! Swappable (allowed to change without breaking the contract):
//! - iteration strategy (serial vs parallel)
//! - convergence details (so long as tolerance semantics remain correct)
//! - internal data structures (so long as invariants hold)

#[cfg(feature = "petgraph")]
pub mod betweenness;
pub mod centrality;
pub mod eigenvector;
pub mod ellipsoidal;
pub mod graph;
/// Graph kernels: WL subtree, random walk, and sliced Wasserstein.
pub mod graph_kernel;
pub mod katz;
pub mod leiden;
pub mod louvain;
pub mod node2vec;
pub mod pagerank;
pub mod partition;
pub mod ppr;
pub mod random_walk;
pub mod reachability;
pub mod shortest_path;
pub mod similarity;
pub mod topk;
pub mod triangle;

#[cfg(feature = "petgraph")]
pub use betweenness::betweenness_centrality;
#[cfg(feature = "petgraph")]
pub use graph::PetgraphRef;
pub use graph::{AdjacencyMatrix, Graph, GraphRef, WeightedGraph, WeightedGraphRef};
pub use node2vec::{
    generate_biased_walks_precomp_ref, generate_biased_walks_precomp_ref_from_nodes,
    generate_biased_walks_weighted_plus_ref, generate_biased_walks_weighted_ref,
    PrecomputedBiasedWalks, WeightedNode2VecPlusConfig,
};

#[cfg(feature = "parallel")]
pub use node2vec::generate_biased_walks_precomp_ref_parallel_from_nodes;
#[cfg(feature = "parallel")]
pub use random_walk::{
    generate_biased_walks_ref_parallel, generate_biased_walks_ref_parallel_from_nodes,
    generate_walks_ref_parallel, generate_walks_ref_parallel_from_nodes,
};

pub use centrality::{closeness_centrality, harmonic_centrality, hits};
pub use eigenvector::{eigenvector_centrality, eigenvector_centrality_run, EigenvectorRun};
pub use ellipsoidal::{
    ellipsoid_distance, ellipsoid_overlap, ellipsoidal_embedding, Ellipsoid, EllipsoidalConfig,
};
pub use graph_kernel::{
    random_walk_kernel, sliced_wasserstein_graph_kernel, structural_node_features,
    wl_subtree_kernel,
};
pub use katz::{
    katz_centrality, katz_centrality_checked, katz_centrality_run, KatzConfig, KatzRun,
};
pub use leiden::{leiden, leiden_seeded, leiden_weighted, leiden_weighted_seeded};
pub use louvain::{louvain, louvain_seeded, louvain_weighted, louvain_weighted_seeded};
pub use pagerank::{pagerank, pagerank_weighted, PageRankConfig};
pub use pagerank::{pagerank_checked, pagerank_weighted_checked};
pub use pagerank::{
    pagerank_checked_run, pagerank_run, pagerank_weighted_checked_run, pagerank_weighted_run,
    PageRankRun,
};
pub use partition::{
    connected_components, core_numbers, k_core, label_propagation, strongly_connected_components,
    topological_sort,
};
pub use ppr::{personalized_pagerank, personalized_pagerank_checked};
pub use ppr::{personalized_pagerank_checked_run, personalized_pagerank_run};
pub use random_walk::{
    generate_biased_walks, generate_biased_walks_from_nodes, generate_biased_walks_ref,
    generate_biased_walks_ref_from_nodes, generate_biased_walks_ref_streaming_from_nodes,
    generate_walks, generate_walks_from_nodes, generate_walks_ref, generate_walks_ref_from_nodes,
    generate_walks_ref_streaming_from_nodes, sample_start_nodes_reservoir, WalkConfig,
};
pub use reachability::reachability_counts_edges;
pub use shortest_path::{bfs_distances, bfs_path, dijkstra_distances};
pub use similarity::{cosine, jaccard, overlap, top_k_similar_jaccard};
pub use topk::{normalize, top_k};
pub use triangle::{clustering_coefficients, global_clustering_coefficient, triangle_count};

/// Errors returned by the `*_checked` variants of graph operators.
///
/// The non-checked variants (`pagerank`, `personalized_pagerank`, etc.) panic
/// or clamp to defensible defaults on invalid input. The `*_checked` variants
/// return this error type so callers can distinguish input-validation
/// failures from operator-internal issues. Marked `#[non_exhaustive]` so new
/// variants can be added without a semver break.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A node id was outside `0..node_count()` — typically from a user-supplied
    /// personalization vector or seed set.
    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),
    /// A configuration value violated the operator's preconditions. The string
    /// names the offending parameter (e.g. `"damping must be in [0, 1]"`).
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// Two slices that should agree in length did not. The pair is
    /// `(actual, expected)`.
    #[error("dimension mismatch: {0} vs {1}")]
    DimensionMismatch(usize, usize),
}

/// Shorthand for `std::result::Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;
