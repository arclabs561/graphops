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
pub mod graph;
pub mod node2vec;
pub mod pagerank;
pub mod partition;
pub mod ppr;
pub mod random_walk;
pub mod reachability;
pub mod topk;

#[cfg(feature = "petgraph")]
pub use betweenness::betweenness_centrality;
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

pub use pagerank::{pagerank, pagerank_weighted, PageRankConfig};
pub use pagerank::{pagerank_checked, pagerank_weighted_checked};
pub use pagerank::{
    pagerank_checked_run, pagerank_run, pagerank_weighted_checked_run, pagerank_weighted_run,
    PageRankRun,
};
pub use partition::{connected_components, label_propagation};
pub use ppr::{personalized_pagerank, personalized_pagerank_checked};
pub use ppr::{personalized_pagerank_checked_run, personalized_pagerank_run};
pub use random_walk::{
    generate_biased_walks, generate_biased_walks_from_nodes, generate_biased_walks_ref,
    generate_biased_walks_ref_from_nodes, generate_biased_walks_ref_streaming_from_nodes,
    generate_walks, generate_walks_from_nodes, generate_walks_ref, generate_walks_ref_from_nodes,
    generate_walks_ref_streaming_from_nodes, sample_start_nodes_reservoir, WalkConfig,
};
pub use reachability::reachability_counts_edges;
pub use topk::{normalize, top_k};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, Error>;
