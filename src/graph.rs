//! Minimal graph adapter traits.
//!
//! Every operator in this crate is generic over one of these traits, so callers
//! can plug in any representation that exposes node count and a neighbor query.
//! Four traits cover the product of two axes: owned vs borrowed neighbor lists,
//! unweighted vs weighted edges.
//!
//! | | Unweighted | Weighted |
//! |---|---|---|
//! | Owned `Vec` | [`Graph`] | [`WeightedGraph`] |
//! | Borrowed slice | [`GraphRef`] | [`WeightedGraphRef`] |
//!
//! Pick the owned variant for ease of implementation; pick the borrowed
//! variant when the operator runs in a tight loop (random walks, message
//! passing) and neighbor-list allocation would dominate.

/// Owned-neighbor graph adapter. The simplest adapter to implement — each
/// `neighbors(node)` call returns a fresh `Vec`.
///
/// Node ids are dense, `0..node_count()`. An edge `u -> v` is expressed by
/// `v` appearing in `neighbors(u)`. Duplicates are allowed but operators
/// assume unique neighbors; deduplicate upstream if needed.
///
/// Prefer [`GraphRef`] when the operator reads neighbors repeatedly in a hot
/// loop — allocation cost of owned `Vec` compounds.
pub trait Graph {
    /// Number of nodes in the graph. Must be stable across calls.
    fn node_count(&self) -> usize;
    /// Out-neighbors of `node` as an owned `Vec` of node ids in `0..node_count()`.
    ///
    /// Behavior is unspecified if `node >= node_count()`; operators either
    /// panic or return empty. Use `*_checked` wrappers for explicit validation.
    fn neighbors(&self, node: usize) -> Vec<usize>;
    /// Number of out-neighbors of `node`. Default calls `neighbors(node).len()`;
    /// override if out-degree is cheaper to compute without materializing the list.
    fn out_degree(&self, node: usize) -> usize {
        self.neighbors(node).len()
    }
}

/// Graph view that returns **borrowed** neighbor slices, avoiding per-query
/// allocation.
///
/// This is the cache-friendly adapter: it fits a CSR-style (compressed sparse
/// row) layout where each node owns a contiguous slice of adjacent node ids.
/// Use for random walks, PageRank iteration, and any operator whose inner
/// loop dereferences neighbors repeatedly.
pub trait GraphRef {
    /// Number of nodes in the graph. Must be stable across calls.
    fn node_count(&self) -> usize;
    /// Out-neighbors of `node` as a borrowed slice of node ids.
    /// Lifetime is tied to `&self`; the slice stays valid until the next
    /// mutating call on the underlying storage.
    fn neighbors_ref(&self, node: usize) -> &[usize];
    /// Number of out-neighbors of `node`. Defaults to the borrowed slice's `len()`.
    fn out_degree(&self, node: usize) -> usize {
        self.neighbors_ref(node).len()
    }
}

/// Graph with scalar `f64` edge weights.
///
/// Operators like `pagerank_weighted` interpret weights as transition-
/// probability proportions (higher weight = more likely transition).
/// Negative weights are invalid input for weighted PageRank; use
/// `pagerank_weighted_checked` to reject them explicitly.
pub trait WeightedGraph: Graph {
    /// Weight of the edge `(source, target)`. Return `0.0` when the edge is
    /// absent — callers treat zero as "no edge", not as a zero-probability
    /// edge, to keep random-walk transition matrices well-defined.
    fn edge_weight(&self, source: usize, target: usize) -> f64;
}

/// Weighted graph view returning **borrowed** `(neighbors, weights)` slice pairs.
///
/// Mirrors the CSR-style representation used by node2vec / PecanPy: each node
/// owns contiguous neighbor and weight lists of equal length, indexed in
/// parallel. Prefer this over [`WeightedGraph`] for operators that iterate
/// over the same node's neighbors many times (biased walks, alias sampling).
pub trait WeightedGraphRef {
    /// Number of nodes in the graph. Must be stable across calls.
    fn node_count(&self) -> usize;

    /// Borrow `(neighbors, weights)` for a node, where the two slices are
    /// indexed in parallel: `weights[i]` is the weight of the edge to
    /// `neighbors[i]`.
    ///
    /// Invariants:
    /// - `neighbors.len() == weights.len()`
    /// - Weights should be non-negative (required for node2vec / node2vec+).
    ///   Negative weights produce undefined sampling behavior.
    fn neighbors_and_weights_ref(&self, node: usize) -> (&[usize], &[f32]);

    /// Number of out-neighbors of `node`. Default delegates to the neighbor slice.
    fn out_degree(&self, node: usize) -> usize {
        self.neighbors_and_weights_ref(node).0.len()
    }
}

/// Row-major adjacency-matrix adapter.
///
/// Wraps an existing `&[Vec<f64>]` where `matrix[i][j]` is the weight of edge
/// `i -> j` and `0.0` means no edge. Suitable for small or dense graphs where
/// a matrix is the natural representation; for sparse graphs prefer an
/// adjacency-list adapter like [`PetgraphRef`].
///
/// # Example
///
/// ```
/// use graphops::{AdjacencyMatrix, Graph};
///
/// // 3-node directed cycle: 0 -> 1 -> 2 -> 0
/// let m = vec![
///     vec![0.0, 1.0, 0.0],
///     vec![0.0, 0.0, 1.0],
///     vec![1.0, 0.0, 0.0],
/// ];
/// let g = AdjacencyMatrix(&m);
/// assert_eq!(g.node_count(), 3);
/// assert_eq!(g.neighbors(0), vec![1]);
/// ```
pub struct AdjacencyMatrix<'a>(pub &'a [Vec<f64>]);

impl<'a> Graph for AdjacencyMatrix<'a> {
    fn node_count(&self) -> usize {
        self.0.len()
    }
    fn neighbors(&self, node: usize) -> Vec<usize> {
        self.0[node]
            .iter()
            .enumerate()
            .filter(|(_, &w)| w > 0.0)
            .map(|(i, _)| i)
            .collect()
    }
}

impl<'a> WeightedGraph for AdjacencyMatrix<'a> {
    fn edge_weight(&self, source: usize, target: usize) -> f64 {
        self.0[source][target]
    }
}

#[cfg(feature = "petgraph")]
impl<N, E, Ty, Ix> Graph for petgraph::Graph<N, E, Ty, Ix>
where
    Ty: petgraph::EdgeType,
    Ix: petgraph::graph::IndexType,
{
    fn node_count(&self) -> usize {
        self.node_count()
    }
    fn neighbors(&self, node: usize) -> Vec<usize> {
        self.neighbors(petgraph::graph::NodeIndex::new(node))
            .map(|idx| idx.index())
            .collect()
    }
}

/// Pre-built adjacency list adapter for petgraph graphs.
/// Converts a petgraph graph into a structure implementing `GraphRef`
/// by pre-computing neighbor lists.
#[cfg(feature = "petgraph")]
pub struct PetgraphRef {
    neighbors: Vec<Vec<usize>>,
}

#[cfg(feature = "petgraph")]
impl PetgraphRef {
    /// Build from any petgraph graph type.
    pub fn from_graph<N, E, Ty, Ix>(graph: &petgraph::Graph<N, E, Ty, Ix>) -> Self
    where
        Ty: petgraph::EdgeType,
        Ix: petgraph::graph::IndexType,
    {
        let n = graph.node_count();
        let neighbors: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                graph
                    .neighbors(petgraph::graph::NodeIndex::new(i))
                    .map(|idx| idx.index())
                    .collect()
            })
            .collect();
        Self { neighbors }
    }
}

#[cfg(feature = "petgraph")]
impl GraphRef for PetgraphRef {
    fn node_count(&self) -> usize {
        self.neighbors.len()
    }
    fn neighbors_ref(&self, node: usize) -> &[usize] {
        &self.neighbors[node]
    }
}

#[cfg(feature = "petgraph")]
impl<N, Ty, Ix> WeightedGraph for petgraph::Graph<N, f64, Ty, Ix>
where
    Ty: petgraph::EdgeType,
    Ix: petgraph::graph::IndexType,
{
    fn edge_weight(&self, source: usize, target: usize) -> f64 {
        let s = petgraph::graph::NodeIndex::new(source);
        let t = petgraph::graph::NodeIndex::new(target);
        self.find_edge(s, t)
            .map(|e| *self.edge_weight(e).unwrap_or(&0.0))
            .unwrap_or(0.0)
    }
}
