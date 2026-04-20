//! Minimal graph adapter traits.

/// Owned-`Vec` graph adapter: the simplest adapter, allocates per neighbor query.
pub trait Graph {
    /// Number of nodes in the graph.
    fn node_count(&self) -> usize;
    /// Out-neighbors of `node` as an owned `Vec` of node ids.
    fn neighbors(&self, node: usize) -> Vec<usize>;
    /// Number of out-neighbors of `node`.
    fn out_degree(&self, node: usize) -> usize {
        self.neighbors(node).len()
    }
}

/// A graph view that can return **borrowed** neighbor slices.
///
/// This is the “cache-friendly” adapter: it avoids allocating a new `Vec`
/// on every step of a random walk.
pub trait GraphRef {
    /// Number of nodes in the graph.
    fn node_count(&self) -> usize;
    /// Out-neighbors of `node` as a borrowed slice.
    fn neighbors_ref(&self, node: usize) -> &[usize];
    /// Number of out-neighbors of `node`.
    fn out_degree(&self, node: usize) -> usize {
        self.neighbors_ref(node).len()
    }
}

/// Graph with scalar edge weights.
pub trait WeightedGraph: Graph {
    /// Weight of the edge `(source, target)`; returns `0.0` when the edge is absent.
    fn edge_weight(&self, source: usize, target: usize) -> f64;
}

/// A weighted graph view that can return **borrowed** neighbor + weight slices.
///
/// This matches the “CSR-style” representation PecanPy relies on:
/// a node has a contiguous neighbor list and a contiguous weight list,
/// with matching indices.
pub trait WeightedGraphRef {
    /// Number of nodes in the graph.
    fn node_count(&self) -> usize;

    /// Return `(neighbors, weights)` for a node.
    ///
    /// Requirements:
    /// - `neighbors.len() == weights.len()`
    /// - Weights should be non-negative for node2vec/node2vec+ semantics.
    fn neighbors_and_weights_ref(&self, node: usize) -> (&[usize], &[f32]);

    /// Number of out-neighbors of `node`.
    fn out_degree(&self, node: usize) -> usize {
        self.neighbors_and_weights_ref(node).0.len()
    }
}

/// Row-major adjacency-matrix adapter. Entry `(i, j)` is the weight of edge `i -> j`;
/// zero means no edge.
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
