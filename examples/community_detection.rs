/// Compare Louvain with connectivity-refined Louvain.
///
/// Builds a small graph with two dense clusters connected by a bridge
/// and prints labels from both algorithms.
///
/// ```sh
/// cargo run --example community_detection
/// ```
use graphops::graph::GraphRef;
use graphops::{louvain_connected_seeded, louvain_seeded};

/// Adjacency list graph (implements GraphRef for borrowed neighbor access).
struct VecGraph {
    adj: Vec<Vec<usize>>,
}

impl GraphRef for VecGraph {
    fn node_count(&self) -> usize {
        self.adj.len()
    }
    fn neighbors_ref(&self, node: usize) -> &[usize] {
        &self.adj[node]
    }
}

fn main() {
    // Two dense clusters (0-4 and 5-8) with one bridge edge (3 <-> 5).
    let g = VecGraph {
        adj: vec![
            vec![1, 2, 3, 4],    // 0: cluster A
            vec![0, 2, 3],       // 1: cluster A
            vec![0, 1, 3, 4],    // 2: cluster A
            vec![0, 1, 2, 4, 5], // 3: bridge
            vec![0, 2, 3],       // 4: cluster A
            vec![3, 6, 7, 8],    // 5: cluster B
            vec![5, 7, 8],       // 6: cluster B
            vec![5, 6, 8],       // 7: cluster B
            vec![5, 6, 7],       // 8: cluster B
        ],
    };

    println!(
        "Graph: {} nodes, two clusters connected by node 3 <-> 5\n",
        g.node_count()
    );

    let louvain = louvain_seeded(&g, 1.0, 42);
    println!("Louvain communities: {:?}", louvain);
    print_communities(&louvain);

    let connected = louvain_connected_seeded(&g, 1.0, 42);
    println!("Connectivity-refined Louvain communities: {connected:?}");
    print_communities(&connected);
}

fn print_communities(labels: &[usize]) {
    let max_label = labels.iter().copied().max().unwrap_or(0);
    for c in 0..=max_label {
        let members: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == c)
            .map(|(i, _)| i)
            .collect();
        if !members.is_empty() {
            println!("  Community {}: nodes {:?}", c, members);
        }
    }
    println!();
}
