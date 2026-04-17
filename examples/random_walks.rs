/// Generate random walks for node embedding pipelines.
///
/// Shows uniform walks and node2vec-style biased walks (with return
/// parameter p and in-out parameter q) on a small social graph.
///
/// ```sh
/// cargo run --example random_walks
/// ```
///
/// Expected output (walks are stochastic, exact nodes will vary):
///
/// ```text
/// === Uniform random walks ===
/// Walk 0 from node 0: [0, 3, 2, 1, 0, 2, ...]
/// Walk 1 from node 0: [0, 1, 3, 0, 2, 1, ...]
/// ...
///
/// === Biased walks (node2vec: p=1.0, q=0.5) ===
/// Walk 0 from node 0: [0, 2, 1, 0, 3, 2, ...]
/// ...
/// ```
use graphops::graph::AdjacencyMatrix;
use graphops::random_walk::{generate_biased_walks, generate_walks, WalkConfig};

fn main() {
    // Small social graph: 6 nodes, undirected.
    #[rustfmt::skip]
    let adj = vec![
        vec![0., 1., 1., 1., 0., 0.], // 0: connected to 1, 2, 3
        vec![1., 0., 1., 1., 1., 0.], // 1: connected to 0, 2, 3, 4
        vec![1., 1., 0., 0., 0., 1.], // 2: connected to 0, 1, 5
        vec![1., 1., 0., 0., 1., 0.], // 3: connected to 0, 1, 4
        vec![0., 1., 0., 1., 0., 1.], // 4: connected to 1, 3, 5
        vec![0., 0., 1., 0., 1., 0.], // 5: connected to 2, 4
    ];
    let g = AdjacencyMatrix(&adj);

    // Uniform random walks.
    let config = WalkConfig {
        length: 10,
        walks_per_node: 3,
        seed: 42,
        ..WalkConfig::default()
    };
    let walks = generate_walks(&g, config);

    println!("=== Uniform random walks ===");
    println!("{} walks, length {}\n", walks.len(), config.length);
    for (i, walk) in walks.iter().take(6).enumerate() {
        println!("  Walk {} from node {}: {:?}", i, walk[0], walk);
    }

    // Node2vec-style biased walks.
    // p < 1: prefer returning to the previous node (local exploration).
    // q < 1: prefer moving away from the previous node (BFS-like).
    let biased_config = WalkConfig {
        length: 10,
        walks_per_node: 3,
        p: 1.0,
        q: 0.5, // bias toward exploring new neighborhoods
        seed: 42,
        ..WalkConfig::default()
    };
    let biased_walks = generate_biased_walks(&g, biased_config);

    println!("\n=== Biased walks (node2vec: p=1.0, q=0.5) ===");
    println!("{} walks, length {}\n", biased_walks.len(), biased_config.length);
    for (i, walk) in biased_walks.iter().take(6).enumerate() {
        println!("  Walk {} from node {}: {:?}", i, walk[0], walk);
    }
}
