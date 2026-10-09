use graphops::reachability_counts_edges;

#[cfg(feature = "petgraph")]
fn assert_prob_like(xs: &[f64]) {
    assert!(!xs.is_empty());
    for &x in xs {
        assert!(x.is_finite(), "non-finite score: {x}");
        assert!(x >= 0.0, "negative score: {x}");
    }
    let s: f64 = xs.iter().copied().sum();
    assert!((s - 1.0).abs() <= 1e-6, "sum={s} not ~1");
}

#[cfg(feature = "petgraph")]
mod petgraph_invariants {
    use super::assert_prob_like;
    use graphops::{pagerank, personalized_pagerank, PageRankConfig};
    use petgraph::prelude::*;

    #[test]
    fn pagerank_is_finite_nonnegative_and_sums_to_one() {
        // 0 -> 1 -> 2
        let mut g: DiGraph<(), f64> = DiGraph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);

        let scores = pagerank(&g, PageRankConfig::default());
        assert_eq!(scores.len(), g.node_count());
        assert_prob_like(&scores);
    }

    #[test]
    fn personalized_pagerank_is_finite_nonnegative_and_sums_to_one() {
        // 0 -> 1 -> 2
        let mut g: DiGraph<(), f64> = DiGraph::new();
        g.add_node(());
        g.add_node(());
        g.add_node(());
        g.add_edge(NodeIndex::new(0), NodeIndex::new(1), 1.0);
        g.add_edge(NodeIndex::new(1), NodeIndex::new(2), 1.0);

        // Teleport strongly to node 0.
        let p = vec![1.0, 0.0, 0.0];
        let scores = personalized_pagerank(&g, PageRankConfig::default(), &p);
        assert_eq!(scores.len(), g.node_count());
        assert_prob_like(&scores);
    }

    #[test]
    fn betweenness_line_graph_middle_is_highest() {
        use graphops::betweenness_centrality;
        // 0 -> 1 -> 2 -> 3
        let mut g: DiGraph<(), ()> = DiGraph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        g.add_edge(a, b, ());
        g.add_edge(b, c, ());
        g.add_edge(c, d, ());

        let bc = betweenness_centrality(&g);
        assert_eq!(bc[a.index()], 0.0);
        assert_eq!(bc[d.index()], 0.0);
        assert!(bc[b.index()] > 0.0, "b={}", bc[b.index()]);
        assert!(bc[c.index()] > 0.0, "c={}", bc[c.index()]);
    }
}

#[test]
fn reachability_counts_edges_matches_toy_graph() {
    // 0 -> 1 -> 2
    let n = 3;
    let edges = vec![(0usize, 1usize), (1usize, 2usize)];
    let (dependents, dependencies) = reachability_counts_edges(n, &edges);
    assert_eq!(dependencies, vec![2, 1, 0]);
    assert_eq!(dependents, vec![0, 1, 2]);
}

#[cfg(feature = "petgraph")]
mod petgraph_multigraph {
    use graphops::{pagerank_weighted, PageRankConfig};
    use petgraph::prelude::*;

    /// Parallel edges a->b (1.0, 2.0) must act as one edge of weight 3.0.
    #[test]
    fn weighted_pagerank_sums_parallel_edge_weights() {
        let mut multi: DiGraph<(), f64> = DiGraph::new();
        let (a, b, c) = (multi.add_node(()), multi.add_node(()), multi.add_node(()));
        multi.add_edge(a, b, 1.0);
        multi.add_edge(a, b, 2.0);
        multi.add_edge(a, c, 1.0);
        multi.add_edge(b, a, 1.0);
        multi.add_edge(c, a, 1.0);

        let mut simple: DiGraph<(), f64> = DiGraph::new();
        let (a, b, c) = (
            simple.add_node(()),
            simple.add_node(()),
            simple.add_node(()),
        );
        simple.add_edge(a, b, 3.0);
        simple.add_edge(a, c, 1.0);
        simple.add_edge(b, a, 1.0);
        simple.add_edge(c, a, 1.0);

        let cfg = PageRankConfig::default();
        let m = pagerank_weighted(&multi, cfg);
        let s = pagerank_weighted(&simple, cfg);
        for (x, y) in m.iter().zip(&s) {
            assert!((x - y).abs() < 1e-9, "multi={m:?} simple={s:?}");
        }
    }
}

/// node2vec second-order transitions (Grover & Leskovec 2016, Sec. 3.2.2):
/// after stepping t -> v, the next node x is chosen with weight 1/p if x == t,
/// 1 if x is adjacent to t, and 1/q otherwise.
#[test]
fn node2vec_second_step_frequencies_match_the_paper_weights() {
    use graphops::{generate_biased_walks_from_nodes, AdjacencyMatrix, WalkConfig};

    // Edges 0-1, 0-2, 1-2, 1-3. From t = 0, v = 1: x = 0 returns, x = 2 is
    // adjacent to t, x = 3 is not.
    let adj = vec![
        vec![0.0, 1.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0, 1.0],
        vec![1.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
    ];
    let (p, q) = (2.0_f32, 0.5_f32);
    let config = WalkConfig {
        length: 3,
        walks_per_node: 40_000,
        p,
        q,
        seed: 11,
    };
    let walks = generate_biased_walks_from_nodes(&AdjacencyMatrix(&adj), &[0], config);

    let mut counts = [0usize; 4];
    for w in walks.iter().filter(|w| w.len() == 3 && w[1] == 1) {
        counts[w[2]] += 1;
    }
    let total: usize = counts.iter().sum();
    assert!(total > 10_000, "too few t=0, v=1 walks: {total}");

    let weights = [1.0 / p as f64, 0.0, 1.0, 1.0 / q as f64];
    let z: f64 = weights.iter().sum();
    for x in [0, 2, 3] {
        let observed = counts[x] as f64 / total as f64;
        let expected = weights[x] / z;
        assert!(
            (observed - expected).abs() < 0.02,
            "x={x}: observed {observed:.4}, expected {expected:.4}"
        );
    }
    assert_eq!(counts[1], 0);
}
