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
