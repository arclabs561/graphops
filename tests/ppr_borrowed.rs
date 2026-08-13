use graphops::{
    personalized_pagerank, personalized_pagerank_ref, personalized_pagerank_ref_checked,
    personalized_pagerank_ref_checked_run, personalized_pagerank_ref_run, Graph, GraphRef,
    PageRankConfig,
};
use proptest::prelude::*;

#[derive(Debug)]
struct Adjacency(Vec<Vec<usize>>);

impl Graph for Adjacency {
    fn node_count(&self) -> usize {
        self.0.len()
    }

    fn neighbors(&self, node: usize) -> Vec<usize> {
        self.0[node].clone()
    }
}

impl GraphRef for Adjacency {
    fn node_count(&self) -> usize {
        self.0.len()
    }

    fn neighbors_ref(&self, node: usize) -> &[usize] {
        &self.0[node]
    }
}

fn strict_config() -> PageRankConfig {
    PageRankConfig {
        damping: 0.85,
        max_iterations: 500,
        tolerance: 1e-12,
    }
}

#[test]
fn borrowed_matches_owned_diagnostics_exactly() {
    let graph = Adjacency(vec![vec![1, 2], vec![], vec![0, 1], vec![2]]);
    let personalization = [3.0, 0.0, 1.0, 2.0];

    let owned = graphops::personalized_pagerank_run(&graph, strict_config(), &personalization);
    let borrowed = personalized_pagerank_ref_run(&graph, strict_config(), &personalization);

    assert_eq!(owned.scores, borrowed.scores);
    assert_eq!(owned.iterations, borrowed.iterations);
    assert_eq!(owned.diff_l1, borrowed.diff_l1);
    assert_eq!(owned.converged, borrowed.converged);
}

#[test]
fn dangling_mass_follows_personalization() {
    let graph = Adjacency(vec![vec![], vec![]]);
    let scores = personalized_pagerank_ref(&graph, strict_config(), &[3.0, 1.0]);
    assert_eq!(scores, vec![0.75, 0.25]);
}

#[test]
fn duplicate_edges_contribute_with_multiplicity() {
    let duplicated = Adjacency(vec![vec![1, 1, 2], vec![0], vec![0]]);
    let one_iteration = PageRankConfig {
        damping: 1.0,
        max_iterations: 1,
        tolerance: 1e-12,
    };
    let scores = personalized_pagerank_ref(&duplicated, one_iteration, &[1.0, 0.0, 0.0]);

    assert_eq!(scores, vec![0.0, 2.0 / 3.0, 1.0 / 3.0]);
}

#[test]
fn checked_borrowed_entrypoints_validate_inputs() {
    let graph = Adjacency(vec![vec![1], vec![]]);
    assert!(personalized_pagerank_ref_checked(&graph, strict_config(), &[1.0]).is_err());
    assert!(personalized_pagerank_ref_checked_run(&graph, strict_config(), &[0.0, 0.0]).is_err());
}

proptest! {
    #[test]
    fn borrowed_and_owned_are_exact_for_generated_graphs(
        n in 1usize..20,
        raw_edges in prop::collection::vec((0usize..20, 0usize..20), 0..100),
        raw_personalization in prop::collection::vec(0u16..1000, 1..20),
    ) {
        let mut adjacency = vec![Vec::new(); n];
        for (source, target) in raw_edges {
            adjacency[source % n].push(target % n);
        }
        let mut personalization = vec![0.0; n];
        for (i, value) in raw_personalization.into_iter().enumerate().take(n) {
            personalization[i] = f64::from(value);
        }
        if personalization.iter().all(|&value| value == 0.0) {
            personalization[0] = 1.0;
        }
        let graph = Adjacency(adjacency);

        prop_assert_eq!(
            personalized_pagerank(&graph, strict_config(), &personalization),
            personalized_pagerank_ref(&graph, strict_config(), &personalization),
        );
    }
}
