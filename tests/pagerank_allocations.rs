use graphops::{
    pagerank, pagerank_ref, personalized_pagerank, personalized_pagerank_ref, Graph, GraphRef,
    PageRankConfig,
};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::alloc::System;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

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

#[test]
fn borrowed_pagerank_avoids_per_node_adjacency_allocations() {
    let n = 2_048;
    let adjacency = Adjacency(
        (0..n)
            .map(|u| (1..=8).map(|offset| (u + offset) % n).collect())
            .collect(),
    );
    let config = PageRankConfig {
        max_iterations: 10,
        tolerance: 1e-12,
        ..PageRankConfig::default()
    };

    let owned_region = Region::new(GLOBAL);
    let owned = pagerank(&adjacency, config);
    let owned_stats = owned_region.change();
    std::hint::black_box(&owned);

    let borrowed_region = Region::new(GLOBAL);
    let borrowed = pagerank_ref(&adjacency, config);
    let borrowed_stats = borrowed_region.change();
    std::hint::black_box(&borrowed);

    assert_eq!(owned, borrowed);
    assert!(
        owned_stats.allocations >= borrowed_stats.allocations + n,
        "owned={} borrowed={} allocations",
        owned_stats.allocations,
        borrowed_stats.allocations
    );
    assert!(
        owned_stats.bytes_allocated >= borrowed_stats.bytes_allocated + n * 8 * size_of::<usize>(),
        "owned={} borrowed={} allocated bytes",
        owned_stats.bytes_allocated,
        borrowed_stats.bytes_allocated
    );
}

#[test]
fn borrowed_personalized_pagerank_avoids_adjacency_allocations() {
    let n = 2_048;
    let adjacency = Adjacency(
        (0..n)
            .map(|u| (1..=8).map(|offset| (u + offset) % n).collect())
            .collect(),
    );
    let personalization: Vec<f64> = (0..n)
        .map(|u| if u % 31 == 0 { 1.0 } else { 0.0 })
        .collect();
    let config = PageRankConfig {
        max_iterations: 10,
        tolerance: 1e-12,
        ..PageRankConfig::default()
    };

    let owned_region = Region::new(GLOBAL);
    let owned = personalized_pagerank(&adjacency, config, &personalization);
    let owned_stats = owned_region.change();
    std::hint::black_box(&owned);

    let borrowed_region = Region::new(GLOBAL);
    let borrowed = personalized_pagerank_ref(&adjacency, config, &personalization);
    let borrowed_stats = borrowed_region.change();
    std::hint::black_box(&borrowed);

    assert_eq!(owned, borrowed);
    assert!(
        owned_stats.allocations >= borrowed_stats.allocations + n,
        "owned={} borrowed={} allocations",
        owned_stats.allocations,
        borrowed_stats.allocations
    );
    assert!(
        owned_stats.bytes_allocated >= borrowed_stats.bytes_allocated + n * 8 * size_of::<usize>(),
        "owned={} borrowed={} allocated bytes",
        owned_stats.bytes_allocated,
        borrowed_stats.bytes_allocated
    );
}
