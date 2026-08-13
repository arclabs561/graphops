use criterion::{criterion_group, criterion_main, Criterion};
use graphops::{
    pagerank, pagerank_ref, personalized_pagerank, personalized_pagerank_ref, Graph, GraphRef,
    PageRankConfig,
};
use std::hint::black_box;

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

fn pagerank_paths(c: &mut Criterion) {
    let n = 10_000;
    let graph = Adjacency(
        (0..n)
            .map(|u| {
                if u % 17 == 0 {
                    Vec::new()
                } else {
                    (1..=1 + u % 8).map(|offset| (u + offset) % n).collect()
                }
            })
            .collect(),
    );
    let config = PageRankConfig {
        max_iterations: 20,
        tolerance: 1e-12,
        ..PageRankConfig::default()
    };

    let mut group = c.benchmark_group("pagerank_10k_varied_degree");
    group.bench_function("owned", |b| {
        b.iter(|| black_box(pagerank(black_box(&graph), black_box(config))))
    });
    group.bench_function("borrowed", |b| {
        b.iter(|| black_box(pagerank_ref(black_box(&graph), black_box(config))))
    });
    group.finish();

    let personalization: Vec<f64> = (0..n)
        .map(|u| if u % 101 == 0 { 1.0 } else { 0.0 })
        .collect();
    let mut group = c.benchmark_group("personalized_pagerank_10k_varied_degree");
    group.bench_function("owned", |b| {
        b.iter(|| {
            black_box(personalized_pagerank(
                black_box(&graph),
                black_box(config),
                black_box(&personalization),
            ))
        })
    });
    group.bench_function("borrowed", |b| {
        b.iter(|| {
            black_box(personalized_pagerank_ref(
                black_box(&graph),
                black_box(config),
                black_box(&personalization),
            ))
        })
    });
    group.finish();
}

criterion_group!(benches, pagerank_paths);
criterion_main!(benches);
