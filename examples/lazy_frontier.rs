/// Lazy graph expansion for evidence-driven retrieval.
///
/// This keeps the "lazygraph" idea example-local: start from a query node,
/// expand the most promising frontier edge first, and stop once enough evidence
/// has been found. The eager baseline expands every node within the same depth
/// budget.
///
/// ```sh
/// cargo run --example lazy_frontier
/// ```
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};

const START: usize = 0;
const MAX_DEPTH: usize = 3;
const STOP_EVIDENCE: f32 = 0.90;

#[derive(Clone, Copy)]
struct Edge {
    to: usize,
    affinity: f32,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    node: usize,
    depth: usize,
    priority: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
            && self.depth == other.depth
            && self.priority.to_bits() == other.priority.to_bits()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| other.depth.cmp(&self.depth))
            .then_with(|| other.node.cmp(&self.node))
    }
}

#[derive(Clone, Debug)]
struct SearchRun {
    visited: Vec<usize>,
    best_node: usize,
    best_evidence: f32,
}

fn main() {
    let labels = [
        "query",
        "retrieval",
        "algebra",
        "recipes",
        "ColBERT",
        "Matryoshka",
        "Clifford",
        "Koopman",
        "shopping",
        "salsa",
        "gardens",
        "index tuning",
        "additive codebooks",
    ];
    let graph = make_graph();
    let evidence = [
        0.00, 0.20, 0.35, 0.05, 0.55, 0.72, 0.45, 0.42, 0.10, 0.04, 0.02, 0.58, 0.97,
    ];

    let eager = eager_expand(&graph, &evidence);
    let lazy = lazy_expand(&graph, &evidence);

    println!("query: Matryoshka quantization follow-up");
    println!("strategy  visited  best evidence");
    println!(
        "eager     {:>7}  {} ({:.2})",
        eager.visited.len(),
        labels[eager.best_node],
        eager.best_evidence
    );
    println!(
        "lazy      {:>7}  {} ({:.2})",
        lazy.visited.len(),
        labels[lazy.best_node],
        lazy.best_evidence
    );
    println!(
        "lazy visit order: {}",
        lazy.visited
            .iter()
            .map(|&node| labels[node])
            .collect::<Vec<_>>()
            .join(" -> ")
    );

    assert_eq!(lazy.best_node, eager.best_node);
    assert!(lazy.visited.len() < eager.visited.len());
}

fn make_graph() -> Vec<Vec<Edge>> {
    vec![
        vec![
            Edge {
                to: 1,
                affinity: 0.95,
            },
            Edge {
                to: 2,
                affinity: 0.85,
            },
            Edge {
                to: 3,
                affinity: 0.15,
            },
        ],
        vec![
            Edge {
                to: 4,
                affinity: 0.90,
            },
            Edge {
                to: 5,
                affinity: 0.88,
            },
            Edge {
                to: 11,
                affinity: 0.80,
            },
        ],
        vec![
            Edge {
                to: 6,
                affinity: 0.75,
            },
            Edge {
                to: 7,
                affinity: 0.70,
            },
        ],
        vec![
            Edge {
                to: 8,
                affinity: 0.50,
            },
            Edge {
                to: 9,
                affinity: 0.40,
            },
            Edge {
                to: 10,
                affinity: 0.30,
            },
        ],
        vec![],
        vec![Edge {
            to: 12,
            affinity: 0.95,
        }],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    ]
}

fn eager_expand(graph: &[Vec<Edge>], evidence: &[f32]) -> SearchRun {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut queue = VecDeque::from([(START, 0usize)]);

    while let Some((node, depth)) = queue.pop_front() {
        if !visited.insert(node) {
            continue;
        }
        order.push(node);

        if depth < MAX_DEPTH {
            for edge in &graph[node] {
                queue.push_back((edge.to, depth + 1));
            }
        }
    }

    summarize(order, evidence)
}

fn lazy_expand(graph: &[Vec<Edge>], evidence: &[f32]) -> SearchRun {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut heap = BinaryHeap::from([Candidate {
        node: START,
        depth: 0,
        priority: 1.0,
    }]);

    let mut best_node = START;
    let mut best_evidence = evidence[START];

    while let Some(candidate) = heap.pop() {
        if !visited.insert(candidate.node) {
            continue;
        }
        order.push(candidate.node);

        if evidence[candidate.node] > best_evidence {
            best_node = candidate.node;
            best_evidence = evidence[candidate.node];
        }
        if best_evidence >= STOP_EVIDENCE {
            break;
        }

        if candidate.depth < MAX_DEPTH {
            for edge in &graph[candidate.node] {
                heap.push(Candidate {
                    node: edge.to,
                    depth: candidate.depth + 1,
                    priority: candidate.priority * edge.affinity,
                });
            }
        }
    }

    SearchRun {
        visited: order,
        best_node,
        best_evidence,
    }
}

fn summarize(visited: Vec<usize>, evidence: &[f32]) -> SearchRun {
    let (best_node, best_evidence) = visited
        .iter()
        .map(|&node| (node, evidence[node]))
        .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .unwrap();

    SearchRun {
        visited,
        best_node,
        best_evidence,
    }
}
