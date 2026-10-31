//! Newman betweenness regressions. Reference values from networkx 3.x
//! `current_flow_betweenness_centrality`.

use graphops::graph::GraphRef;
use graphops::{newman_betweenness, newman_betweenness_run, NewmanBetweennessConfig};

struct VG(Vec<Vec<usize>>);
impl GraphRef for VG {
    fn node_count(&self) -> usize {
        self.0.len()
    }
    fn neighbors_ref(&self, n: usize) -> &[usize] {
        &self.0[n]
    }
}

fn path(n: usize) -> VG {
    VG((0..n)
        .map(|k| {
            let mut v = Vec::new();
            if k > 0 {
                v.push(k - 1);
            }
            if k + 1 < n {
                v.push(k + 1);
            }
            v
        })
        .collect())
}

// The docs say exact mode "is fine" for n <= ~500. A 40-node path is tiny,
// but the Jacobi solve on it needs O(n^2) sweeps, so max_iter = 200 stops it
// early and the default entry point returns unconverged potentials silently.
#[test]
fn newman_default_config_converges_on_small_path() {
    let g = path(40);
    let run = newman_betweenness_run(&g, NewmanBetweennessConfig::default());
    assert!(
        run.converged,
        "default config did not converge on a 40-node path"
    );
}

// Two disjoint, identical triangles: every node is equivalent under an
// automorphism, so every score must be equal; "unreachable pairs contribute 0".
#[test]
fn newman_disconnected_is_symmetric() {
    let g = VG(vec![
        vec![1, 2],
        vec![0, 2],
        vec![0, 1],
        vec![4, 5],
        vec![3, 5],
        vec![3, 4],
    ]);
    let s = newman_betweenness(&g, NewmanBetweennessConfig::default());
    for i in 1..6 {
        assert!((s[i] - s[0]).abs() < 1e-6, "asymmetric scores {s:?}");
    }
}

// Path 0-1-2 plus an isolated node 3 (which is the pinned node n-1). If
// unreachable pairs contribute 0, the raw pair sums equal those of the bare
// path P3, and only the (n-1)(n-2) normaliser changes: 2 -> 6. So every
// score must be exactly one third of the P3 score, under any endpoint
// convention.
#[test]
fn newman_isolated_pinned_node_only_rescales() {
    let p3 = newman_betweenness(&path(3), NewmanBetweennessConfig::default());
    let g = VG(vec![vec![1], vec![0, 2], vec![1], vec![]]);
    let s = newman_betweenness(&g, NewmanBetweennessConfig::default());
    for i in 0..3 {
        assert!(
            (s[i] - p3[i] / 3.0).abs() < 1e-6,
            "P3+isolated {s:?} vs P3/3 {:?}",
            p3.iter().map(|x| x / 3.0).collect::<Vec<_>>()
        );
    }
    assert_eq!(s[3], 0.0);
}
