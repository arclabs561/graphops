//! Regression tests for fixed bugs. Each test states the correct behavior and
//! the reference it comes from.

use graphops::graph::GraphRef;
use graphops::{newman_betweenness, NewmanBetweennessConfig};

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

fn close(got: &[f64], want: &[f64], tol: f64) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() <= tol)
}

/// Current-flow betweenness excludes the source and sink of each pair, so the
/// `(n-1)(n-2)` normalizer maps scores into [0, 1]. Reference values from
/// networkx 3.7 `current_flow_betweenness_centrality`.
#[test]
fn newman_betweenness_excludes_endpoints_like_networkx() {
    let star = VG(vec![
        vec![1, 2, 3, 4, 5],
        vec![0],
        vec![0],
        vec![0],
        vec![0],
        vec![0],
    ]);
    let s = newman_betweenness(&star, NewmanBetweennessConfig::default());
    let want = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert!(close(&s, &want, 1e-6), "star: got {s:?}, networkx {want:?}");

    let s = newman_betweenness(&path(4), NewmanBetweennessConfig::default());
    let want = [0.0, 2.0 / 3.0, 2.0 / 3.0, 0.0];
    assert!(close(&s, &want, 1e-6), "P4: got {s:?}, networkx {want:?}");
}

/// node2vec requires p, q > 0 (Grover and Leskovec 2016). With p or q zero,
/// infinite or NaN, the rejection sampler's acceptance ratio is 0 or NaN for
/// every candidate on some step, so it would loop forever; the call must
/// return or panic instead.
#[test]
fn biased_walk_rejects_non_positive_or_non_finite_p_q() {
    use graphops::{generate_biased_walks_ref, WalkConfig};
    for (p, q) in [
        (0.0, 1.0),
        (1.0, 0.0),
        (f32::NAN, 1.0),
        (f32::INFINITY, 1.0),
    ] {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let r = std::panic::catch_unwind(|| {
                let cfg = WalkConfig {
                    length: 4,
                    walks_per_node: 1,
                    p,
                    q,
                    ..WalkConfig::default()
                };
                generate_biased_walks_ref(&path(3), cfg).len()
            });
            let _ = tx.send(r.is_err());
        });
        let panicked = rx.recv_timeout(std::time::Duration::from_secs(5));
        assert_eq!(
            panicked,
            Ok(true),
            "p = {p}, q = {q}: expected a panic, got {panicked:?} (timeout means the sampler looped)"
        );
    }
}

/// Each parallel walk gets its own RNG stream. Seeds built as
/// `seed ^ node ^ index` collide whenever `node ^ index` matches, which gave
/// distinct walks identical step sequences. On a 64-cycle, two independent
/// 30-step direction sequences agree with probability 2^-30.
#[cfg(feature = "parallel")]
#[test]
fn parallel_walks_use_independent_streams() {
    use graphops::{
        generate_biased_walks_precomp_ref_parallel_from_nodes, generate_biased_walks_ref_parallel,
        generate_walks_ref_parallel, PrecomputedBiasedWalks, WalkConfig,
    };
    let n = 64;
    let g = VG((0..n).map(|k| vec![(k + n - 1) % n, (k + 1) % n]).collect());
    let cfg = WalkConfig {
        length: 31,
        walks_per_node: 1,
        seed: 3,
        ..WalkConfig::default()
    };
    let pre = PrecomputedBiasedWalks::new(&g, cfg.p, cfg.q);
    let starts: Vec<usize> = (0..n).collect();
    for (name, walks) in [
        ("unbiased", generate_walks_ref_parallel(&g, cfg)),
        ("biased", generate_biased_walks_ref_parallel(&g, cfg)),
        (
            "precomputed",
            generate_biased_walks_precomp_ref_parallel_from_nodes(&pre, &starts, cfg),
        ),
    ] {
        let dirs: Vec<Vec<bool>> = walks
            .iter()
            .map(|w| w.windows(2).map(|p| p[1] == (p[0] + 1) % n).collect())
            .collect();
        let mut same = 0;
        for a in 0..dirs.len() {
            for b in (a + 1)..dirs.len() {
                same += usize::from(dirs[a] == dirs[b]);
            }
        }
        assert_eq!(
            same, 0,
            "{name}: {same} pairs of walks share a step sequence"
        );
    }
}
