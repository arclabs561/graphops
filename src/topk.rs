//! Ranking utilities.

use ordered_float::NotNan;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Return the top-k entries by score (descending).
///
/// # Examples
///
/// ```
/// let scores = vec![0.1, 0.5, 0.3, 0.9, 0.2];
/// let top = graphops::top_k(&scores, 2);
/// assert_eq!(top[0].0, 3); // node 3 has score 0.9
/// assert_eq!(top[1].0, 1); // node 1 has score 0.5
/// ```
pub fn top_k(scores: &[f64], k: usize) -> Vec<(usize, f64)> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (i, &score) in scores.iter().enumerate() {
        if !score.is_finite() || score <= 0.0 {
            continue;
        }
        let s = NotNan::new(score).unwrap();
        if heap.len() < k {
            heap.push(Reverse((s, i)));
        } else if let Some(&Reverse((min_score, _))) = heap.peek() {
            if s > min_score {
                heap.pop();
                heap.push(Reverse((s, i)));
            }
        }
    }
    let mut results: Vec<(usize, f64)> = heap
        .into_iter()
        .map(|Reverse((s, i))| (i, s.into_inner()))
        .collect();
    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Normalize scores to sum to 1.0. No-op if sum is zero.
///
/// # Examples
///
/// ```
/// let mut s = vec![2.0, 3.0, 5.0];
/// graphops::normalize(&mut s);
/// assert!((s[0] - 0.2).abs() < 1e-12);
/// assert!((s[2] - 0.5).abs() < 1e-12);
/// ```
pub fn normalize(scores: &mut [f64]) {
    let sum: f64 = scores.iter().sum();
    if sum > 0.0 {
        for s in scores {
            *s /= sum;
        }
    }
}
