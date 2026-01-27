//! Personalized PageRank.

use crate::graph::Graph;
use crate::pagerank::{PageRankConfig, PageRankRun};
use crate::{Error, Result};

pub fn personalized_pagerank_checked<G: Graph>(
    graph: &G,
    config: PageRankConfig,
    personalization: &[f64],
) -> Result<Vec<f64>> {
    config.validate()?;
    let n = graph.node_count();
    if personalization.len() != n {
        return Err(Error::InvalidParameter(format!(
            "personalization length must equal node_count (len={} node_count={})",
            personalization.len(),
            n
        )));
    }
    for &x in personalization {
        if !x.is_finite() {
            return Err(Error::InvalidParameter(
                "personalization entries must be finite".to_string(),
            ));
        }
        if x < 0.0 {
            return Err(Error::InvalidParameter(
                "personalization entries must be non-negative".to_string(),
            ));
        }
    }
    let sum: f64 = personalization.iter().sum();
    if sum <= 0.0 {
        return Err(Error::InvalidParameter(
            "personalization sum must be > 0".to_string(),
        ));
    }
    Ok(personalized_pagerank(graph, config, personalization))
}

pub fn personalized_pagerank<G: Graph>(
    graph: &G,
    config: PageRankConfig,
    personalization: &[f64],
) -> Vec<f64> {
    personalized_pagerank_run(graph, config, personalization).scores
}

pub fn personalized_pagerank_checked_run<G: Graph>(
    graph: &G,
    config: PageRankConfig,
    personalization: &[f64],
) -> Result<PageRankRun> {
    // validate via existing checked entrypoint
    let _ = personalized_pagerank_checked(graph, config, personalization)?;
    Ok(personalized_pagerank_run(graph, config, personalization))
}

pub fn personalized_pagerank_run<G: Graph>(
    graph: &G,
    config: PageRankConfig,
    personalization: &[f64],
) -> PageRankRun {
    let n = graph.node_count();
    if n == 0 {
        return PageRankRun {
            scores: Vec::new(),
            iterations: 0,
            diff_l1: 0.0,
            converged: true,
        };
    }
    let p_sum: f64 = personalization.iter().sum();
    let p_vec: Vec<f64> = if p_sum > 0.0 {
        personalization.iter().map(|&x| x / p_sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    };
    let mut scores = p_vec.clone();
    let mut new_scores = vec![0.0; n];
    let out_degrees: Vec<usize> = (0..n).map(|i| graph.out_degree(i)).collect();

    let mut iters = 0usize;
    let mut last_diff = f64::INFINITY;
    let mut converged = false;
    for _ in 0..config.max_iterations {
        iters += 1;
        let dangling_sum: f64 = out_degrees
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(i, _)| scores[i])
            .sum();
        for i in 0..n {
            new_scores[i] =
                (1.0 - config.damping) * p_vec[i] + config.damping * dangling_sum * p_vec[i];
        }
        for u in 0..n {
            let deg = out_degrees[u];
            if deg > 0 {
                let share = config.damping * scores[u] / deg as f64;
                for v in graph.neighbors(u) {
                    new_scores[v] += share;
                }
            }
        }
        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();
        last_diff = diff;
        std::mem::swap(&mut scores, &mut new_scores);
        if diff < config.tolerance {
            converged = true;
            break;
        }
    }
    PageRankRun {
        scores,
        iterations: iters,
        diff_l1: last_diff,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AdjacencyMatrix;

    #[test]
    fn ppr_checked_rejects_wrong_len() {
        let adj = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
        let g = AdjacencyMatrix(&adj);
        let p = vec![1.0];
        let err = personalized_pagerank_checked(&g, PageRankConfig::default(), &p).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("personalization length"));
    }

    #[test]
    fn ppr_checked_rejects_zero_sum() {
        let adj = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
        let g = AdjacencyMatrix(&adj);
        let p = vec![0.0, 0.0];
        let err = personalized_pagerank_checked(&g, PageRankConfig::default(), &p).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("sum"));
    }
}
