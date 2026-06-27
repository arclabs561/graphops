# /// script
# requires-python = ">=3.10"
# dependencies = ["networkx", "numpy", "scipy"]
# ///
"""Rosetta fixture generator for graphops PageRank.

Provenance for graphops_pagerank.json. PageRank reference values come from
networkx.pagerank. graphops and networkx use the same random-surfer model
(uniform teleport (1-alpha)/n, dangling mass redistributed uniformly), so the
unique fixed point agrees once both are converged tightly. The two stopping
rules differ (graphops: L1 delta < tol; networkx: L1 delta < n*tol), so both are
run to a much tighter tolerance than the 1e-9 comparison to remove that as a
variable.

Covered: pagerank (unweighted, with a dangling node) and pagerank_weighted.
Deferred: Brandes betweenness (petgraph-feature-gated; normalization matches
networkx 1/((n-1)(n-2)) but needs the feature wired in CI first); connected
components (no single graphops function maps cleanly to networkx).

Regenerate: uv run tests/fixtures/rosetta/gen_graphops.py
"""

import json
import platform
from pathlib import Path

import networkx as nx
import numpy as np

DAMPING = 0.85
# Tighter than the 1e-9 comparison so the differing stop rules do not matter.
NX_TOL = 1e-13
MAX_ITER = 1000

# Directed graph, node 5 is dangling (all-zero row -> no out-edges). 0/1 weights.
adj_unweighted = [
    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # dangling
]

# Weighted directed graph (positive weights, same sparsity is not required).
adj_weighted = [
    [0.0, 2.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.5, 0.0, 0.0, 3.0],
    [1.0, 0.0, 0.0, 2.5, 0.0, 0.0],
    [0.0, 0.8, 0.0, 0.0, 1.2, 0.0],
    [0.6, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # dangling
]

n = len(adj_unweighted)


def pagerank_scores(matrix, weight):
    a = np.array(matrix)
    g = nx.from_numpy_array(a, create_using=nx.DiGraph)
    pr = nx.pagerank(g, alpha=DAMPING, tol=NX_TOL, max_iter=MAX_ITER, weight=weight)
    return [pr[i] for i in range(n)]


# Unweighted: all present edges have weight 1, so weighted and unweighted PageRank
# coincide; pass weight=None to make graphops's "split uniformly" semantics explicit.
pagerank = pagerank_scores(adj_unweighted, weight=None)
pagerank_weighted = pagerank_scores(adj_weighted, weight="weight")

fixture = {
    "provenance": {
        "generator": "gen_graphops.py",
        "library": "networkx",
        "networkx_version": nx.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "note": "Both sides converged far tighter than the 1e-9 comparison.",
    },
    "params": {
        "damping": DAMPING,
        "graphops_tolerance": 1e-12,
        "max_iterations": MAX_ITER,
    },
    "adj_unweighted": adj_unweighted,
    "adj_weighted": adj_weighted,
    "expected": {
        "pagerank": pagerank,
        "pagerank_weighted": pagerank_weighted,
    },
}

out = Path(__file__).parent / "graphops_pagerank.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
print("pagerank         ", [f"{v:.6f}" for v in pagerank])
print("pagerank_weighted", [f"{v:.6f}" for v in pagerank_weighted])
print(f"wrote {out}")
