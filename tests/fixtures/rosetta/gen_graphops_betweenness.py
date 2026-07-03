# /// script
# requires-python = ">=3.10"
# dependencies = ["networkx", "numpy"]
# ///
"""Rosetta fixture generator for graphops Brandes betweenness centrality.

Provenance for graphops_betweenness.json. Reference values come from
networkx.betweenness_centrality on DIRECTED graphs with normalized=True and
endpoints=False (its defaults). graphops::betweenness_centrality (Brandes,
petgraph-feature-gated) targets the same convention:

  - directed shortest-path counting (sigma split over multiple shortest paths),
  - endpoints excluded from their own pair's dependency,
  - directed normalization 1 / ((n-1)(n-2)).

Brandes betweenness on integer-hop unweighted graphs is exact rational
arithmetic on both sides, so the only gap is f64 rounding; the Rust test uses the
1e-9 EXACT tolerance class, not an f32 floor.

Graphs (node ids are insertion order 0..n-1, matching petgraph NodeIndex and
graphops output ordering):
  - path_5:   0->1->2->3->4, single shortest path per pair.
  - diamond_6: two equal-length shortest paths between some pairs, so sigma > 1
    and the path-count split is actually exercised.
  - random_dag_15: a fixed-seed random DAG, the general mixed case.

Regenerate: uv run tests/fixtures/rosetta/gen_graphops_betweenness.py
"""

import json
import platform
from pathlib import Path

import networkx as nx
import numpy as np


def make_graph(name, n, edges):
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    bc = nx.betweenness_centrality(g, normalized=True, endpoints=False)
    scores = [bc[i] for i in range(n)]
    return {
        "name": name,
        "n": n,
        "edges": [[int(u), int(v)] for (u, v) in edges],
        "expected": {"betweenness": scores},
    }


cases = []

# path_5: 0->1->2->3->4. Middle nodes carry all the flow; ends carry none.
cases.append(make_graph("path_5", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]))

# diamond_6: 0 splits to {1,2}, both rejoin at 3, then 3->4->5. The pair (0,3)
# has TWO shortest paths (0-1-3 and 0-2-3), so sigma[3]=2 and nodes 1,2 each get
# a half share -- this is the sigma-splitting case.
cases.append(
    make_graph(
        "diamond_6",
        6,
        [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)],
    )
)

# random_dag_15: fixed-seed random DAG. Edges only from lower to higher index
# (guarantees acyclicity), each candidate edge included with prob p.
rng = np.random.default_rng(20260702)
n = 15
p = 0.25
dag_edges = []
for u in range(n):
    for v in range(u + 1, n):
        if rng.random() < p:
            dag_edges.append((u, v))
cases.append(make_graph("random_dag_15", n, dag_edges))

fixture = {
    "provenance": {
        "generator": "gen_graphops_betweenness.py",
        "library": "networkx",
        "networkx_version": nx.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": 20260702,
        "note": "networkx betweenness_centrality(normalized=True, endpoints=False) on DiGraphs; exact rational, f64 floor.",
    },
    "cases": cases,
}

out = Path(__file__).parent / "graphops_betweenness.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
for c in cases:
    scores = c["expected"]["betweenness"]
    print(
        f"{c['name']:16s} n={c['n']:2d} edges={len(c['edges']):3d} "
        f"betweenness={[f'{v:.5f}' for v in scores]}"
    )
print(f"wrote {out}")
