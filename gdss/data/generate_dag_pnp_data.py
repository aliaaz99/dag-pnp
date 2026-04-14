"""
Generate a GDSS-compatible dataset using random_dag() from dag-pnp/data_utils.py.

Each graph is a weighted Erdos-Renyi DAG drawn with the ERk convention:
    E[# edges] = k * d   (default k=4, so ER4)

Weights are uniform in [weight_low, weight_high] with random signs,
matching the dag-pnp defaults (0.5 to 1.0).

Usage:
    # from the GDSS root directory
    python data/generate_dag_pnp_data.py --d 20 --n_graphs 700
    python data/generate_dag_pnp_data.py --d 20 --n_graphs 700 --k 2 --name sem_ER2_d20
"""

import os
import sys
import pickle
import argparse

import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# Locate data_utils.py by walking up from this file's location.
# Works whether GDSS lives at repo-root/gdss/ or anywhere else relative
# to the dag-pnp package that contains data_utils.py.
# ---------------------------------------------------------------------------
def _find_data_utils() -> str:
    """Return the directory that contains data_utils.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):          # search up to 5 levels up
        if os.path.isfile(os.path.join(here, "data_utils.py")):
            return here
        here = os.path.dirname(here)
    raise FileNotFoundError(
        "Could not find data_utils.py within 5 parent directories of this script."
    )

sys.path.insert(0, _find_data_utils())

from data_utils import random_dag           # noqa: E402


def generate(
    n_graphs: int,
    d: int,
    k: int = 4,
    weight_low: float = 0.5,
    weight_high: float = 1.0,
    seed: int = 42,
) -> list:
    """Return a list of nx.DiGraph objects with float edge weights."""
    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(n_graphs):
        graph_seed = int(rng.integers(0, 2**31 - 1))
        W = random_dag(d=d, k=k, weight_low=weight_low, weight_high=weight_high,
                       seed=graph_seed)
        G = nx.from_numpy_array(W, create_using=nx.DiGraph)
        G.remove_edges_from(nx.selfloop_edges(G))
        graphs.append(G)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_graphs}")
    return graphs


def print_stats(graphs: list, d: int, k: int) -> None:
    edge_counts = [g.number_of_edges() for g in graphs]
    weights = [data["weight"] for g in graphs
               for _, _, data in g.edges(data=True)]
    weights = np.array(weights)
    print(f"\n=== Dataset statistics ===")
    print(f"  d={d}, k={k}  (expected {k*d} edges/graph)")
    print(f"  # graphs   : {len(graphs)}")
    print(f"  Edges/graph: {np.mean(edge_counts):.1f} ± {np.std(edge_counts):.1f}  "
          f"(min={min(edge_counts)}, max={max(edge_counts)})")
    print(f"  Density    : {np.mean(edge_counts) / (d*(d-1)):.3f}")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GDSS training data using random_dag() from dag-pnp"
    )
    parser.add_argument("--n_graphs",     type=int,   default=700)
    parser.add_argument("--d",            type=int,   default=20, choices=[10, 20, 30])
    parser.add_argument("--k",            type=int,   default=4,
                        help="Expected edges per node (ER4 = k=4, default)")
    parser.add_argument("--weight_low",   type=float, default=0.5)
    parser.add_argument("--weight_high",  type=float, default=1.0)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--data_dir",     type=str,   default="data")
    parser.add_argument("--name",         type=str,   default=None,
                        help="Output filename stem (default: dag_pnp_ER<k>_d<d>)")
    args = parser.parse_args()

    print(f"Generating {args.n_graphs} random_dag graphs  "
          f"(d={args.d}, k={args.k}, weights=[{args.weight_low},{args.weight_high}]) ...")

    graphs = generate(
        n_graphs=args.n_graphs,
        d=args.d,
        k=args.k,
        weight_low=args.weight_low,
        weight_high=args.weight_high,
        seed=args.seed,
    )

    print_stats(graphs, args.d, args.k)

    name = args.name or f"dag_pnp_ER{args.k}_d{args.d}"
    os.makedirs(args.data_dir, exist_ok=True)
    out_path = os.path.join(args.data_dir, name + ".pkl")
    with open(out_path, "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved {len(graphs)} graphs → {out_path}")


if __name__ == "__main__":
    main()
