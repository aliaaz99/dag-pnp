"""
Generate linear SEM graph datasets for GDSS training.

Data model: X = XW + E, E ~ N(0,I)
  - W is a weighted DAG adjacency matrix (W[i,j] != 0 means edge i -> j)
  - Non-zero weights drawn from [-2,-0.5] ∪ [0.5, 2] uniformly
  - ER: Erdős-Rényi DAGs with edge probability p=0.3
  - SF: Scale-free DAGs via Barabási-Albert preferential attachment

Output: a pickle file containing a list of nx.DiGraph objects with weighted
        edges, ready to be loaded by GDSS's data_loader.py pipeline.

Usage:
    python data/generate_sem_data.py --graph_type ER --d 20 --n_graphs 700
    python data/generate_sem_data.py --graph_type SF --d 20 --n_graphs 700
"""

import os
import pickle
import argparse

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# DAG structure generators
# ---------------------------------------------------------------------------

def simulate_dag_er(d: int, p: float = 0.3) -> np.ndarray:
    """
    Sample an Erdős-Rényi DAG on d nodes.

    Strategy: draw a lower-triangular binary matrix where each entry below
    the diagonal is 1 with probability p (guarantees acyclicity in the
    original ordering), then apply a uniformly random permutation of nodes
    to break the structured ordering.

    Returns B (d x d) with B[i,j]=1 meaning directed edge i -> j.
    """
    # Lower-triangular Bernoulli (includes only strict lower triangle)
    B = (np.random.random((d, d)) < p).astype(float)
    B = np.tril(B, -1)          # zeros on diagonal and upper triangle
    # Random permutation: relabels nodes, preserving acyclicity
    perm = np.random.permutation(d)
    B = B[np.ix_(perm, perm)]
    return B


def simulate_dag_sf(d: int, m: int = 4) -> np.ndarray:
    """
    Sample a scale-free DAG on d nodes via Barabási-Albert attachment.

    The BA graph is built by adding nodes one at a time; we orient every
    undirected edge from the lower-index node to the higher-index one
    (respecting the attachment order, which naturally yields a DAG), then
    apply a random permutation of node labels.

    Args:
        d: number of nodes
        m: number of edges each new node attaches to (controls density)

    Returns B (d x d) with B[i,j]=1 meaning directed edge i -> j.
    """
    G = nx.barabasi_albert_graph(d, m)
    B = np.zeros((d, d), dtype=float)
    for u, v in G.edges():
        # Orient from smaller to larger attachment index
        src, dst = (u, v) if u < v else (v, u)
        B[src, dst] = 1.0
    # Random permutation
    perm = np.random.permutation(d)
    B = B[np.ix_(perm, perm)]
    return B


# ---------------------------------------------------------------------------
# Edge weight sampler
# ---------------------------------------------------------------------------

def assign_weights(B: np.ndarray) -> np.ndarray:
    """
    Replace binary entries of B with weights from [-2,-0.5] ∪ [0.5, 2].

    Each non-zero entry gets a magnitude drawn uniformly from [0.5, 2]
    and a sign drawn uniformly from {-1, +1}.
    """
    W = np.zeros_like(B, dtype=np.float64)
    mask = B != 0
    n_edges = int(mask.sum())
    if n_edges == 0:
        return W
    magnitudes = np.random.uniform(0.5, 2.0, size=n_edges)
    signs = np.random.choice([-1.0, 1.0], size=n_edges)
    W[mask] = signs * magnitudes
    return W


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_sem_dataset(
    n_graphs: int = 700,
    d: int = 20,
    graph_type: str = "ER",
    m_sf: int = 4,
    p_er: float = 0.3,
    seed: int = 42,
) -> list:
    """
    Generate a list of nx.DiGraph objects representing weighted SEM DAGs.

    Args:
        n_graphs:   number of graphs to generate
        d:          number of nodes per graph
        graph_type: "ER" or "SF"
        m_sf:       BA attachment parameter (SF only)
        p_er:       edge probability (ER only)
        seed:       global random seed for reproducibility

    Returns:
        list of nx.DiGraph, each with float 'weight' attributes on edges.
    """
    rng = np.random.RandomState(seed)
    graphs = []

    for i in range(n_graphs):
        # Per-graph seed for reproducibility
        np.random.seed(int(rng.randint(0, 2**31 - 1)))

        if graph_type == "ER":
            B = simulate_dag_er(d, p=p_er)
        elif graph_type == "SF":
            B = simulate_dag_sf(d, m=m_sf)
        else:
            raise ValueError(f"Unknown graph_type '{graph_type}'. Choose 'ER' or 'SF'.")

        W = assign_weights(B)

        # Build DiGraph; nx.from_numpy_array creates edges with 'weight' attr
        G = nx.from_numpy_array(W, create_using=nx.DiGraph)
        # Paranoia: remove any accidental self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        graphs.append(G)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_graphs} graphs")

    return graphs


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def print_dataset_stats(graphs: list, graph_type: str, d: int) -> None:
    edge_counts = [g.number_of_edges() for g in graphs]
    weights = []
    for g in graphs:
        for _, _, data in g.edges(data=True):
            weights.append(data.get("weight", 0.0))
    weights = np.array(weights)

    print(f"\n=== Dataset statistics ===")
    print(f"  Graph type : {graph_type}")
    print(f"  Nodes (d)  : {d}")
    print(f"  # graphs   : {len(graphs)}")
    print(f"  Edges/graph: {np.mean(edge_counts):.1f} ± {np.std(edge_counts):.1f}  "
          f"(min={min(edge_counts)}, max={max(edge_counts)})")
    print(f"  Avg density: {np.mean(edge_counts) / (d * (d - 1)):.3f}  "
          f"(max possible directed edges = d*(d-1) = {d*(d-1)})")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"  |weight| ≥ 0.5 fraction: {(np.abs(weights) >= 0.5).mean():.3f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate linear SEM graph datasets for GDSS")
    parser.add_argument("--n_graphs", type=int, default=700,
                        help="Number of graphs to generate (default: 700)")
    parser.add_argument("--d", type=int, default=20, choices=[10, 20, 30],
                        help="Number of nodes per graph (default: 20)")
    parser.add_argument("--graph_type", type=str, default="ER", choices=["ER", "SF"],
                        help="Graph family: Erdős-Rényi (ER) or Scale-Free (SF) (default: ER)")
    parser.add_argument("--p_er", type=float, default=0.3,
                        help="Edge probability for ER graphs (default: 0.3)")
    parser.add_argument("--m_sf", type=int, default=4,
                        help="BA attachment parameter for SF graphs (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Output directory (default: data/)")
    parser.add_argument("--name", type=str, default=None,
                        help="Override output filename stem (default: sem_<TYPE>_d<D>)")
    args = parser.parse_args()

    print(f"Generating {args.n_graphs} {args.graph_type} SEM DAGs with d={args.d} nodes ...")
    graphs = generate_sem_dataset(
        n_graphs=args.n_graphs,
        d=args.d,
        graph_type=args.graph_type,
        m_sf=args.m_sf,
        p_er=args.p_er,
        seed=args.seed,
    )

    print_dataset_stats(graphs, args.graph_type, args.d)

    save_name = args.name if args.name else f"sem_{args.graph_type}_d{args.d}"
    os.makedirs(args.data_dir, exist_ok=True)
    file_path = os.path.join(args.data_dir, save_name + ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved to {file_path}")


if __name__ == "__main__":
    main()
