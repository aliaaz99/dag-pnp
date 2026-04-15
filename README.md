# dag-pnp

Score-based diffusion model for directed acyclic graph (DAG) generation, built on top of [GDSS](https://github.com/GDSS/GDSS) with directed graph support and an optional DIRECTO-style dual-attention Transformer backbone.

## Overview

Two score networks are trained jointly via continuous VP-SDE score matching:

- **ScoreNetworkX** — denoises node features `x ∈ ℝ^{B×N×F}`
- **ScoreNetworkA** — denoises the adjacency matrix `adj ∈ ℝ^{B×N×N}`

Both networks are conditioned on each other and on the diffusion time `t`.

## Datasets

| Dataset | Type | Nodes | Edges | Notes |
|---|---|---|---|---|
| `dag_pnp_ER4_d20` | Erdős–Rényi DAG | 20 | ~38 | edge prob p ≈ 0.21 |
| `sem_SF_d20` | Scale-free (BA m=4) DAG + SEM weights | 20 | 36 | heavy-tailed degree dist |

Generate datasets:
```bash
cd gdss
python data/generate_dag_pnp_data.py --graph_type ER --n_nodes 20 --expected_degree 4 --n_graphs 700
python data/generate_sem_data.py --graph_type SF --m_sf 4 --d 20 --n_graphs 700
```

## Models

### 1. GCN Baseline (`sem_sf_d20`)

Standard GDSS backbone with directed GCN score networks.

- **Architecture**: `ScoreNetworkX` + `ScoreNetworkA` with `conv: GCN`
- **Directed normalisation**: asymmetric `D_out^{-1/2} A D_in^{-1/2}` in `DenseGCNConv`
- **Config**: `config/sem_sf_d20.yaml`
- **Checkpoint**: `checkpoints/sem_SF_d20/Apr15-19:25:01.pth`
- **Train adj loss at epoch 5000**: ~44

Train:
```bash
python main.py --type train --config sem_sf_d20
```

Sample:
```bash
python main.py --type sample --config sample_sem_sf_d20_gcn
```

---

### 2. DIRECTO Transformer (`sem_sf_d20_transformer`)

Dual-attention Graph Transformer backbone ported from [DIRECTO](https://github.com/DIRECTO/DIRECTO), wired into the GDSS continuous VP-SDE training framework.

- **Architecture**: `ScoreNetworkX_Transformer` + `ScoreNetworkA_Transformer`
- **Backbone**: `DualNodeEdgeBlock` — separate source→target and target→source attention streams, edge-feature FiLM modulation, sinusoidal time embedding
- **Config**: `config/sem_sf_d20_transformer.yaml`  (`n_layers=5, dx=128, de=64, dy=64, n_head=4`)
- **Checkpoint**: `checkpoints/sem_SF_d20/Apr15-15:58:08.pth`
- **Train adj loss at epoch 5000**: ~40

Train:
```bash
python main.py --type train --config sem_sf_d20_transformer
```

Sample:
```bash
python main.py --type sample --config sample_sem_sf_d20
```

---

## Key fixes vs. vanilla GDSS

| Fix | File | Description |
|---|---|---|
| Asymmetric GCN normalisation | `models/layers.py` | `D_out^{-1/2} A D_in^{-1/2}` for directed graphs |
| `directed` flag propagation | `models/attention.py` | passes `directed` to `DenseGCNConv` constructors |
| Quantize uses `abs()` | `utils/graph_utils.py` | SEM data has signed weights; `abs(adj) < thr` prevents negative edges being discarded |
| Diagonal zeroed during sampling | `solver.py` | zeros `adj` diagonal at every reverse-SDE step; prevents self-loop noise from corrupting attention |
| `takes_time` interface | `losses.py` | Transformer score networks receive diffusion time `t` explicitly |
| Transformer config support | `utils/loader.py`, `utils/logger.py` | detects `Transformer` in model name and loads/logs the right hyperparameters |

## Evaluation

```bash
# open gdss/evaluate_generated_dags.ipynb
# set EXPERIMENTS list to compare runs side-by-side
```

The notebook computes DAG validity, edge count / density distributions, in/out-degree KS statistics, joint degree heatmaps, and visual graph comparison.
