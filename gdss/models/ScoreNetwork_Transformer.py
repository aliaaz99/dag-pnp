"""
ScoreNetwork_Transformer.py

Drop-in replacements for ScoreNetworkA / ScoreNetworkX that use
DIRECTO's dual-attention Graph Transformer as the denoising backbone,
while keeping GDSS's continuous VP-SDE training unchanged.

Architecture adapted from:
  A. Carballo Castro et al., "DIRECTO: Directed Graph Generation via
  Discrete Flow Matching", ICLR 2026.
  https://github.com/acarballocastro/DIRECTO

GDSS interface (unchanged):
  ScoreNetworkA_Transformer.forward(x, adj, flags, t=None) -> (B, N, N)
  ScoreNetworkX_Transformer.forward(x, adj, flags, t=None) -> (B, N, F)

Time conditioning:
  Both models accept an optional `t` (B,).  When provided, t is mapped
  through a sinusoidal embedding and injected as the global feature
  vector y of the transformer.  losses.py passes t when the model
  has `takes_time = True`.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, LayerNorm, Dropout

from utils.graph_utils import mask_adjs, mask_x


# ---------------------------------------------------------------------------
# Helpers (ported from DIRECTO/src/models/layers.py)
# ---------------------------------------------------------------------------

class Xtoy(nn.Module):
    """Aggregate node features into a global vector."""
    def __init__(self, dx, dy):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        # X: (B, N, dx)
        m   = X.mean(dim=1)
        mi  = X.min(dim=1)[0]
        ma  = X.max(dim=1)[0]
        std = X.std(dim=1)
        return self.lin(torch.cat([m, mi, ma, std], dim=-1))


class Etoy(nn.Module):
    """Aggregate edge features into a global vector."""
    def __init__(self, de, dy):
        super().__init__()
        self.lin = nn.Linear(4 * de, dy)

    def forward(self, E):
        # E: (B, N, N, de)
        m   = E.mean(dim=(1, 2))
        mi  = E.min(dim=2)[0].min(dim=1)[0]
        ma  = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        return self.lin(torch.cat([m, mi, ma, std], dim=-1))


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


# ---------------------------------------------------------------------------
# Dual-attention block (ported from DIRECTO/src/models/transformer_model_directed.py)
# ---------------------------------------------------------------------------

class DualNodeEdgeBlock(nn.Module):
    """
    Self-attention with separate source→target and target→source streams.

    Inputs / outputs:
      X  : (B, N, dx)
      E  : (B, N, N, de)   dense directed edge features
      y  : (B, dy)          global features (time embedding)
      node_mask : (B, N)
    """

    def __init__(self, dx, de, dy, n_head):
        super().__init__()
        assert dx % n_head == 0, f"dx={dx} must be divisible by n_head={n_head}"
        self.dx     = dx
        self.de     = de
        self.dy     = dy
        self.n_head = n_head
        self.df     = dx // n_head

        # Source (row) and target (col) Q/K/V projections
        self.q_s = Linear(dx, dx)
        self.k_s = Linear(dx, dx)
        self.v_s = Linear(dx, dx)
        self.q_t = Linear(dx, dx)
        self.k_t = Linear(dx, dx)
        self.v_t = Linear(dx, dx)

        # FiLM: y → E
        self.y_e_mul = Linear(dy, dx)
        self.y_e_add = Linear(dy, dx)

        # FiLM: y → X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # FiLM: E → X  (both directions)
        self.e_mul_s = Linear(de, dx)
        self.e_add_s = Linear(de, dx)
        self.e_mul_t = Linear(de, dx)
        self.e_add_t = Linear(de, dx)

        # Global feature update
        self.y_y  = Linear(dy, dy)
        self.x_y  = Xtoy(dx, dy)
        self.e_y  = Etoy(de, dy)

        # Output projections
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(
            nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy)
        )

        # Gating for node update
        self.gate_x = Linear(2 * dx, dx)

        # Learned edge-gating factors
        self.edge_gate_linear   = Linear(de, n_head)
        self.edge_gate_linear_t = Linear(de, n_head)

    def forward(self, X, E, y, node_mask):
        bs, n, _ = X.shape
        x_mask  = node_mask.unsqueeze(-1)          # (B, N, 1)
        e_mask1 = x_mask.unsqueeze(2)              # (B, N, 1, 1)  — row mask
        e_mask2 = x_mask.unsqueeze(1)              # (B, 1, N, 1)  — col mask

        X_res, E_res, y_res = X, E, y

        # --- Q / K / V for both streams ---
        Q_s = self.q_s(X) * x_mask                # (B, N, dx)
        K_s = self.k_s(X) * x_mask
        V_s = self.v_s(X) * x_mask                # kept as (B, N, dx)
        Q_t = self.q_t(X) * x_mask
        K_t = self.k_t(X) * x_mask
        V_t = self.v_t(X) * x_mask                # kept as (B, N, dx)

        # Reshape for multi-head → (B, N, n_head, df), then broadcast dim
        Q_s = Q_s.reshape(bs, n, self.n_head, self.df).unsqueeze(2)   # (B, N, 1, H, df)
        K_s = K_s.reshape(bs, n, self.n_head, self.df).unsqueeze(1)   # (B, 1, N, H, df)
        Q_t = Q_t.reshape(bs, n, self.n_head, self.df).unsqueeze(2)
        K_t = K_t.reshape(bs, n, self.n_head, self.df).unsqueeze(1)

        # Attention score tensors  (B, N, N, n_head, df)
        Y_st = Q_s * K_t / math.sqrt(self.df)
        Y_ts = Q_t * K_s / math.sqrt(self.df)

        # --- Modulate with directed edge features ---
        E_T = E.permute(0, 2, 1, 3).contiguous()  # (B, N, N, de) — transpose

        # Project E → (B, N, N, dx) → reshape to (B, N, N, n_head, df)
        e1_4d = e_mask1 * e_mask2                  # (B, N, N, 1) shared 4-D mask
        E1_s = (self.e_mul_s(E  ) * e1_4d).reshape(bs, n, n, self.n_head, self.df)
        E1_t = (self.e_mul_t(E_T) * e1_4d).reshape(bs, n, n, self.n_head, self.df)
        E2_s = (self.e_add_s(E  ) * e1_4d).reshape(bs, n, n, self.n_head, self.df)
        E2_t = (self.e_add_t(E_T) * e1_4d).reshape(bs, n, n, self.n_head, self.df)

        # Learned gating  (B, N, N, n_head) → unsqueeze to (B, N, N, n_head, 1)
        g   = torch.sigmoid(self.edge_gate_linear  (E  )).unsqueeze(-1)
        g_t = torch.sigmoid(self.edge_gate_linear_t(E_T)).unsqueeze(-1)

        Y_st = Y_st * (g   * (E1_s + 1)) + E2_s
        Y_ts = Y_ts * (g_t * (E1_t + 1)) + E2_t

        # --- Edge feature update (→ new E) ---
        newE = Y_st.flatten(start_dim=3)                         # (B, N, N, dx)
        ye1  = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        ye2  = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE
        newE = self.e_out(newE) * e1_4d                          # (B, N, N, de)
        newE = (E_res + newE) * e1_4d

        # --- Attention-weighted node aggregation ---
        # mask over key positions: (B, 1, N, n_head) → expand to (B, N, N, n_head)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # (B, N, N, n_head)
        attn     = torch.cat([Y_st, Y_ts], dim=2)                # (B, N, 2N, n_head, df)
        attn_msk = torch.cat([softmax_mask, softmax_mask], dim=2) # (B, N, 2N, n_head)
        # masked_softmax: mask (B,N,2N,n_head) broadcasts to (B,N,2N,n_head,df)
        attn = masked_softmax(attn, attn_msk.unsqueeze(-1).expand_as(attn), dim=2)

        # V_t, V_s: (B, N, dx) — concatenate along node dim → (B, 2N, dx)
        weighted_V = torch.cat([V_t, V_s], dim=1)
        # attn: (B, N, 2N, n_head, df); weighted_V: (B, 2N, dx=n_head*df)
        # reshape weighted_V to (B, 1, 2N, n_head, df) then multiply and sum
        weighted_V_5d = weighted_V.reshape(bs, 1, 2 * n, self.n_head, self.df)
        weighted_V = (attn * weighted_V_5d).sum(dim=2).flatten(start_dim=2)  # (B, N, dx)

        # FiLM from y → X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        modulated = yx1 + (yx2 + 1) * weighted_V
        gate  = torch.sigmoid(self.gate_x(torch.cat([X, weighted_V], dim=-1)))
        newX  = gate * X_res + (1 - gate) * modulated
        newX  = self.x_out(newX) * x_mask

        # --- Global update ---
        new_y = self.y_y(y) + self.x_y(X) + self.e_y(E)
        new_y = y_res + self.y_out(new_y)

        return newX, newE, new_y


# ---------------------------------------------------------------------------
# Transformer layer: attention + FFN + layer-norm
# ---------------------------------------------------------------------------

class XEyTransformerLayer(nn.Module):
    """One full transformer block (attention + feed-forward + norms)."""

    def __init__(self, dx, de, dy, n_head,
                 dim_ffX=256, dim_ffE=128, dim_ffy=128,
                 dropout=0.0):
        super().__init__()
        self.attn   = DualNodeEdgeBlock(dx, de, dy, n_head)

        # Node FFN
        self.linX1    = Linear(dx, dim_ffX)
        self.linX2    = Linear(dim_ffX, dx)
        self.normX1   = LayerNorm(dx)
        self.normX2   = LayerNorm(dx)
        self.dropX1   = Dropout(dropout)
        self.dropX2   = Dropout(dropout)
        self.dropX3   = Dropout(dropout)

        # Edge FFN
        self.linE1    = Linear(de, dim_ffE)
        self.linE2    = Linear(dim_ffE, de)
        self.normE1   = LayerNorm(de)
        self.normE2   = LayerNorm(de)
        self.dropE1   = Dropout(dropout)
        self.dropE2   = Dropout(dropout)
        self.dropE3   = Dropout(dropout)

        # Global FFN
        self.lin_y1   = Linear(dy, dim_ffy)
        self.lin_y2   = Linear(dim_ffy, dy)
        self.norm_y1  = LayerNorm(dy)
        self.norm_y2  = LayerNorm(dy)
        self.drop_y1  = Dropout(dropout)
        self.drop_y2  = Dropout(dropout)
        self.drop_y3  = Dropout(dropout)

        self.act = F.relu

    def forward(self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor):
        newX, newE, new_y = self.attn(X, E, y, node_mask)

        X = self.normX1(X + self.dropX1(newX))
        E = self.normE1(E + self.dropE1(newE))
        y = self.norm_y1(y + self.drop_y1(new_y))

        X = self.normX2(X + self.dropX3(self.linX2(self.dropX2(self.act(self.linX1(X))))))
        E = self.normE2(E + self.dropE3(self.linE2(self.dropE2(self.act(self.linE1(E))))))
        y = self.norm_y2(y + self.drop_y3(self.lin_y2(self.drop_y2(self.act(self.lin_y1(y))))))

        return X, E, y


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """t ∈ [0,1] → dense vector of dimension `dim`."""

    def __init__(self, dim):
        super().__init__()
        self.dim  = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, t):
        # t: (B,)
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args  = t[:, None] * freqs[None] * 1000.0
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Full directed graph transformer (adapted from DIRECTO)
# ---------------------------------------------------------------------------

class DirectedGraphTransformer(nn.Module):
    """
    Input:
      X  : (B, N, in_X)   noisy node features
      E  : (B, N, N, in_E) noisy edge features
      y  : (B, dy)         time embedding (or zeros)
      node_mask : (B, N)

    Output:
      X_out : (B, N, out_X)
      E_out : (B, N, N, out_E)
    """

    def __init__(self, n_layers,
                 in_X, in_E, dy,
                 out_X, out_E,
                 dx, de,
                 n_head,
                 dim_ffX=256, dim_ffE=128, dim_ffy=128,
                 dropout=0.0):
        super().__init__()
        assert dx % n_head == 0

        # Input projections
        self.mlp_in_X = nn.Sequential(
            nn.Linear(in_X, 2 * dx), nn.ReLU(),
            nn.Linear(2 * dx, dx),  nn.ReLU(),
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(in_E, 2 * de), nn.ReLU(),
            nn.Linear(2 * de, de),   nn.ReLU(),
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(dy, 2 * dy), nn.ReLU(),
            nn.Linear(2 * dy, dy), nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            XEyTransformerLayer(dx, de, dy, n_head, dim_ffX, dim_ffE, dim_ffy, dropout)
            for _ in range(n_layers)
        ])

        # Output projections
        self.mlp_out_X = nn.Sequential(
            nn.Linear(dx, 2 * dx), nn.ReLU(),
            nn.Linear(2 * dx, out_X),
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(de, 2 * de), nn.ReLU(),
            nn.Linear(2 * de, out_E),
        )

        self.out_X = out_X
        self.out_E = out_E

    def forward(self, X, E, y, node_mask):
        """
        node_mask: (B, N) float tensor (0 or 1)
        """
        bs, n = X.shape[:2]

        # Keep skip connections to input scale
        X_skip = X[..., :self.out_X]
        E_skip = E[..., :self.out_E]

        # Project to hidden dims
        X = self.mlp_in_X(X)
        E = self.mlp_in_E(E)
        y = self.mlp_in_y(y)

        # Apply node/edge masks after initial projection
        x_mask  = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)
        X = X * x_mask
        E = E * e_mask1 * e_mask2

        # Transformer layers
        for layer in self.layers:
            X, E, y = layer(X, E, y, node_mask)

        # Output projections + skip
        X_out = self.mlp_out_X(X) + X_skip
        E_out = self.mlp_out_E(E) + E_skip

        # Zero diagonal (no self-loops) and re-mask
        diag = torch.eye(n, device=E.device).bool().unsqueeze(0).unsqueeze(-1)
        E_out = E_out.masked_fill(diag, 0.0)
        X_out = X_out * x_mask
        E_out = E_out * e_mask1 * e_mask2

        return X_out, E_out


# ---------------------------------------------------------------------------
# GDSS-compatible score networks
# ---------------------------------------------------------------------------

class ScoreNetworkA_Transformer(nn.Module):
    """
    Directed adj score network using dual-attention Graph Transformer.

    GDSS interface: forward(x, adj, flags, t=None) -> (B, N, N)
    """
    takes_time = True

    def __init__(self, max_feat_num, max_node_num,
                 n_layers=5, dx=128, de=64, dy=64,
                 n_head=4, dim_ffX=256, dim_ffE=128,
                 dropout=0.0, **kwargs):
        super().__init__()
        self.max_node_num = max_node_num
        self.dy           = dy

        self.time_embed = SinusoidalTimeEmbedding(dy)

        self.transformer = DirectedGraphTransformer(
            n_layers=n_layers,
            in_X=max_feat_num,   # noisy node features
            in_E=1,              # noisy adj value (scalar per edge slot)
            dy=dy,               # time embedding dim
            out_X=max_feat_num,  # unused output
            out_E=1,             # adj score (scalar per edge slot)
            dx=dx, de=de,
            n_head=n_head,
            dim_ffX=dim_ffX, dim_ffE=dim_ffE, dim_ffy=dy,
            dropout=dropout,
        )

    def forward(self, x, adj, flags, t=None):
        """
        x     : (B, N, F)
        adj   : (B, N, N)   noisy adjacency
        flags : (B, N)      node mask (float 0/1)
        t     : (B,)        time ∈ [eps, T]   (optional)
        Returns (B, N, N) adj score.
        """
        B, N = adj.shape[:2]

        # Time embedding → global y
        if t is not None:
            y = self.time_embed(t)               # (B, dy)
        else:
            y = torch.zeros(B, self.dy, device=adj.device)

        E = adj.unsqueeze(-1)                    # (B, N, N, 1)

        _, E_out = self.transformer(x, E, y, flags)  # E_out: (B, N, N, 1)

        score = E_out.squeeze(-1)                # (B, N, N)
        score = mask_adjs(score, flags)
        return score


class ScoreNetworkX_Transformer(nn.Module):
    """
    Node-feature score network using dual-attention Graph Transformer.

    GDSS interface: forward(x, adj, flags, t=None) -> (B, N, F)
    """
    takes_time = True

    def __init__(self, max_feat_num, depth=5, nhid=64,
                 dx=128, de=64, dy=64,
                 n_head=4, dim_ffX=256, dim_ffE=128,
                 dropout=0.0, **kwargs):
        super().__init__()
        self.dy = dy

        self.time_embed = SinusoidalTimeEmbedding(dy)

        self.transformer = DirectedGraphTransformer(
            n_layers=depth,
            in_X=max_feat_num,
            in_E=1,
            dy=dy,
            out_X=max_feat_num,  # node score output
            out_E=1,             # unused output
            dx=dx, de=de,
            n_head=n_head,
            dim_ffX=dim_ffX, dim_ffE=dim_ffE, dim_ffy=dy,
            dropout=dropout,
        )

    def forward(self, x, adj, flags, t=None):
        """
        x     : (B, N, F)   noisy node features
        adj   : (B, N, N)   noisy adjacency (context)
        flags : (B, N)
        t     : (B,)        time (optional)
        Returns (B, N, F) node score.
        """
        B, N = x.shape[:2]

        if t is not None:
            y = self.time_embed(t)
        else:
            y = torch.zeros(B, self.dy, device=x.device)

        E = adj.unsqueeze(-1)                    # (B, N, N, 1)

        X_out, _ = self.transformer(x, E, y, flags)   # X_out: (B, N, F)

        score = mask_x(X_out, flags)
        return score
