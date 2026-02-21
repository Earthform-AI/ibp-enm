"""Graph Neural Cellular Automata (GNCA) for protein archetype classification.

Phase C of the GNCA infrastructure.  Implements a Graph NCA following
Grattarola et al. 2021 ("Learning Graph Cellular Automata", NeurIPS)
adapted for graph-level classification using Walker et al. 2022
("Physical NCA for 2D Shape Classification") readout strategy.

Architecture
------------
Each node carries a state vector of dimension ``state_dim``:

- Channels ``[0:input_dim]`` — initialised from node features (B-factors,
  Fiedler, hinge scores, etc.)
- Channels ``[input_dim:state_dim-n_classes]`` — hidden channels (init 0)
- Channels ``[-n_classes:]`` — class logit channels (init 0)

At each NCA step, the update rule is **additive** (residual):

    h_i(t) = h_i(t-1) + f_θ(h_i, {(h_j, e_ji) | j ∈ N(i)})

where f_θ is a small message-passing network with shared weights.

After T steps, the class logit channels are mean-pooled across all
nodes to produce graph-level predictions:

    ŷ = softmax( mean_i( h_i[-n_classes:] ) )

Key design choices for our 200-protein, 5-class setting:
- **Tiny model** (~4K params) to avoid massive overfitting
- **Stochastic node updates** (50% dropout per step) — primary regularisation
- **Random T ∈ [T_min, T_max]** per forward pass — prevents step-count coupling
- **Mean aggregation** (not sum) — handles variable graph sizes (100–400 nodes)
- **No normalization layers** — following both reference papers

Requires
--------
``pip install ibp-enm[gnca]`` (torch + torch-geometric)

References
----------
- Grattarola et al. 2021, NeurIPS — Graph NCA (architecture)
- Walker et al. 2022, IEEE — Physical NCA shape classification (readout)
- Mordvintsev et al. 2020 — Growing NCA (stochastic updates)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

__all__ = [
    "GNCAConfig",
    "GNCACell",
    "GNCAClassifier",
]


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GNCAConfig:
    """Hyperparameters for the GNCA classifier.

    Designed for a small-data regime (200 proteins, 5 classes)
    where regularisation is paramount.
    """

    # ── Dimensions ──────────────────────────────────────────────
    input_dim: int = 15          # node feature dimension (from graph_data.py)
    edge_dim: int = 2            # edge feature dimension
    hidden_dim: int = 32         # message-passing hidden width
    state_dim: int = 48          # total per-node state dimension
    n_classes: int = 5           # number of archetypes

    # ── NCA dynamics ────────────────────────────────────────────
    t_min: int = 8               # minimum NCA steps
    t_max: int = 16              # maximum NCA steps
    update_prob: float = 0.5     # stochastic update probability per node per step

    # ── Regularisation ──────────────────────────────────────────
    state_noise: float = 0.01    # Gaussian noise added to states each step
    dropout: float = 0.0         # MLP dropout (on top of stochastic updates)

    # ── Training ────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 0.01
    lr_patience: int = 30        # reduce-on-plateau patience (epochs)
    lr_factor: float = 0.5       # reduce-on-plateau factor
    epochs: int = 300
    early_stop_patience: int = 50
    batch_size: int = 16

    def __post_init__(self):
        assert self.state_dim > self.input_dim + self.n_classes, (
            f"state_dim ({self.state_dim}) must be > "
            f"input_dim ({self.input_dim}) + n_classes ({self.n_classes})"
        )


# ═══════════════════════════════════════════════════════════════════
# NCA Cell — single update step
# ═══════════════════════════════════════════════════════════════════

class GNCACell(MessagePassing):
    """Single Graph NCA transition step.

    Implements:
        m_j  = ReLU( W_msg · [h_j ‖ e_ji] + b_msg )
        m̄_i  = mean_{j ∈ N(i)} m_j
        Δh_i = MLP_post( [h_i ‖ m̄_i] )
        h_i  ← h_i + mask_i · Δh_i

    where mask_i ∈ {0, 1} is sampled per-node with P(1) = update_prob.
    """

    def __init__(self, config: GNCAConfig):
        super().__init__(aggr="mean")  # mean aggregation for variable-size graphs
        sd = config.state_dim
        ed = config.edge_dim
        hd = config.hidden_dim

        # Message function: state + edge_attr → hidden
        self.msg_lin = nn.Linear(sd + ed, hd)

        # Update function: [h_i ‖ aggregated_msg] → Δh
        self.update_mlp = nn.Sequential(
            nn.Linear(sd + hd, hd),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(hd, sd),
        )

        self.update_prob = config.update_prob
        self.state_noise = config.state_noise

        # Initialise final linear layer to near-zero for stable start
        nn.init.zeros_(self.update_mlp[-1].weight)
        nn.init.zeros_(self.update_mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,          # (N_total, state_dim)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:
        """One NCA step with stochastic update masking."""
        # Message passing
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update
        delta = self.update_mlp(torch.cat([x, agg], dim=-1))

        # Stochastic update mask (per-node)
        if self.training and self.update_prob < 1.0:
            mask = (torch.rand(x.shape[0], 1, device=x.device) < self.update_prob).float()
            delta = delta * mask

        # Additive residual update
        x = x + delta

        # State noise (training only)
        if self.training and self.state_noise > 0:
            x = x + torch.randn_like(x) * self.state_noise

        return x

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute messages from neighbours."""
        return F.relu(self.msg_lin(torch.cat([x_j, edge_attr], dim=-1)))


# ═══════════════════════════════════════════════════════════════════
# Full GNCA Classifier
# ═══════════════════════════════════════════════════════════════════

class GNCAClassifier(nn.Module):
    """Graph Neural Cellular Automata for protein archetype classification.

    Architecture summary:
    1. **Encoder**: project input features → state vector (zero-pad hidden & class channels)
    2. **NCA dynamics**: apply GNCACell T times (T random during training)
    3. **Readout**: mean-pool class logit channels → softmax

    Parameters
    ----------
    config : GNCAConfig
        All hyperparameters.
    """

    def __init__(self, config: Optional[GNCAConfig] = None):
        super().__init__()
        self.config = config or GNCAConfig()
        c = self.config

        # Encoder: input features → full state dimension (learnable projection)
        self.encoder = nn.Sequential(
            nn.Linear(c.input_dim, c.state_dim),
            nn.ReLU(inplace=True),
        )

        # Single NCA cell (shared weights across all T steps)
        self.cell = GNCACell(c)

        # Number of classes (last n_classes channels are logits)
        self.n_classes = c.n_classes

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass: encode → run NCA → readout.

        Parameters
        ----------
        data : torch_geometric.data.Data (or Batch)
            Must have ``.x``, ``.edge_index``, ``.edge_attr``, ``.batch``.

        Returns
        -------
        logits : (B, n_classes) tensor
            Unnormalised class logits for each graph in the batch.
        """
        c = self.config

        # ── 1. Encode ──────────────────────────────────────────
        state = self.encoder(data.x)  # (N_total, state_dim)

        # ── 2. NCA dynamics ────────────────────────────────────
        if self.training:
            T = torch.randint(c.t_min, c.t_max + 1, (1,)).item()
        else:
            # Deterministic at eval: use midpoint
            T = (c.t_min + c.t_max) // 2

        for _ in range(T):
            state = self.cell(state, data.edge_index, data.edge_attr)

        # ── 3. Readout ─────────────────────────────────────────
        # Extract class logit channels (last n_classes dims)
        class_channels = state[:, -self.n_classes:]  # (N_total, n_classes)

        # Mean pool per graph
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
            torch.zeros(state.shape[0], dtype=torch.long, device=state.device)

        # Manual scatter mean for clarity
        B = int(batch.max().item()) + 1
        logits = torch.zeros(B, self.n_classes, device=state.device)
        counts = torch.zeros(B, 1, device=state.device)
        logits.scatter_add_(0, batch.unsqueeze(1).expand(-1, self.n_classes), class_channels)
        counts.scatter_add_(0, batch.unsqueeze(1), torch.ones_like(batch, dtype=torch.float).unsqueeze(1))
        logits = logits / counts.clamp(min=1)

        return logits

    def predict(self, data: Data) -> torch.Tensor:
        """Predict class indices."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
        return logits.argmax(dim=-1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Human-readable model summary."""
        c = self.config
        lines = [
            "GNCAClassifier",
            "=" * 40,
            f"Input dim:      {c.input_dim}",
            f"Edge dim:       {c.edge_dim}",
            f"State dim:      {c.state_dim}",
            f"Hidden dim:     {c.hidden_dim}",
            f"Classes:        {c.n_classes}",
            f"NCA steps:      [{c.t_min}, {c.t_max}]",
            f"Update prob:    {c.update_prob}",
            f"State noise:    {c.state_noise}",
            f"Parameters:     {self.count_parameters():,}",
            "",
            "Architecture:",
            f"  Encoder:  Linear({c.input_dim}→{c.state_dim}) + ReLU",
            f"  Cell:     msg_lin({c.state_dim+c.edge_dim}→{c.hidden_dim})",
            f"           update_mlp({c.state_dim+c.hidden_dim}→{c.hidden_dim}→{c.state_dim})",
            f"  Readout:  mean_pool(state[-{c.n_classes}:]) → logits",
        ]
        return "\n".join(lines)
