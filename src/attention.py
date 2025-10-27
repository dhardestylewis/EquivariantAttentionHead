"""
Relative-Position Attention from Theorem T4.5

Implements:
    α_ij = (1/√(2m)) q_i^(act)^T (Π_act R_STR(r_j - r_i) Π_act) k_j^(act)

where:
- q_i^(act) = Π_act q_i (T4.2: active subspace projection)
- R_STR(r_j - r_i) from T2.9 (relative-position property)
- Π_act from T2.10 (orthogonal projection onto active subspace)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rotations import RelativePositionRotation


class ActiveSubspaceProjector(nn.Module):
    """
    Implements Π_act from Theorem T2.10:

    Π_act = U diag(I_{2m}, 0) U^T

    where:
    - Π_act^2 = Π_act (idempotent)
    - Π_act^T = Π_act (symmetric)
    - d_act = rank(Π_act) = 2m

    For simplicity, we use a learnable orthogonal projection.
    """

    def __init__(self, d_h: int, d_act: int):
        """
        Args:
            d_h: Full head dimension
            d_act: Active dimension (2m in the theorem)
        """
        super().__init__()
        self.d_h = d_h
        self.d_act = d_act

        # Learnable orthogonal basis for active subspace
        # Initialize as partial identity (first d_act dimensions active)
        U_init = torch.eye(d_h, d_act)
        self.U = nn.Parameter(U_init)  # (d_h, d_act)

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        """
        Project x onto active subspace: I_pi_act x

        Args:
            x: Tensor of shape (..., d_h)
            return_stats: If True, also return diagnostic statistics.

        Returns:
            Projected tensor of shape (..., d_h) if return_stats is False.
            Otherwise returns (projected, stats) where stats contains norms.
        """
        # Orthonormalize U using QR decomposition
        U, _ = torch.linalg.qr(self.U)  # (d_h, d_act)

        # Project: I_pi_act x = U U^T x
        proj_matrix = U @ U.T  # (d_h, d_h)
        x_proj = torch.matmul(x, proj_matrix)  # (..., d_h)

        if not return_stats:
            return x_proj

        # Collect diagnostics about active/null components
        input_norm = x.norm(dim=-1)
        proj_norm = x_proj.norm(dim=-1)
        residual = x - x_proj
        residual_norm = residual.norm(dim=-1)
        eta_mix = residual_norm / (input_norm + 1e-6)

        stats = {
            'input_norm': input_norm.detach(),
            'proj_norm': proj_norm.detach(),
            'residual_norm': residual_norm.detach(),
            'eta_mix': eta_mix.detach(),
        }

        return x_proj, stats

    def get_projection_matrix(self) -> torch.Tensor:
        """Returns Π_act = U U^T ∈ ℝ^{d_h × d_h}"""
        U, _ = torch.linalg.qr(self.U)
        return U @ U.T


class RelativePositionAttention(nn.Module):
    """
    Implements Theorem T4.5 (exact relative-position attention):

    Under hypotheses T1.1 ∧ T2.3 ∧ P_sp ∈ 𝒜:
        α_ij = (1/√(2m)) q_i^(act)^T (Π_act R_STR(r_j - r_i) Π_act) k_j^(act)

    This is the core contribution: attention scores depend ONLY on relative position r_j - r_i.
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_heads: int = 1,
        d_coord: int = 2,
        dropout: float = 0.1,
        use_active_projection: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            d_head: Dimension per attention head
            n_heads: Number of attention heads
            d_coord: Coordinate dimension (2 for images, 3 for 3D)
            dropout: Dropout probability
            use_active_projection: Whether to use Π_act projection (T2.10)
        """
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.d_coord = d_coord
        self.use_active_projection = use_active_projection

        # Query, Key, Value projections (T4.1)
        self.W_Q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_head, bias=False)

        # Output projection
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)

        # Relative position rotation R_STR (T2.7)
        self.rel_rotation = nn.ModuleList([
            RelativePositionRotation(d_head, d_coord) for _ in range(n_heads)
        ])

        # Active subspace projectors Π_act (T2.10)
        if use_active_projection:
            # Use d_act = d_head for simplicity (can be smaller, must be even: d_act = 2m)
            d_act = d_head if d_head % 2 == 0 else d_head - 1
            self.projectors = nn.ModuleList([
                ActiveSubspaceProjector(d_head, d_act) for _ in range(n_heads)
            ])
        else:
            self.projectors = None

        self.dropout = nn.Dropout(dropout)
        self.scale = (d_head ** -0.5) if not use_active_projection else ((d_head // 2) ** -0.5)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor = None,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input of shape (B, N, d_model) where N is sequence length
            positions: Position vectors of shape (N, d_coord) or (B, N, d_coord)
            mask: Optional attention mask of shape (B, N, N)

        Returns:
            output: Tensor of shape (B, N, d_model)
            metrics: Dict with 'commutator_loss' and other diagnostics
        """
        B, N, _ = x.shape

        # Handle position broadcasting
        if positions.dim() == 2:
            positions = positions.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_coord)

        # Linear projections (T4.1)
        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_head)  # (B, N, H, d_h)
        K = self.W_K(x).view(B, N, self.n_heads, self.d_head)  # (B, N, H, d_h)
        V = self.W_V(x).view(B, N, self.n_heads, self.d_head)  # (B, N, H, d_h)

        diagnostics = None

        # Apply active subspace projection if enabled (T4.2)
        if self.projectors is not None:
            q_proj_heads = []
            k_proj_heads = []
            q_stats = []
            k_stats = []
            for h in range(self.n_heads):
                if collect_diagnostics:
                    q_proj, q_head_stats = self.projectors[h](Q[:, :, h, :], return_stats=True)
                    k_proj, k_head_stats = self.projectors[h](K[:, :, h, :], return_stats=True)
                    q_stats.append(q_head_stats)
                    k_stats.append(k_head_stats)
                else:
                    q_proj = self.projectors[h](Q[:, :, h, :])
                    k_proj = self.projectors[h](K[:, :, h, :])
                q_proj_heads.append(q_proj)
                k_proj_heads.append(k_proj)

            Q_proj = torch.stack(q_proj_heads, dim=2)  # (B, N, H, d_h)
            K_proj = torch.stack(k_proj_heads, dim=2)  # (B, N, H, d_h)

            if collect_diagnostics:
                def _stack_stats(key, stats_list):
                    stacked = torch.stack([s[key] for s in stats_list], dim=0)  # (H, B, N)
                    return stacked.permute(1, 2, 0)  # (B, N, H)

                diagnostics = {
                    'q_input_norm': _stack_stats('input_norm', q_stats),
                    'q_proj_norm': _stack_stats('proj_norm', q_stats),
                    'q_residual_norm': _stack_stats('residual_norm', q_stats),
                    'q_eta_mix': _stack_stats('eta_mix', q_stats),
                    'k_input_norm': _stack_stats('input_norm', k_stats),
                    'k_proj_norm': _stack_stats('proj_norm', k_stats),
                    'k_residual_norm': _stack_stats('residual_norm', k_stats),
                    'k_eta_mix': _stack_stats('eta_mix', k_stats),
                }
        else:
            Q_proj, K_proj = Q, K
            if collect_diagnostics:
                q_input_norm = Q.norm(dim=-1).detach()
                k_input_norm = K.norm(dim=-1).detach()
                zeros_q = torch.zeros_like(q_input_norm)
                zeros_k = torch.zeros_like(k_input_norm)
                diagnostics = {
                    'q_input_norm': q_input_norm,
                    'q_proj_norm': q_input_norm,
                    'q_residual_norm': zeros_q,
                    'q_eta_mix': zeros_q,
                    'k_input_norm': k_input_norm,
                    'k_proj_norm': k_input_norm,
                    'k_residual_norm': zeros_k,
                    'k_eta_mix': zeros_k,
                }

        # Compute relative rotations R_STR(r_j - r_i) for each head (T2.9)
        # We compute per-batch-item to handle varying positions
        attn_outputs = []
        total_comm_loss = 0.0

        for b in range(B):
            batch_outputs = []
            for h in range(self.n_heads):
                # Get relative rotations for this head: (N, N, d_head, d_head)
                R_rel = self.rel_rotation[h](positions[b])  # (N, N, d_h, d_h)

                # Apply rotations to keys: k'_j = R_STR(r_j - r_i) k_j
                # Q_proj[b, :, h, :] has shape (N, d_h)
                # K_proj[b, :, h, :] has shape (N, d_h)
                # R_rel has shape (N, N, d_h, d_h)

                # For each query i, rotate all keys j: k'_{i,j} = R_{i,j} @ k_j
                K_rot = torch.einsum('ijhw,jw->ijh', R_rel, K_proj[b, :, h, :])  # (N, N, d_h)

                # Compute attention scores: α_ij = q_i^T k'_{i,j} (T4.4)
                attn_scores = torch.einsum('ih,ijh->ij', Q_proj[b, :, h, :], K_rot)  # (N, N)
                attn_scores = attn_scores * self.scale

                # Apply mask if provided
                if mask is not None:
                    attn_scores = attn_scores.masked_fill(mask[b] == 0, float('-inf'))

                # Softmax
                attn_weights = F.softmax(attn_scores, dim=-1)  # (N, N)
                attn_weights = self.dropout(attn_weights)

                # Apply attention to values
                out = attn_weights @ V[b, :, h, :]  # (N, d_h)
                batch_outputs.append(out)

                # Accumulate commutator loss (T5.1)
                total_comm_loss += self.rel_rotation[h].compute_commutator_loss()

            # Stack heads for this batch item
            batch_out = torch.stack(batch_outputs, dim=1)  # (N, H, d_h)
            attn_outputs.append(batch_out)

        # Stack batches
        attn_output = torch.stack(attn_outputs, dim=0)  # (B, N, H, d_h)

        # Reshape and project output
        attn_output = attn_output.reshape(B, N, self.n_heads * self.d_head)
        output = self.W_O(attn_output)

        # Compute average commutator loss
        avg_comm_loss = total_comm_loss / (B * self.n_heads)

        metrics = {
            'commutator_loss': avg_comm_loss,
        }

        if collect_diagnostics and diagnostics is not None:
            metrics['diagnostics'] = {key: value.detach().cpu() for key, value in diagnostics.items()}

        return output, metrics


class RelativePositionTransformerBlock(nn.Module):
    """
    Transformer block with relative-position attention.
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_heads: int,
        d_ff: int,
        d_coord: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = RelativePositionAttention(
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
            d_coord=d_coord,
            dropout=dropout,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor = None,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, N, d_model)
            positions: (N, d_coord) or (B, N, d_coord)
            mask: (B, N, N) optional

        Returns:
            output: (B, N, d_model)
            metrics: dict
        """
        # Attention with residual
        attn_out, metrics = self.attn(
            self.norm1(x),
            positions,
            mask,
            collect_diagnostics=collect_diagnostics,
        )
        x = x + attn_out

        # Feedforward with residual
        x = x + self.ff(self.norm2(x))

        return x, metrics
