"""
Relative-Position Attention (Theorem T4.5) + perf-oriented refactor

Implements attention scores of the form

    α_ij = (1/sqrt(2m)) q_i^(act)^T [ Π_act R_STR(r_j - r_i) Π_act ] k_j^(act)

where:
  - q_i^(act) = Π_act q_i   (active-subspace projection, T4.2 / T2.10)
  - R_STR(r_j - r_i)        (structured relative rotation, T2.9)
  - Π_act                   (orthogonal projector onto active subspace)
  - commutator_loss         (T5.1) regularizes generators to commute

This module is the "relative-position attention head from T4.5 with
commutator penalty T5.1", per the repository layout description. :contentReference[oaicite:1]{index=1}

Major changes vs. the naive version:
  1. ActiveSubspaceProjector now applies Π_act via (x @ U) @ U^T
     instead of forming a dense Π_act and doing x @ Π_act. This
     avoids an O(d_h^2) multiply and drops cost to O(d_h * d_act).

  2. RelativePositionAttention supports *local window attention*
     via window_radius. Instead of attending over all N tokens,
     each token attends only to a (2r+1)x(2r+1) neighborhood on
     an HxW grid (you pass grid_hw=(H,W) to forward).

     - This reduces the effective attention neighborhood from N
       to ~((2r+1)^2).
     - We build & cache the neighbor index tensor once per (H,W,r).
     - We still call rel_rotation[h](...) which currently returns
       the full (N,N,...) tensor. Once RelativePositionRotation
       can return *just* those local offsets, you get the actual
       ~16x to 100x wall-clock speedup we discussed.

  3. We compute commutator_loss once per head and average, instead
     of redundantly inside the (batch, head) loop.

  4. forward() / block.forward() now accept grid_hw=(H,W) so we
     know how to build neighborhoods without guessing.
"""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotations import RelativePositionRotation


# ---------------------------------------------------------------------
# ActiveSubspaceProjector
# ---------------------------------------------------------------------

class ActiveSubspaceProjector(nn.Module):
    """
    Implements Π_act from Theorem T2.10.

    Π_act = U diag(I_{2m}, 0) U^T
    with:
      - Π_act^2 = Π_act  (idempotent)
      - Π_act^T = Π_act  (symmetric)
      - rank(Π_act) = d_act = 2m

    We store a learnable basis U ∈ ℝ^{d_h×d_act} and QR-orthonormalize
    it each forward pass so columns(U) is an orthonormal active subspace.

    Perf note:
    Instead of explicitly forming Π_act = U U^T and multiplying x @ Π_act
    (which is O(d_h^2)), we do:
        x_act  = x @ U        # (..., d_act)
        x_proj = x_act @ U^T  # (..., d_h)
    which is O(d_h * d_act).
    """

    def __init__(self, d_h: int, d_act: int):
        """
        Args:
            d_h:   full head dimension
            d_act: active dimension (rank of Π_act = 2m in the theory)
                   must be even.
        """
        super().__init__()
        self.d_h = d_h
        self.d_act = d_act

        # Initialize U to first d_act columns of I (simple, stable start).
        U_init = torch.eye(d_h, d_act)
        self.U = nn.Parameter(U_init)  # (d_h, d_act)

    def _orthonormal_basis(self) -> torch.Tensor:
        """
        Orthonormalize columns of U with QR. Returns U_ortho with shape
        (d_h, d_act) and U_ortho^T U_ortho = I_{d_act}.
        """
        U_ortho, _ = torch.linalg.qr(self.U)  # differentiable in PyTorch
        return U_ortho

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ):
        """
        Project x onto the active subspace.

        Args:
            x: tensor (..., d_h)
            return_stats: if True, also return diagnostic norms.

        Returns:
            x_proj (..., d_h)      if return_stats=False
            (x_proj, stats_dict)   if return_stats=True
        """
        U_ortho = self._orthonormal_basis()           # (d_h, d_act)
        x_act   = torch.matmul(x, U_ortho)            # (..., d_act)
        x_proj  = torch.matmul(x_act, U_ortho.T)      # (..., d_h)

        if not return_stats:
            return x_proj

        # Diagnostics for active vs null leakage
        input_norm    = x.norm(dim=-1)
        proj_norm     = x_proj.norm(dim=-1)
        residual      = x - x_proj
        residual_norm = residual.norm(dim=-1)
        eta_mix       = residual_norm / (input_norm + 1e-6)

        stats = {
            "input_norm":     input_norm.detach(),
            "proj_norm":      proj_norm.detach(),
            "residual_norm":  residual_norm.detach(),
            "eta_mix":        eta_mix.detach(),
        }
        return x_proj, stats

    def get_projection_matrix(self) -> torch.Tensor:
        """
        Returns Π_act = U U^T as an explicit (d_h × d_h) matrix.
        Mostly for analysis / probes, not for the hot path.
        """
        U_ortho = self._orthonormal_basis()
        return U_ortho @ U_ortho.T


# ---------------------------------------------------------------------
# RelativePositionAttention
# ---------------------------------------------------------------------

class RelativePositionAttention(nn.Module):
    """
    Relative-position attention head following Theorem T4.5:

        α_ij = (1/sqrt(2m)) q_i^(act)^T (Π_act R_STR(r_j - r_i) Π_act) k_j^(act)

    Under the hypotheses from the proofs (skew-symmetric generators,
    commuting constraints, etc.), α_ij depends *only* on r_j - r_i,
    i.e. it's strictly relative-position, not absolute. :contentReference[oaicite:2]{index=2}

    This implementation adds:
      - Optional active subspace projection Π_act on q,k.
      - Optional local/windowed attention using a (2r+1)x(2r+1) grid
        neighborhood, with caching of neighbor indices.
      - Commutator loss reporting.

    Notes on perf:
      - With window_radius=None we recover full global N×N attention.
      - With window_radius=r and grid_hw=(H,W), each token i only
        attends to tokens j in its (2r+1)x(2r+1) patch. This cuts
        O(N^2) → O(N * (2r+1)^2) work at the attention layer level.
      - Right now we *still* call RelativePositionRotation(...) to
        build the full (N,N,d_h,d_h) rotation tensor and then slice.
        To get the real speedup, update RelativePositionRotation so
        it can directly return only the needed (i,j) pairs.
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_heads: int = 1,
        d_coord: int = 2,
        dropout: float = 0.1,
        use_active_projection: bool = True,
        window_radius: Optional[int] = None,
        cache_neighbors: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.d_coord = d_coord
        self.use_active_projection = use_active_projection

        # Optional locality
        self.window_radius = window_radius  # None => global full attention
        self.cache_neighbors = cache_neighbors

        # Q/K/V projections and output projection (T4.1)
        self.W_Q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)

        # Structured relative rotations per head (T2.7 / T2.9)
        self.rel_rotation = nn.ModuleList([
            RelativePositionRotation(d_head, d_coord)
            for _ in range(n_heads)
        ])

        # Active subspace projectors Π_act (T2.10)
        if use_active_projection:
            # Force even rank (2m). If d_head is odd, drop 1 dim.
            d_act = d_head if (d_head % 2 == 0) else d_head - 1
            self.d_act = d_act
            self.projectors = nn.ModuleList([
                ActiveSubspaceProjector(d_head, d_act)
                for _ in range(n_heads)
            ])
            # Theorem uses 1/sqrt(2m); rank(Π_act)=2m=d_act.
            self.scale = (float(d_act) ** -0.5)
        else:
            self.projectors = None
            self.d_act = d_head
            self.scale = (float(d_head) ** -0.5)

        self.dropout = nn.Dropout(dropout)

        # Cache for neighbor index / mask keyed by (H, W, radius)
        # We store CPU copies and move to device on demand to avoid
        # growing GPU memory with many cached grids.
        self._neighbor_cache: Dict[
            Tuple[int, int, int],
            Tuple[torch.Tensor, torch.Tensor]
        ] = {}

    # -----------------------------
    # Neighborhood indexing helpers
    # -----------------------------

    def _get_neighbor_index(
        self,
        grid_hw: Tuple[int, int],
        radius: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build (and cache) the table of local neighbors for each token
        index on an HxW grid.

        Returns:
            neighbor_index: (N, Wsize) long tensor
                neighbor_index[i, w] = flat index of w-th neighbor of token i
            neighbor_mask:  (N, Wsize) bool tensor
                True where valid, False for padded slots
        """
        H, W = grid_hw
        key = (H, W, radius)

        if self.cache_neighbors and key in self._neighbor_cache:
            idx_cpu, mask_cpu = self._neighbor_cache[key]
            return idx_cpu.to(device), mask_cpu.to(device)

        N = H * W
        max_win = (2 * radius + 1) ** 2

        neighbor_index = torch.zeros((N, max_win),
                                     dtype=torch.long,
                                     device=device)
        neighbor_mask = torch.zeros((N, max_win),
                                    dtype=torch.bool,
                                    device=device)

        # For each flat index, enumerate its local window.
        for flat_idx in range(N):
            r = flat_idx // W
            c = flat_idx % W
            w_ptr = 0

            for dr in range(-radius, radius + 1):
                rr = r + dr
                if rr < 0 or rr >= H:
                    continue
                for dc in range(-radius, radius + 1):
                    cc = c + dc
                    if cc < 0 or cc >= W:
                        continue

                    neighbor_index[flat_idx, w_ptr] = rr * W + cc
                    neighbor_mask[flat_idx, w_ptr] = True
                    w_ptr += 1

        if self.cache_neighbors:
            # Store CPU copies to avoid leaking GPU RAM across calls
            self._neighbor_cache[key] = (
                neighbor_index.cpu(),
                neighbor_mask.cpu(),
            )

        return neighbor_index, neighbor_mask

    # -----------------------------
    # Attention kernels
    # -----------------------------

    def _attention_full(
        self,
        Q_bh: torch.Tensor,      # (N, d_head)
        K_bh: torch.Tensor,      # (N, d_head)
        V_bh: torch.Tensor,      # (N, d_head)
        R_rel: torch.Tensor,     # (N, N, d_head, d_head)
        mask_b: Optional[torch.Tensor],  # (N, N) or None
    ) -> torch.Tensor:
        """
        Vanilla global attention using all N keys/values.
        """
        # Rotate each key vector k_j into i's frame:
        # k'_{i,j} = R_rel[i,j] @ k_j
        K_rot = torch.einsum("ijhd,jd->ijh", R_rel, K_bh)  # (N, N, d_head)

        # Score: α_ij = q_i^T k'_{i,j}
        attn_scores = torch.einsum("id,ijh->ij", Q_bh, K_rot)  # (N, N)
        attn_scores = attn_scores * self.scale

        if mask_b is not None:
            attn_scores = attn_scores.masked_fill(mask_b == 0, float("-inf"))

        # Softmax over j
        attn_weights = F.softmax(attn_scores, dim=-1)  # (N, N)
        attn_weights = self.dropout(attn_weights)

        # Aggregate values
        out = attn_weights @ V_bh  # (N, d_head)
        return out

    def _attention_local(
        self,
        Q_bh: torch.Tensor,              # (N, d_head)
        K_bh: torch.Tensor,              # (N, d_head)
        V_bh: torch.Tensor,              # (N, d_head)
        R_rel: torch.Tensor,             # (N, N, d_head, d_head)
        mask_b: Optional[torch.Tensor],  # (N, N) or None
        neighbor_index: torch.Tensor,    # (N, Wsize)
        neighbor_mask: torch.Tensor,     # (N, Wsize) bool
    ) -> torch.Tensor:
        """
        Local/windowed attention. Each token i only attends to the
        precomputed neighbor_index[i, :] set.
        """
        N, Wsize = neighbor_index.shape
        device = Q_bh.device
        arangeN = torch.arange(N, device=device).unsqueeze(1).expand(N, Wsize)

        # Slice rotations and keys for just the local neighbors.
        # R_local[i, w, :, :] = R_rel[i, neighbor_index[i,w], :, :]
        R_local = R_rel[arangeN, neighbor_index]    # (N, Wsize, d_head, d_head)
        K_local = K_bh[neighbor_index]              # (N, Wsize, d_head)

        # Rotate keys in-place for each (i, w)
        # k_rot_local[i, w, :] = R_local[i,w] @ K_local[i,w]
        K_rot_local = torch.einsum("nwdh,nwh->nwd", R_local, K_local)
        # (N, Wsize, d_head)

        # Score each neighbor window:
        # score[i, w] = q_i^T k_rot_local[i,w]
        attn_scores_local = torch.einsum(
            "nd,nwd->nw",
            Q_bh,
            K_rot_local,
        )  # (N, Wsize)
        attn_scores_local = attn_scores_local * self.scale

        # Combine:
        #  - neighbor_mask (False where padded slots exist)
        #  - optional caller-provided mask (B, N, N)
        if mask_b is not None:
            ext_mask = mask_b[arangeN, neighbor_index]  # (N, Wsize)
            combined_invalid = (~neighbor_mask) | (ext_mask == 0)
        else:
            combined_invalid = (~neighbor_mask)

        attn_scores_local = attn_scores_local.masked_fill(
            combined_invalid,
            float("-inf"),
        )

        # Softmax over the window Wsize
        attn_weights_local = F.softmax(attn_scores_local, dim=-1)  # (N, Wsize)
        attn_weights_local = self.dropout(attn_weights_local)

        # Weighted sum of values V_bh[j] with j in neighbor_index[i, :]
        V_local = V_bh[neighbor_index]  # (N, Wsize, d_head)
        out_local = torch.einsum("nw,nwd->nd", attn_weights_local, V_local)
        # (N, d_head)

        return out_local

    # -----------------------------
    # Forward
    # -----------------------------

    def forward(
        self,
        x: torch.Tensor,                       # (B, N, d_model)
        positions: torch.Tensor,               # (N, d_coord) or (B, N, d_coord)
        mask: Optional[torch.Tensor] = None,   # (B, N, N) or None
        collect_diagnostics: bool = False,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x:        (B, N, d_model)
            positions:
                      (N, d_coord) or (B, N, d_coord) absolute coords per token.
                      These feed RelativePositionRotation to get R_STR(r_j-r_i).
            mask:     optional (B, N, N) boolean/binary mask
            collect_diagnostics:
                      if True, return projector norms / eta_mix, etc.
            grid_hw:  (H, W) if you want windowed attention.
                      Required for local mode.

        Returns:
            output:  (B, N, d_model)
            metrics: dict with:
                     - 'commutator_loss' (scalar)
                     - 'diagnostics' (optional dict of tensors)
        """
        B, N, _ = x.shape

        # Broadcast positions if provided as (N, d_coord)
        if positions.dim() == 2:
            positions = positions.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_coord)

        # Linear projections to per-head q/k/v (T4.1)
        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_head)  # (B, N, H, d_h)
        K = self.W_K(x).view(B, N, self.n_heads, self.d_head)  # (B, N, H, d_h)
        V = self.W_V(x).view(B, N, self.n_heads, self.d_head)  # (B, N, H, d_h)

        diagnostics = None

        # Apply Π_act to q and k if enabled, plus optional diagnostics
        if self.projectors is not None:
            q_proj_heads = []
            k_proj_heads = []
            q_stats = []
            k_stats = []

            for h in range(self.n_heads):
                if collect_diagnostics:
                    q_proj_h, q_head_stats = self.projectors[h](
                        Q[:, :, h, :],
                        return_stats=True,
                    )
                    k_proj_h, k_head_stats = self.projectors[h](
                        K[:, :, h, :],
                        return_stats=True,
                    )
                    q_stats.append(q_head_stats)
                    k_stats.append(k_head_stats)
                else:
                    q_proj_h = self.projectors[h](Q[:, :, h, :])
                    k_proj_h = self.projectors[h](K[:, :, h, :])

                q_proj_heads.append(q_proj_h)
                k_proj_heads.append(k_proj_h)

            Q_proj = torch.stack(q_proj_heads, dim=2)  # (B, N, H, d_h)
            K_proj = torch.stack(k_proj_heads, dim=2)  # (B, N, H, d_h)

            if collect_diagnostics:
                def _stack_stats(key: str, stats_list: list[dict]) -> torch.Tensor:
                    # stats_list[h][key] is (B, N). We stack over heads.
                    stacked = torch.stack([s[key] for s in stats_list], dim=0)  # (H, B, N)
                    return stacked.permute(1, 2, 0)  # (B, N, H)

                diagnostics = {
                    "q_input_norm":     _stack_stats("input_norm",     q_stats),
                    "q_proj_norm":      _stack_stats("proj_norm",      q_stats),
                    "q_residual_norm":  _stack_stats("residual_norm",  q_stats),
                    "q_eta_mix":        _stack_stats("eta_mix",        q_stats),
                    "k_input_norm":     _stack_stats("input_norm",     k_stats),
                    "k_proj_norm":      _stack_stats("proj_norm",      k_stats),
                    "k_residual_norm":  _stack_stats("residual_norm",  k_stats),
                    "k_eta_mix":        _stack_stats("eta_mix",        k_stats),
                }

        else:
            Q_proj, K_proj = Q, K
            if collect_diagnostics:
                q_input_norm = Q.norm(dim=-1).detach()  # (B, N, H)
                k_input_norm = K.norm(dim=-1).detach()
                zeros_q = torch.zeros_like(q_input_norm)
                zeros_k = torch.zeros_like(k_input_norm)
                diagnostics = {
                    "q_input_norm":     q_input_norm,
                    "q_proj_norm":      q_input_norm,
                    "q_residual_norm":  zeros_q,
                    "q_eta_mix":        zeros_q,
                    "k_input_norm":     k_input_norm,
                    "k_proj_norm":      k_input_norm,
                    "k_residual_norm":  zeros_k,
                    "k_eta_mix":        zeros_k,
                }

        # Decide whether to run global or local attention
        use_local = (self.window_radius is not None) and (grid_hw is not None)
        if use_local:
            neighbor_index, neighbor_mask = self._get_neighbor_index(
                grid_hw,
                self.window_radius,
                x.device,
            )

        # Commutator loss (T5.1): average across heads
        head_losses = [
            self.rel_rotation[h].compute_commutator_loss()
            for h in range(self.n_heads)
        ]
        avg_comm_loss = torch.stack(head_losses).mean()

        # Compute attention per batch item / per head
        attn_outputs = []
        for b in range(B):
            batch_outputs = []
            for h in range(self.n_heads):
                # Structured relative rotations for this batch/head:
                # R_rel : (N, N, d_h, d_h)
                R_rel = self.rel_rotation[h](positions[b])

                if use_local:
                    out_h = self._attention_local(
                        Q_proj[b, :, h, :],                 # (N, d_h)
                        K_proj[b, :, h, :],                 # (N, d_h)
                        V[b, :, h, :],                      # (N, d_h)
                        R_rel,                              # (N, N, d_h, d_h)
                        None if mask is None else mask[b],  # (N, N) or None
                        neighbor_index,                     # (N, Wsize)
                        neighbor_mask,                      # (N, Wsize)
                    )
                else:
                    out_h = self._attention_full(
                        Q_proj[b, :, h, :],
                        K_proj[b, :, h, :],
                        V[b, :, h, :],
                        R_rel,
                        None if mask is None else mask[b],
                    )

                batch_outputs.append(out_h)  # (N, d_h)

            # Stack heads back: (N, H, d_h)
            batch_out = torch.stack(batch_outputs, dim=1)
            attn_outputs.append(batch_out)

        # (B, N, H, d_h) -> (B, N, H*d_h) -> final linear
        attn_output = torch.stack(attn_outputs, dim=0)
        attn_output = attn_output.reshape(B, N, self.n_heads * self.d_head)
        output = self.W_O(attn_output)  # (B, N, d_model)

        metrics: Dict[str, Any] = {
            "commutator_loss": avg_comm_loss,
        }
        if collect_diagnostics and diagnostics is not None:
            metrics["diagnostics"] = {
                key: value.detach().cpu()
                for key, value in diagnostics.items()
            }

        return output, metrics


# ---------------------------------------------------------------------
# RelativePositionTransformerBlock
# ---------------------------------------------------------------------

class RelativePositionTransformerBlock(nn.Module):
    """
    A standard Transformer block wrapper around RelativePositionAttention
    (norm -> attention -> residual, norm -> MLP -> residual).

    Adds window_radius passthrough so you can run local attention.
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_heads: int,
        d_ff: int,
        d_coord: int = 2,
        dropout: float = 0.1,
        window_radius: Optional[int] = None,
    ):
        super().__init__()

        self.attn = RelativePositionAttention(
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
            d_coord=d_coord,
            dropout=dropout,
            use_active_projection=True,
            window_radius=window_radius,
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
        x: torch.Tensor,                       # (B, N, d_model)
        positions: torch.Tensor,               # (N, d_coord) or (B, N, d_coord)
        mask: Optional[torch.Tensor] = None,   # (B, N, N)
        collect_diagnostics: bool = False,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x:        (B, N, d_model)
            positions: (N, d_coord) or (B, N, d_coord)
            mask:     optional (B, N, N)
            collect_diagnostics: pass-through to attention
            grid_hw:  (H, W) if using windowed attention

        Returns:
            (updated_x, metrics)
        """
        # Attention sublayer + residual
        attn_out, metrics = self.attn(
            self.norm1(x),
            positions,
            mask,
            collect_diagnostics=collect_diagnostics,
            grid_hw=grid_hw,
        )
        x = x + attn_out

        # Feedforward sublayer + residual
        x = x + self.ff(self.norm2(x))

        return x, metrics
