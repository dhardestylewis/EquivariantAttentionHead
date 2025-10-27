"""
Vision Transformer with Relative-Position Attention

Applies the T-series theorem to vision tasks (MNIST, CIFAR-10).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import RelativePositionTransformerBlock


class PatchEmbedding(nn.Module):
    """
    Convert image into sequence of patch embeddings.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Patch projection (conv with kernel = stride = patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Image tensor of shape (B, C, H, W)

        Returns:
            embeddings: (B, N, d_model) where N = n_patches
            positions: (N, 2) position coordinates for each patch
        """
        B, C, H, W = x.shape

        # Extract patches
        x = self.proj(x)  # (B, d_model, H', W') where H' = W' = img_size // patch_size
        B, D, H_p, W_p = x.shape

        # Flatten spatial dimensions
        x = x.flatten(2)  # (B, d_model, H'*W')
        x = x.transpose(1, 2)  # (B, N, d_model)

        # Generate position coordinates for each patch
        # Positions in [0, 1]^2 normalized coordinates
        positions = self._generate_positions(H_p, W_p, x.device)  # (N, 2)

        return x, positions

    def _generate_positions(self, H: int, W: int, device) -> torch.Tensor:
        """
        Generate 2D position coordinates for patches.

        Returns (H*W, 2) tensor where each row is [x, y] ∈ [0, 1]^2
        """
        # Create grid
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)

        # Meshgrid
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Stack and flatten
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (H*W, 2)

        return positions


class RelativePositionViT(nn.Module):
    """
    Vision Transformer using Relative-Position Attention from T-series proof.

    Key difference from standard ViT:
    - No learned absolute positional embeddings
    - Attention uses R_STR(r_j - r_i) rotations (Theorem T2.9)
    - Enforces [L_a, L_b] ≈ 0 via commutator loss (Theorem T5.1)
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        d_model: int = 64,
        d_head: int = 32,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        d_coord: int = 2,
    ):
        """
        Args:
            img_size: Input image size (assumes square images)
            patch_size: Size of each patch
            in_channels: Number of input channels (1 for MNIST, 3 for CIFAR)
            num_classes: Number of output classes
            d_model: Model dimension
            d_head: Dimension per attention head
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            d_ff: Feedforward dimension
            dropout: Dropout probability
            d_coord: Coordinate dimension (2 for 2D images)
        """
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)

        # CLS token (for classification)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            RelativePositionTransformerBlock(
                d_model=d_model,
                d_head=d_head,
                n_heads=n_heads,
                d_ff=d_ff,
                d_coord=d_coord,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _forward_impl(
        self,
        x: torch.Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Shared forward path with optional diagnostic collection.
        """
        B = x.shape[0]

        # Patch embedding
        x, positions = self.patch_embed(x)  # x: (B, N, d_model), positions: (N, 2)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, d_model)

        # CLS position at origin
        cls_pos = torch.zeros(1, self.patch_embed.n_patches + 1, 2, device=x.device)
        cls_pos[:, 1:, :] = positions.unsqueeze(0)  # (1, N+1, 2)

        # Transformer blocks
        total_comm_loss = 0.0
        diagnostics_per_block = []
        for block in self.blocks:
            x, block_metrics = block(
                x,
                cls_pos.squeeze(0),
                collect_diagnostics=collect_diagnostics,
            )
            total_comm_loss += block_metrics['commutator_loss']
            if collect_diagnostics:
                diagnostics_per_block.append(block_metrics.get('diagnostics'))

        # Average commutator loss across layers
        avg_comm_loss = total_comm_loss / len(self.blocks)

        # Classification head (use CLS token)
        x = self.norm(x)
        cls_out = x[:, 0]  # (B, d_model)
        logits = self.head(cls_out)  # (B, num_classes)

        metrics = {
            'commutator_loss': avg_comm_loss,
        }
        if collect_diagnostics:
            metrics['diagnostics'] = diagnostics_per_block

        return logits, metrics

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self._forward_impl(x, collect_diagnostics=False)

    def forward_with_diagnostics(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self._forward_impl(x, collect_diagnostics=True)




class AbsolutePositionTransformerBlock(nn.Module):
    """Standard multi-head self-attention block with learned absolute positions."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
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
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor = None,
        mask: torch.Tensor = None,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.resid_dropout(attn_output)
        x = x + self.ff(self.norm2(x))

        metrics = {
            'commutator_loss': x.new_zeros(()),
        }
        if collect_diagnostics:
            metrics['diagnostics'] = None

        return x, metrics


class AbsolutePositionViT(nn.Module):
    """Vision Transformer baseline with learned absolute positional embeddings."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        d_model: int,
        d_head: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        d_coord: int = 2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            AbsolutePositionTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _forward_impl(
        self,
        x: torch.Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        B = x.shape[0]
        x, _ = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_dropout(x)

        total_comm_loss = x.new_zeros(())
        diagnostics_per_block = []
        for block in self.blocks:
            x, metrics = block(x, collect_diagnostics=collect_diagnostics)
            total_comm_loss = total_comm_loss + metrics['commutator_loss']
            if collect_diagnostics:
                diagnostics_per_block.append(metrics.get('diagnostics'))

        x = self.norm(x)
        logits = self.head(x[:, 0])

        metrics = {
            'commutator_loss': total_comm_loss / max(len(self.blocks), 1),
        }
        if collect_diagnostics:
            metrics['diagnostics'] = diagnostics_per_block

        return logits, metrics

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self._forward_impl(x, collect_diagnostics=False)

    def forward_with_diagnostics(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self._forward_impl(x, collect_diagnostics=True)


class RotarySelfAttention(nn.Module):
    """Multi-head self attention with rotary positional embeddings (RoPE)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "RoPE requires even head dimension"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.scale = self.head_dim ** -0.5

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        rotated = torch.stack((-x2, x1), dim=-1)
        return rotated.reshape_as(x)

    def _apply_rotary(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotary_cache(self, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = sin[None, None, :, :].to(dtype)
        cos = cos[None, None, :, :].to(dtype)
        return sin, cos

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        sin, cos = self._rotary_cache(N, x.device, x.dtype)
        q = self._apply_rotary(q, sin, cos)
        k = self._apply_rotary(k, sin, cos)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.n_heads * self.head_dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class RoPETransformerBlock(nn.Module):
    """Transformer block that uses rotary self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = RotarySelfAttention(d_model, n_heads, dropout)
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
        positions: torch.Tensor = None,
        mask: torch.Tensor = None,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        attn_out = self.attn(self.norm1(x), mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))

        metrics = {
            'commutator_loss': x.new_zeros(()),
        }
        if collect_diagnostics:
            metrics['diagnostics'] = None

        return x, metrics


class RoPEViT(nn.Module):
    """Vision Transformer baseline using rotary positional embeddings."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        d_model: int,
        d_head: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        d_coord: int = 2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _forward_impl(
        self,
        x: torch.Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        B = x.shape[0]
        x, _ = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        total_comm_loss = x.new_zeros(())
        diagnostics_per_block = []
        for block in self.blocks:
            x, metrics = block(x, collect_diagnostics=collect_diagnostics)
            total_comm_loss = total_comm_loss + metrics['commutator_loss']
            if collect_diagnostics:
                diagnostics_per_block.append(metrics.get('diagnostics'))

        x = self.norm(x)
        logits = self.head(x[:, 0])

        metrics = {
            'commutator_loss': total_comm_loss / max(len(self.blocks), 1),
        }
        if collect_diagnostics:
            metrics['diagnostics'] = diagnostics_per_block

        return logits, metrics

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self._forward_impl(x, collect_diagnostics=False)

    def forward_with_diagnostics(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self._forward_impl(x, collect_diagnostics=True)

MODEL_REGISTRY = {
    'relative': RelativePositionViT,
    'absolute': AbsolutePositionViT,
    'rope': RoPEViT,
}


def _create_model(dataset: str, architecture: str, overrides: dict) -> nn.Module:
    dataset = dataset.lower()
    architecture = architecture.lower()

    if dataset == 'mnist':
        defaults = dict(
            img_size=28,
            patch_size=4,
            in_channels=1,
            num_classes=10,
            d_model=64,
            d_head=16,
            n_heads=4,
            n_layers=4,
            d_ff=256,
            dropout=0.1,
            d_coord=2,
        )
    elif dataset in {'cifar', 'cifar10'}:
        defaults = dict(
            img_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            d_model=128,
            d_head=32,
            n_heads=4,
            n_layers=6,
            d_ff=512,
            dropout=0.1,
            d_coord=2,
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    config = defaults.copy()
    config.update(overrides)

    try:
        model_cls = MODEL_REGISTRY[architecture]
    except KeyError as err:
        raise ValueError(f"Unknown architecture '{architecture}'") from err

    return model_cls(**config)


def create_mnist_model(architecture: str = 'relative', **kwargs) -> nn.Module:
    """Create model for MNIST (28x28 grayscale images)."""
    architecture = kwargs.pop('architecture', architecture)
    return _create_model('mnist', architecture, kwargs)


def create_cifar_model(architecture: str = 'relative', **kwargs) -> nn.Module:
    """Create model for CIFAR-10/100 (32x32 RGB images)."""
    architecture = kwargs.pop('architecture', architecture)
    return _create_model('cifar', architecture, kwargs)
