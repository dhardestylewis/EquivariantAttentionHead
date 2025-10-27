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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: Image tensor of shape (B, C, H, W)

        Returns:
            logits: Class logits of shape (B, num_classes)
            metrics: Dict containing loss terms and diagnostics
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
        for block in self.blocks:
            x, metrics = block(x, cls_pos.squeeze(0))  # positions: (N+1, 2)
            total_comm_loss += metrics['commutator_loss']

        # Average commutator loss across layers
        avg_comm_loss = total_comm_loss / len(self.blocks)

        # Classification head (use CLS token)
        x = self.norm(x)
        cls_out = x[:, 0]  # (B, d_model)
        logits = self.head(cls_out)  # (B, num_classes)

        metrics = {
            'commutator_loss': avg_comm_loss,
        }

        return logits, metrics


def create_mnist_model(**kwargs) -> RelativePositionViT:
    """Create model for MNIST (28x28 grayscale images)"""
    defaults = dict(
        img_size=28,
        patch_size=4,  # 7x7 = 49 patches
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
    defaults.update(kwargs)
    return RelativePositionViT(**defaults)


def create_cifar_model(**kwargs) -> RelativePositionViT:
    """Create model for CIFAR-10/100 (32x32 RGB images)"""
    defaults = dict(
        img_size=32,
        patch_size=4,  # 8x8 = 64 patches
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
    defaults.update(kwargs)
    return RelativePositionViT(**defaults)
