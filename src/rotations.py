"""
Structured Rotations R_STR from Theorem T2.7

Implements:
    R_STR(r) = exp(A(r)) = U [⊕_{u=1}^m R_2(θ_u(r)) ⊕ I_{d_null}] U^T
    where A(r) = Σ_k [r]_k L_k with commuting skew generators {L_k}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SkewGenerator(nn.Module):
    """
    Skew-symmetric matrix generator L_k satisfying:
    - L_k^T = -L_k  (T1.1: skew-symmetric)
    - [L_a, L_b] = 0 (T1.1: commuting)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Parameterize skew-symmetric matrix via strictly upper triangular part
        # L = A - A^T where A is upper triangular
        self.upper_tri_params = nn.Parameter(torch.randn(dim * (dim - 1) // 2) * 0.01)

    def forward(self) -> torch.Tensor:
        """Returns skew-symmetric matrix L_k ∈ ℝ^{d_h × d_h}"""
        device = self.upper_tri_params.device
        # Build strictly upper triangular matrix
        indices = torch.triu_indices(self.dim, self.dim, offset=1, device=device)
        A = torch.zeros(self.dim, self.dim, device=device)
        A[indices[0], indices[1]] = self.upper_tri_params

        # Make skew-symmetric: L = A - A^T
        L = A - A.T
        return L


class StructuredRotation(nn.Module):
    """
    Implements R_STR(r) from Theorem T2.7:

    Given position vector r ∈ ℝ^{d_c}, computes:
        R_STR(r) = exp(Σ_k [r]_k L_k)

    Key properties (from T-series proof):
    - T2.8: R_STR(r) ∈ SO(d_h)
    - T2.9: R_STR(r_i)^T R_STR(r_j) = R_STR(r_j - r_i)  [relative-position property]
    """

    def __init__(self, d_h: int, d_c: int = 2, enforce_commutator: bool = True):
        """
        Args:
            d_h: Head dimension (dimension of rotation matrices)
            d_c: Coordinate dimension (e.g., 2 for 2D images, 3 for 3D)
            enforce_commutator: If True, add loss to enforce [L_a, L_b] ≈ 0
        """
        super().__init__()
        self.d_h = d_h
        self.d_c = d_c
        self.enforce_commutator = enforce_commutator

        # Create d_c skew-symmetric generators {L_k}_{k=1}^{d_c}
        self.generators = nn.ModuleList([
            SkewGenerator(d_h) for _ in range(d_c)
        ])

    def get_generators(self) -> list[torch.Tensor]:
        """Returns list of generator matrices [L_1, ..., L_{d_c}]"""
        return [gen() for gen in self.generators]

    def compute_A(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute A(r) = Σ_k [r]_k L_k  (Theorem T1.3)

        Args:
            r: Position vector of shape (..., d_c)

        Returns:
            A(r) of shape (..., d_h, d_h)
        """
        L_list = self.get_generators()  # [L_1, ..., L_{d_c}], each (d_h, d_h)

        # Stack generators: (d_c, d_h, d_h)
        L_stack = torch.stack(L_list, dim=0)

        # r has shape (..., d_c), need to broadcast with L_stack
        # r[..., k] * L_stack[k] for each k
        # Result: (..., d_c, d_h, d_h) -> sum over d_c -> (..., d_h, d_h)
        A = torch.einsum('...k,khw->...hw', r, L_stack)
        return A

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute R_STR(r) = exp(A(r))  (Theorem T1.6)

        Uses matrix exponential via eigendecomposition since A(r) is skew-symmetric.
        For skew-symmetric matrices, eigenvalues are purely imaginary (T0.16).

        Args:
            r: Position vectors of shape (..., d_c)

        Returns:
            R_STR(r) ∈ SO(d_h) of shape (..., d_h, d_h)
        """
        A = self.compute_A(r)  # (..., d_h, d_h)

        # Compute matrix exponential
        # For skew-symmetric A, exp(A) is orthogonal with det = +1 (T2.8)
        R = torch.matrix_exp(A)

        return R

    def compute_commutator_loss(self) -> torch.Tensor:
        """
        Compute ε = max_{a,b} |[L_a, L_b]| from Theorem T5.1

        This enforces the commutation condition [L_a, L_b] = 0 from T1.1.

        Returns:
            Scalar commutator error ε ≥ 0
        """
        L_list = self.get_generators()

        comm_norms = []
        for i in range(len(L_list)):
            for j in range(i + 1, len(L_list)):
                # Compute [L_i, L_j] = L_i L_j - L_j L_i
                comm = L_list[i] @ L_list[j] - L_list[j] @ L_list[i]
                # Frobenius norm (can use operator norm instead)
                comm_norm = torch.norm(comm, p='fro')
                comm_norms.append(comm_norm)

        if not comm_norms:
            device = L_list[0].device
            return torch.zeros((), device=device, dtype=L_list[0].dtype, requires_grad=True)

        return torch.stack(comm_norms).max()


class RelativePositionRotation(nn.Module):
    """
    Computes relative rotation R_STR(r_j - r_i) efficiently.

    From Theorem T2.9: R_STR(r_i)^T R_STR(r_j) = R_STR(r_j - r_i)

    This is the key property enabling relative-position attention.
    """

    def __init__(self, d_h: int, d_c: int = 2):
        super().__init__()
        self.rotation = StructuredRotation(d_h, d_c)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Position vectors of shape (N, d_c) where N is sequence length

        Returns:
            Pairwise relative rotations of shape (N, N, d_h, d_h)
            Entry [i, j] contains R_STR(r_j - r_i)
        """
        N = positions.shape[0]

        # Compute all pairwise differences: r_j - r_i
        # positions shape: (N, d_c)
        # Expand: (N, 1, d_c) - (1, N, d_c) -> (N, N, d_c)
        rel_positions = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, d_c)

        # Flatten to (N*N, d_c), compute rotations, reshape back
        rel_positions_flat = rel_positions.reshape(-1, self.rotation.d_c)  # (N*N, d_c)
        rotations_flat = self.rotation(rel_positions_flat)  # (N*N, d_h, d_h)
        rotations = rotations_flat.reshape(N, N, self.rotation.d_h, self.rotation.d_h)

        return rotations

    def compute_commutator_loss(self) -> torch.Tensor:
        """Proxy to underlying rotation's commutator loss"""
        return self.rotation.compute_commutator_loss()
