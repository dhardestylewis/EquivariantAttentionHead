
"""
STRING Robustness "Consumer Report"
===================================
Empirical validation of STRING robustness theorems (Fragility & Zero-Gap).
Demonstrates:
1.  Exact STRING (Commuting L, Block-Diag P) => Zero Invariant Residual => Zero Generalization Gap.
2.  Relaxed STRING (Non-commuting L, Mixed P) => Metric Growth => OOD Failure.

References:
- Schenck et al., "Learning the RoPEs..." (2025)
- Technical Report: publications/STRING-Robustness.tex.d/main.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SEED = 42
D_MODEL = 32
HEAD_DIM = 16  # Must be even
NUM_HEADS = 2
NUM_CLASSES = 10
SEQ_LEN = 196  # 14x14 patches

torch.manual_seed(SEED)

# ==============================================================================
# Helper: Explicit Lie Algebra Generator for Validation
# ==============================================================================

class ExplicitLiePE(nn.Module):
    """
    A reference implementation of STRING acting via explicit Matrix Exponential.
    Allows testing 'Relaxed' constraints by explicitly constructing L_k matrices.
    """
    def __init__(self, d_h, d_c, allow_noncommuting=False, allow_mixing=False):
        super().__init__()
        self.d_h = d_h # Hidden dim per head
        self.d_c = d_c # Coordinate dim (2 for 2D images)
        
        # 1. Construct Generators L_k (d_h x d_h skew-symmetric)
        # We start with a commuting basis (Exact STRING)
        # L_k = U @ BlockDiag(0, beta * J, ...) @ U.T
        
        # Random orthogonal basis U
        U = torch.linalg.qr(torch.randn(d_h, d_h))[0]
        self.register_buffer("U", U)
        
        # Create commuting blocks (2x2 skew matrices)
        self.L_param = nn.ParameterList()
        
        for k in range(d_c):
            # Create a block-diagonal skew matrix in the U-basis
            # We explicitly enforce block-diag structure to ensure commutativity
            # unless 'allow_noncommuting' is True
            
            L_base = torch.zeros(d_h, d_h)
            # Fill 2x2 blocks
            for i in range(0, d_h-1, 2):
                freq = torch.randn(1) * 2.0 # Random frequency
                L_base[i, i+1] = -freq
                L_base[i+1, i] = freq
            
            # Rotate to shared basis
            L_k = U @ L_base @ U.T
            
            if allow_noncommuting:
                # Add random skew-symmetric noise that breaks the shared block structure
                noise = torch.randn(d_h, d_h) * 0.5
                noise = (noise - noise.T)
                L_k = L_k + noise # Now [L_a, L_b] != 0 potentially
                
            self.L_param.append(nn.Parameter(L_k))

        # 2. Construct Post-Rotation P_sp
        if allow_mixing:
            # Random rotation (mixes subspaces)
            P_rand = torch.linalg.qr(torch.randn(d_h, d_h))[0]
            self.register_buffer("P_sp", P_rand)
        else:
            # Identity (Block diagonal wrt representation)
            self.register_buffer("P_sp", torch.eye(d_h))

    def get_L(self, k):
        # Enforce skew-symmetry in case of updates
        Lk = self.L_param[k]
        return 0.5 * (Lk - Lk.T)

    def forward(self, x, r_grid):
        """
        x: [Batch, Seq, d_h]
        r_grid: [Batch, Seq, d_c]
        Returns: Rotated x
        """
        B, S, _ = x.shape
        
        # Compute A(r) = sum r_k L_k for each token
        # A_r: [B, S, d_h, d_h]
        A_r = torch.zeros(B, S, self.d_h, self.d_h, device=x.device)
        
        for k in range(self.d_c):
            Lk = self.get_L(k) # [d_h, d_h]
            # r_grid[..., k]: [B, S]
            A_r += r_grid[..., k].unsqueeze(-1).unsqueeze(-1) * Lk.unsqueeze(0).unsqueeze(0)
            
        # Compute R(r) = exp(A(r))
        # Note: matrix_exp is computationally heavy, but necessary for exact validation
        R_r = torch.linalg.matrix_exp(A_r) # [B, S, d_h, d_h]
        
        # Apply Post-Rotation P_sp
        R_eff = R_r @ self.P_sp # [B, S, d_h, d_h]
        
        # Apply rotation to x: x_rot = R x
        # x is [B, S, d_h] -> [B, S, d_h, 1]
        x_out = R_eff @ x.unsqueeze(-1)
        return x_out.squeeze(-1)

# ==============================================================================
# Verifier Functions (Metrics from Theorems)
# ==============================================================================

def compute_commutator_norm(model_pe):
    """
    Computes epsilon = max || [L_a, L_b] ||
    """
    d_c = model_pe.d_c
    eps_max = 0.0
    with torch.no_grad():
        for i in range(d_c):
            for j in range(i+1, d_c):
                Li = model_pe.get_L(i)
                Lj = model_pe.get_L(j)
                comm = Li @ Lj - Lj @ Li
                eps = torch.norm(comm, p='fro').item()
                eps_max = max(eps_max, eps)
    return eps_max

def compute_mixing_norm(model_pe):
    """
    Computes off-block mass of P_sp with respect to the generator basis U.
    We transform P_sp back to the U basis: P_aligned = U.T @ P_sp @ U
    And check if it is approximately 2x2 block diagonal.
    """
    with torch.no_grad():
        P = model_pe.P_sp
        U = model_pe.U
        P_aligned = U.T @ P @ U
        # For simplicity in this demo, "Mixing" is defined as any mass outside the diagonal blocks
        # This is a proxy for active-null mixing.
        d = P.shape[0]
        mask = torch.zeros_like(P_aligned)
        for i in range(0, d-1, 2):
            mask[i:i+2, i:i+2] = 1.0
        
        off_block = P_aligned * (1.0 - mask)
        return torch.norm(off_block, p='fro').item()

# ==============================================================================
# Direct Operator Identity Test (Theorem-Level Validation)
# ==============================================================================

def check_operator_identity(model_pe, num_samples=100):
    """
    Directly tests the algebraic identity:
    || R(r)^T R(s) - R(s - r) || / || R(s - r) ||
    This is the purest test of the 'Relative Position Property'.
    """
    d_c = model_pe.d_c
    d_h = model_pe.d_h
    
    # Random positions
    r = torch.randn(num_samples, d_c, device=DEVICE)
    s = torch.randn(num_samples, d_c, device=DEVICE)
    
    # We need to compute R(r) manually using the PE module's internals (A_r)
    # The ExplicitLiePE.forward applies R(r)x. We need the matrix itself.
    # A(r) = sum r_k L_k
    
    def get_R(pos):
        B = pos.shape[0]
        # pos: [B, d_c]
        A = torch.zeros(B, d_h, d_h, device=DEVICE)
        for k in range(d_c):
            Lk = model_pe.get_L(k)
            A += pos[:, k].view(B, 1, 1) * Lk.view(1, d_h, d_h)
        R_base = torch.linalg.matrix_exp(A)
        return R_base @ model_pe.P_sp # Apply post-rotation
    
    with torch.no_grad():
        Rr = get_R(r) # [B, d_h, d_h]
        Rs = get_R(s)
        R_rel_true = get_R(s - r) # Note: For relaxed, this uses P_sp too, which might not commute
        
        # Test LHS: R(r)^T R(s)
        # Transpose of Rr:
        Rr_T = Rr.transpose(1, 2)
        LHS = Rr_T @ Rs
        
        # Difference
        diff = torch.norm(LHS - R_rel_true, dim=(1,2))
        base = torch.norm(R_rel_true, dim=(1,2))
        
        rel_error = (diff / base).mean().item()
    return rel_error

# ==============================================================================
# Model Wrapper for Training
# ==============================================================================

class TinyViT(nn.Module):
    def __init__(self, pe_module):
        super().__init__()
        self.patch_embed = nn.Linear(1, D_MODEL)
        self.pe_mod = pe_module
        # Single head for clarity, or explicit reshaping
        self.head_dim = HEAD_DIM
        self.num_heads = NUM_HEADS
        
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        
        self.fc = nn.Linear(D_MODEL, NUM_CLASSES)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))

    def forward(self, x, r_grid, r_cls_shift=None):
        """
        x: [B, H, W] -> [B, N, 1]
        r_grid: [B, N, 2] coordinates for patches
        r_cls_shift: [B, 1, 2] optional coordinate for CLS (default 0)
        """
        B, H, W = x.shape
        x_flat = x.view(B, H*W, 1)
        
        # Embed
        x_emb = self.patch_embed(x_flat) # [B, N, D]
        
        # Add CLS
        cls = self.cls_token.expand(B, -1, -1)
        x_in = torch.cat([cls, x_emb], dim=1)
        
        # QKV
        q = self.q_proj(x_in).view(B, -1, self.num_heads, self.head_dim)
        k = self.k_proj(x_in).view(B, -1, self.num_heads, self.head_dim)
        # v: [B, N+1, D] -> [B, N+1, H, d_h] -> [B, H, N+1, d_h]
        v = self.v_proj(x_in).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE/STRING to Q, K (Skip CLS or shift CLS)
        if r_cls_shift is None:
            r_cls = torch.zeros(B, 1, 2, device=x.device)
        else:
            r_cls = r_cls_shift

        r_seq = torch.cat([r_cls, r_grid], dim=1) # [B, N+1, 2]
        
        # Apply Explicit Lie PE
        # Reshape for application: [B*Heads, Seq, HeadDim]
        q_flat = q.transpose(1, 2).reshape(-1, q.shape[1], self.head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, k.shape[1], self.head_dim)
        r_rep = r_seq.repeat_interleave(self.num_heads, dim=0)

        q_rot = self.pe_mod(q_flat, r_rep)
        k_rot = self.pe_mod(k_flat, r_rep)
        
        # Reshape back [B, H, N+1, d_h]
        q_rot = q_rot.view(B, self.num_heads, -1, self.head_dim)
        k_rot = k_rot.view(B, self.num_heads, -1, self.head_dim)
        
        # Attention: [B, H, N, N]
        attn = (q_rot @ k_rot.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Out: [B, H, N, d_h] -> [B, N, H, d_h] -> [B, N, D]
        out = (attn @ v).transpose(1, 2).reshape(B, -1, D_MODEL)
        
        # Pool
        out_cls = out[:, 0]
        return self.fc(out_cls)

# ==============================================================================
# Helper: Data & Coordinates
# ==============================================================================

def get_grid(h, w, batch_size):
    y = torch.linspace(-1, 1, h)
    x = torch.linspace(-1, 1, w)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1) # [H, W, 2]
    return grid.view(h*w, 2).unsqueeze(0).repeat(batch_size, 1, 1).to(DEVICE)

def get_mnist_loader(batch_size=64, shift_pixels=None):
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if shift_pixels is not None:
        dx, dy = shift_pixels
        # Integer pixel translation
        transform_list.append(
            transforms.Lambda(lambda x: transforms.functional.affine(x, angle=0, translate=(dx, dy), scale=1, shear=0))
        )
    
    transform = transforms.Compose(transform_list)
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    # Take subset for speed
    subset = torch.utils.data.Subset(dataset, range(1000))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)

def train_quick(model, epochs=1):
    """
    Train briefly so the loss landscape is valid for Metric C.
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ]))
    # Small subset 
    subset = torch.utils.data.Subset(loader, range(2000)) 
    train_loader = DataLoader(subset, batch_size=32, shuffle=True)
    
    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    for ep in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE).squeeze(1), y.to(DEVICE)
            r0 = get_grid(28, 28, x.shape[0])
            opt.zero_grad()
            out = model(x, r0)
            loss = crit(out, y)
            loss.backward()
            opt.step()
    model.eval()

# ==============================================================================
# Invariance Measurement (Theorem-Level B)
# ==============================================================================

def measure_invariant_residual(model_wrapper, delta_tensor):
    """
    Approximates IR_spec: E[ || Op(r)^T Op(r+d) - Op(d) || ]
    We measure this via Logit Difference on FIXED images but SHIFTED grids.
    FIX: We must shift CLS token too to maintain relative positions!
    """
    loader = get_mnist_loader(BATCH_SIZE)
    diffs = []
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE).squeeze(1) # [B, 28, 28]
            B = x.shape[0]
            
            # 1. Base Grid
            r0 = get_grid(28, 28, B)
            r_cls_0 = torch.zeros(B, 1, 2, device=DEVICE)
            
            # 2. Shifted Grid (Global Shift)
            d_expand = delta_tensor.view(1, 1, 2)
            r_shift = r0 + d_expand
            r_cls_shift = r_cls_0 + d_expand
            
            logits_0 = model_wrapper(x, r0, r_cls_shift=r_cls_0)
            logits_shift = model_wrapper(x, r_shift, r_cls_shift=r_cls_shift)
            
            # Norm difference
            diff = torch.norm(logits_0 - logits_shift, p='fro') / torch.norm(logits_0, p='fro')
            diffs.append(diff.item())
            
            if len(diffs) > 5: break
            
    return np.mean(diffs)

def measure_loss_gap(model_wrapper, delta_pixels, delta_grid):
    """
    Approximates Generalization Gap: | Risk_shift - Risk_train |
    We shift pixels AND coordinate grid -> "Camera Motion".
    If invariant, loss should be identical.
    """
    # Train risk (no shift)
    # Note: Model should be pre-trained
    loader_train = get_mnist_loader(BATCH_SIZE, shift_pixels=(0,0))
    loss_train = []
    crit = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in loader_train:
            x, y = x.to(DEVICE).squeeze(1), y.to(DEVICE)
            r0 = get_grid(28, 28, x.shape[0])
            out = model_wrapper(x, r0) # Standard coords
            loss_train.append(crit(out, y).item())
    
    # Target risk (shift pixels + shift grid)
    loader_target = get_mnist_loader(BATCH_SIZE, shift_pixels=delta_pixels)
    loss_target = []
    
    with torch.no_grad():
        for x, y in loader_target:
            x, y = x.to(DEVICE).squeeze(1), y.to(DEVICE)
            
            # Shift grid to match pixels
            B = x.shape[0]
            r0 = get_grid(28, 28, B)
            d_expand = delta_grid.view(1, 1, 2)
            r_shift = r0 + d_expand
            r_cls_shift = torch.zeros(B, 1, 2, device=DEVICE) + d_expand
            
            # If model is perfectly Equivariant, feeding ShiftedPixels + ShiftedGrid
            # should yield SameLogits as UnshiftedPixels + UnshiftedGrid
            # Because Relative Positions are preserved.
            out = model_wrapper(x, r_shift, r_cls_shift=r_cls_shift)
            loss_target.append(crit(out, y).item())
            
    return abs(np.mean(loss_target) - np.mean(loss_train))

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

if __name__ == "__main__":
    print("Initializing Robustness Demonstration with Explicit Lie Generators...")
    
    # 1. Instantiate Models
    # Exact: Commuting, No Mixing
    pe_exact = ExplicitLiePE(d_h=HEAD_DIM, d_c=2, allow_noncommuting=False, allow_mixing=False).to(DEVICE)
    model_exact = TinyViT(pe_exact).to(DEVICE)
    
    # Relaxed: Non-commuting, Mixing
    pe_relaxed = ExplicitLiePE(d_h=HEAD_DIM, d_c=2, allow_noncommuting=True, allow_mixing=True).to(DEVICE)
    model_relaxed = TinyViT(pe_relaxed).to(DEVICE)
    
    # 2. Verify Constraints (Metric A)
    print("\n[Metric A] Verifying Constraints:")
    
    eps_exact = compute_commutator_norm(model_exact.pe_mod)
    mix_exact = compute_mixing_norm(model_exact.pe_mod)
    print(f"Exact STRING   | Commutator Norm (epsilon): {eps_exact:.2e} | Mixing Norm: {mix_exact:.2e}")
    
    eps_relaxed = compute_commutator_norm(model_relaxed.pe_mod)
    mix_relaxed = compute_mixing_norm(model_relaxed.pe_mod)
    print(f"Relaxed STRING | Commutator Norm (epsilon): {eps_relaxed:.2e} | Mixing Norm: {mix_relaxed:.2e}")
    
    assert eps_exact < 1e-4, "Exact model failed commutativity check!"
    assert eps_relaxed > 1e-3, "Relaxed model failed to break commutativity!"

    # 2b. Direct Operator Identity Test (New Pure Test)
    print("\n[Metric A'] Direct Relative Operator Identity check: || R(r)^T R(s) - R(s-r) || / || R(s-r) ||")
    err_op_exact = check_operator_identity(pe_exact)
    err_op_relaxed = check_operator_identity(pe_relaxed)
    print(f"Exact Rel Error:   {err_op_exact:.2e}")
    print(f"Relaxed Rel Error: {err_op_relaxed:.2e}")

    # 2c. Brief Training for Metric C validity
    print("\nTraining models briefly (1 epoch) to establish loss landscape...")
    train_quick(model_exact, epochs=1)
    train_quick(model_relaxed, epochs=1)
    
    # 3. Sweep Shifts
    print("\n[Metric B (Logits) & C (Risk Gap)] Sweeping Shift Magnitude...")
    
    shifts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # Coordinate units
    
    res_exact_ir = []
    res_relaxed_ir = []
    res_exact_gap = []
    res_relaxed_gap = []
    
    for delta in shifts:
        # Vector shift (delta, delta)
        d_vec = torch.tensor([delta, delta], device=DEVICE)
        d_pix = (int(delta*5), int(delta*5)) # Roughly 5 pixels per 1.0 unit (28px span is [-1, 1], so 2.0 dist)
        
        # Measure IR (Logit Diff)
        ir_e = measure_invariant_residual(model_exact, d_vec)
        ir_r = measure_invariant_residual(model_relaxed, d_vec)
        
        # Measure Risk Gap
        gap_e = measure_loss_gap(model_exact, d_pix, d_vec)
        gap_r = measure_loss_gap(model_relaxed, d_pix, d_vec)
        
        res_exact_ir.append(ir_e)
        res_relaxed_ir.append(ir_r)
        
        res_exact_gap.append(gap_e)
        res_relaxed_gap.append(gap_r)
        
        print(f"Shift {delta:.1f} | LogitDiff: Ex={ir_e:.3f}, Rx={ir_r:.3f} | LossGap: Ex={gap_e:.3f}, Rx={gap_r:.3f}")

    # 4. Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(shifts, res_exact_ir, 'o-', label=f"Exact (Err={err_op_exact:.1e})")
    plt.plot(shifts, res_relaxed_ir, 'x--', label=f"Relaxed (Err={err_op_relaxed:.1e})")
    plt.title("Constraint B: Logit Invariance (Proxy)")
    plt.xlabel("Coordinate Shift Magnitude")
    plt.ylabel("|| Logit(r) - Logit(r+d) ||")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(shifts, res_exact_gap, 'o-', label="Exact")
    plt.plot(shifts, res_relaxed_gap, 'x--', label="Relaxed")
    plt.title("Constraint C: Generalization Gap (Risk)")
    plt.xlabel("Shift Magnitude")
    plt.ylabel("| Loss(Shift) - Loss(Train) |")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("mnist_robustness_verified.png")
    print("\nVerified plot saved to 'mnist_robustness_verified.png'")
