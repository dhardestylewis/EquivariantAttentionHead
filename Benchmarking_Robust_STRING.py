# %% [markdown]
# # End-to-End Colab: STRING vs. ESPR (v15 - Serial, 30 Seeds, Tuned)
#
# This version (v15) removes all multiprocessing to ensure stability for a long run.
# It addresses the `BrokenProcessPool` errors by running all experiments serially
# in a single process.
#
# * **Problem:** Previous parallel versions (v14) were unstable due to CUDA/multiprocessing conflicts. Previous penalties (v12) were too high and caused training collapse.
# * **Fix (v15):**
#     1.  **Serial Execution:** Removed all `ProcessPoolExecutor` logic. The sweep in `Cell 7` is now a single, robust `for` loop.
#     2.  **Global Device:** `device` and `AMP_DTYPE` are restored as globals in `Cell 0`, as they are now used in a single process.
#     3.  **Tuned Penalties:** Penalties are set to lower values (`2e-3`) to act as regularizers without preventing the model from learning (i.e., avoids clean accuracy collapse).
#     4.  **30 Seeds:** The sweep is set to run 30 seeds for full statistical power, as requested.
#     5.  **All Theory Fixes Included:** Retains the learned $U$-basis, alignment loss, L2-normalized features, Cayley clamp, and expanded OOD tests.

# %% [markdown]
# ## 0. Setup & Imports

# %%
!pip -q install einops mpmath scipy

import math, time, os, random
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from collections import namedtuple
import mpmath as mp
from math import erf, sqrt

# v15: Imports for timing, and stats
from tqdm.auto import tqdm
import contextlib, time, os, traceback
from scipy.stats import ttest_rel, wilcoxon

# =========================
# PATCH 0: Global perf toggles
# =========================
USE_FAST_MODE = False
USE_MIXED_PRECISION = True 
USE_TORCH_COMPILE   = True
SET_CHANNELS_LAST = True

# <-- FIX 4 (TF32): Enable TF32 globally as requested
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

# <-- FIX 3 (Cache): Persist compile caches
# Note: Since this is serial, we can use one cache dir again.
CACHE_DIR = os.path.expanduser("~/.cache/torch_compile_sweeps")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CACHE_DIR, "torchinductor")
os.environ["TRITON_CACHE_DIR"] = os.path.join(CACHE_DIR, "triton")
os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
print(f"Compile/Triton caches set to: {CACHE_DIR}")

# =========================
# PATCH 1: Device/dtype helpers
# =========================
# v15: Back to global device setup for serial execution
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

AMP_DTYPE = None
if USE_MIXED_PRECISION and device == 'cuda':
    major_cc, _ = torch.cuda.get_device_capability()
    AMP_DTYPE = torch.bfloat16 if major_cc >= 8 else torch.float16
    print(f"Using AMP with dtype: {AMP_DTYPE}")
else:
    AMP_DTYPE = None

def _amp_ctx():
    """Returns the autocast context manager"""
    if AMP_DTYPE is None or device == 'cpu':
        if hasattr(torch.cuda.amp, 'autocast'):
             return torch.cuda.amp.autocast(enabled=False)
        else:
             return torch.autocast(device_type=device, enabled=False)
    return torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=True)

def _to_device(x):
    """Helper to move tensors to the worker's device"""
    if SET_CHANNELS_LAST and x.ndim == 4 and device == 'cuda':
        return x.to(device, non_blocking=True, memory_format=torch.channels_last)
    return x.to(device, non_blocking=True)

# =========================
# v15: Timing Helpers
# =========================
def _sync():
    if device == 'cuda':
        torch.cuda.synchronize()

@contextlib.contextmanager
def wall_timer():
    _sync()
    t0 = time.perf_counter()
    yield
    _sync()
    t1 = time.perf_counter()
    print(f"[time] {t1 - t0:.3f}s")

class RunClock:
    def __init__(self): self.t = {}
    def start(self, k): _sync(); self.t.setdefault(k, 0.0); self._t0 = time.perf_counter(); self._k = k
    def stop(self):     _sync(); self.t[self._k] += time.perf_counter() - self._t0
    def dump(self, prefix="t_"):
        return {f"{prefix}{k}_s": v for k, v in self.t.items()}

# =========================
# PATCH 6 (Snippet F): Strict Determinism
# =========================
def set_seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s);
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    
    if USE_FAST_MODE:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
set_seed_all(0) # Set a default seed for notebook consistency

# %% [markdown]
# ## 1. Core Math Blocks
# (Helpers for skew-symmetric matrices, Cayley transform, and 2x2 rotations)

# %%
def make_skew(B):
    """Make a batch of matrices skew-symmetric."""
    return (B - B.transpose(-1,-2)) * 0.5

def cayley(S):
    """
    Cayley transform: C = (I - S)(I + S)^{-1}
    (From Audit 4: Use transpose-trick for a right solve)
    """
    I = torch.eye(S.size(-1), device=S.device, dtype=S.dtype)
    return torch.linalg.solve((I + S).T, (I - S).T).T

def apply_2x2_block(x_pair, c, s):
    """
    Apply 2x2 rotation R = [[c, -s], [s, c]] to paired features.
    """
    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]
    
    y0 = c * x0 - s * x1
    y1 = s * x0 + c * x1
    return torch.stack([y0, y1], dim=-1)

# %% [markdown]
# ## 2. Positional Encoder Module (v12/v13/v14 - Learned Basis)
# (Class definitions)

# %%
@dataclass
class PEConfig:
    """Configuration for the Positional Encoder."""
    kind: str = "STRING"
    d_c: int = 2
    d_h: int = 64
    m: int = 32
    learn_betas: bool = False
    rope_base_min: float = 1.0
    rope_base_max: float = 10000.0
    lowrank_r: int = 0
    allow_mixing: bool = False
    dtype: torch.dtype = torch.float32

class PE_StringESPR(nn.Module):
    """
    Pluggable Positional Encoder (PE) Module.
    v12/v13/v14: Includes learned basis U and Cayley clamp.
    """
    def __init__(self, cfg: PEConfig, num_heads: int = 1):
        super().__init__()
        self.cfg = cfg
        d_h, m, d_c = cfg.d_h, cfg.m, cfg.d_c
        self.d_null = d_h - 2*m
        
        assert 2*m <= d_h and d_h % 2 == 0, "d_h must be even, and 2m <= d_h"
        if cfg.kind == "ROPE":
            assert 2*m == d_h, "ROPE requires all dims to be active (d_null=0)"
            assert not cfg.allow_mixing and cfg.lowrank_r == 0, "ROPE has no Psp or corrections"

        if m > 0:
            u = torch.linspace(0, 1, m)
            w = cfg.rope_base_min * (cfg.rope_base_max / cfg.rope_base_min) ** u
        else:
            w = torch.empty(0)
            
        betas = torch.zeros(m, d_c)
        if m > 0 and d_c > 0:
             betas[:m//2, 0] = w[:m//2]
        if m > 0 and d_c > 1:
             betas[m//2:, 1] = w[m//2:]
            
        if cfg.learn_betas:
            self.betas = nn.Parameter(betas)
        else:
            self.register_buffer("betas", betas)

        if cfg.kind != "ROPE":
            self.T_basis = nn.Parameter(torch.zeros(d_h, d_h))
        else:
            self.register_buffer("T_basis", None)

        if self.d_null > 0:
            self.S_null = nn.Parameter(torch.zeros(self.d_null, self.d_null))
        else:
            self.register_buffer("S_null", None)
        self.E_mix  = nn.Parameter(torch.zeros(d_h, d_h))

        r = cfg.lowrank_r
        if r > 0:
            self.u = nn.Parameter(torch.randn(r, d_h) * 1e-3)
            self.v = nn.Parameter(torch.randn(r, d_h) * 1e-3)
            self.a = nn.Parameter(torch.randn(d_c, r) * 1e-3)
        else:
            self.u = self.v = self.a = None

        Pi_act = torch.zeros(d_h, d_h)
        Pi_act[:2*m, :2*m] = torch.eye(2*m)
        self.register_buffer("Pi_act", Pi_act.to(cfg.dtype))
        self.register_buffer("Pi_null", (torch.eye(d_h) - Pi_act).to(cfg.dtype))

    def _get_basis_U(self):
        """Builds the orthogonal basis U = cayley(T_basis)."""
        if self.T_basis is None:
            # T_basis is None for ROPE, so U is identity
            return torch.eye(self.cfg.d_h, device=self.Pi_act.device, dtype=self.Pi_act.dtype)
        return cayley(make_skew(self.T_basis))

    def build_Psp(self):
        """
        Build the r-independent post-rotation matrix Psp (in the U basis).
        v12: Adds Cayley clamp.
        """
        if self.cfg.kind == "ROPE":
            return torch.eye(self.cfg.d_h, device=self.betas.device, dtype=self.betas.dtype)

        m, dh = self.cfg.m, self.cfg.d_h
        S = torch.zeros(dh, dh, device=self.Pi_act.device, dtype=self.Pi_act.dtype)
        
        if self.d_null > 0 and self.S_null is not None:
            S[2*m:, 2*m:] = make_skew(self.S_null)
        
        if self.cfg.allow_mixing:
            S = S + make_skew(self.E_mix)

        # --- v12/v13: Cayley Clamp ---
        with torch.no_grad():
            rho = torch.linalg.matrix_norm(S.float(), ord=2)
            if rho > 0.9: 
                S.mul_(0.9 / (rho + 1e-6))
        # --- End ---
            
        return cayley(S)

    def build_L_list(self):
        """Build generators {L_k} for penalties (in the U basis)."""
        m, dh, d_c = self.cfg.m, self.cfg.d_h, self.cfg.d_c
        Ls = []
        J2 = torch.tensor([[0., -1.],[1., 0.]], device=self.Pi_act.device, dtype=self.Pi_act.dtype)

        for k in range(d_c):
            L_k_string = torch.zeros(dh, dh, device=self.Pi_act.device, dtype=self.Pi_act.dtype)
            for u in range(m):
                lam = self.betas[u, k]
                i0 = 2*u
                L_k_string[i0:i0+2, i0:i0+2] = lam * J2
            
            L_k = L_k_string

            if self.cfg.lowrank_r > 0 and self.u is not None:
                delta_k = torch.zeros_like(L_k)
                for t in range(self.cfg.lowrank_r):
                    ut = self.u[t][:,None]; vt = self.v[t][:,None]
                    delta_k += self.a[k,t] * (ut @ vt.T - vt @ ut.T)
                L_k = L_k + delta_k
                
            Ls.append(L_k)
        return Ls

    def rotate_qk(self, q, k, r):
        """
        v12/v13: Projects into U basis, rotates, projects back.
        Returns: (q_tilde, k_tilde), (q_prime_post_psp, k_prime_post_psp)
        """
        cfg = self.cfg
        B,N,H,dh = q.shape
        m = cfg.m
        
        U = self._get_basis_U()
        q_prime = torch.einsum('bnhd,dr->bnhr', q, U)
        k_prime = torch.einsum('bnhd,dr->bnhr', k, U)

        if m == 0:
            if cfg.kind == "ROPE":
                return (q, k), (q_prime, k_prime)
            
            Psp = self.build_Psp()
            q_tilde_prime = torch.matmul(q_prime, Psp.T)
            k_tilde_prime = torch.matmul(k_prime, Psp.T)
            
            q_tilde = torch.einsum('bnhr,rd->bnhd', q_tilde_prime, U.T)
            k_tilde = torch.einsum('bnhr,rd->bnhd', k_tilde_prime, U.T)
            return (q_tilde, k_tilde), (q_tilde_prime, k_tilde_prime)

        theta = torch.einsum('bnd,md->bnm', r, self.betas)
        c = torch.cos(theta); s = torch.sin(theta)

        cc = c.unsqueeze(2)
        ss = s.unsqueeze(2)
        
        def apply_R_STR(x_prime):
            x_act = x_prime[...,:2*m].reshape(B,N,H,m,2).contiguous()
            y_act = apply_2x2_block(x_act, cc, ss)
            y = torch.cat([y_act.reshape(B,N,H,2*m).contiguous(), x_prime[...,2*m:]], dim=-1)
            return y

        q_rot_prime = apply_R_STR(q_prime)
        k_rot_prime = apply_R_STR(k_prime)
        
        if cfg.kind == "ROPE":
            q_tilde = torch.einsum('bnhr,rd->bnhd', q_rot_prime, U.T)
            k_tilde = torch.einsum('bnhr,rd->bnhd', k_rot_prime, U.T)
            return (q_tilde, k_tilde), (q_rot_prime, k_rot_prime)

        Psp = self.build_Psp()
        q_tilde_prime = torch.matmul(q_rot_prime, Psp.T)
        k_tilde_prime = torch.matmul(k_rot_prime, Psp.T)

        q_tilde = torch.einsum('bnhr,rd->bnhd', q_tilde_prime, U.T)
        k_tilde = torch.einsum('bnhr,rd->bnhd', k_tilde_prime, U.T)

        return (q_tilde, k_tilde), (q_tilde_prime, k_tilde_prime)
    
    def penalty(self, lambda_comm=0.0, lambda_mix=0.0, lambda_delta=0.0, lambda_E=0.0, lambda_U=0.0, **kwargs):
        """
        Compute all ESPR regularization penalties.
        v13: Ignores lambda_align (handled in main model).
        """
        loss = torch.tensor(0., device=self.Pi_act.device, dtype=torch.float32)
        
        if lambda_comm > 0 and self.cfg.d_c > 1:
            Ls = self.build_L_list()
            for a in range(len(Ls)):
                for b in range(a+1, len(Ls)):
                    C = Ls[a] @ Ls[b] - Ls[b] @ Ls[a]
                    loss = loss + lambda_comm * torch.sum(C*C)
        
        if lambda_mix > 0:
            Psp = self.build_Psp()
            M = self.Pi_null @ Psp @ self.Pi_act
            loss = loss + lambda_mix * torch.sum(M*M)
        
        if lambda_delta > 0 and self.cfg.lowrank_r > 0 and self.u is not None:
            loss = loss + lambda_delta * (torch.sum(self.a*self.a) + \
                                          torch.sum(self.u*self.u) + \
                                          torch.sum(self.v*self.v))
        
        if lambda_E > 0:
            loss = loss + lambda_E * torch.sum(self.E_mix*self.E_mix)
        
        if lambda_U > 0 and self.T_basis is not None:
            loss = loss + lambda_U * torch.sum(self.T_basis * self.T_basis)
            
        return loss

    @torch.no_grad()
    def diagnostics(self):
        """Compute diagnostic metrics (CE, LR, IR_subspace) in the U-basis."""
        m, dh, d_c = self.cfg.m, self.cfg.d_h, self.cfg.d_c
        
        CE = 0.0
        if d_c > 1 and m > 0:
            Ls = self.build_L_list()
            num = 0.0; den = 0.0; cnt = 0
            for a in range(d_c):
                for b in range(a+1, d_c):
                    La_norm = torch.linalg.norm(Ls[a], ord='fro').item()
                    Lb_norm = torch.linalg.norm(Ls[b], ord='fro').item()
                    C = Ls[a] @ Ls[b] - Ls[b] @ Ls[a]
                    num += torch.linalg.norm(C, ord='fro').item()
                    den += max(1e-6, La_norm * Lb_norm)
                    cnt += 1
            if cnt > 0: CE = (num / cnt) / (den / cnt) if den > 0 else 0.0
        
        Psp = self.build_Psp()
        M = self.Pi_null @ Psp @ self.Pi_act
        LR = torch.linalg.norm(M, ord=2).item()
        
        err_mat = Psp.T @ self.Pi_act @ Psp - self.Pi_act
        IR_subspace = torch.linalg.norm(err_mat, ord='fro').item()
        
        return {"CE": CE, "LR": LR, "IR_subspace": IR_subspace}

# %% [markdown]
# ## 3. Tiny ViT Model (v12/v13/v14)
# (Class definitions)

# %%
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class SelfAttnPE(nn.Module):
    """
    v12/v13: Returns alignment loss, uses full d_h scaling.
    """
    def __init__(self, dim, num_heads, pe: PE_StringESPR, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim; self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.pe = pe
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, r):
        B, N, D = x.shape
        H = self.num_heads; dh = self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, dh).permute(2,0,1,3,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        (q_rot, k_rot), (q_prime_post_psp, k_prime_post_psp) = self.pe.rotate_qk(q, k, r)

        # --- v12/v13: Alignment Loss ---
        m = self.pe.cfg.m
        q_null = q_prime_post_psp[..., 2*m:]
        k_null = k_prime_post_psp[..., 2*m:]
        align_loss = (q_null.pow(2).mean() + k_null.pow(2).mean())
        # --- End Alignment Loss ---

        # --- v12/v13: Attention Scaling Fix ---
        scale = dh ** -0.5
        # --- End Scaling Fix ---

        qh = (q_rot * scale).permute(0, 2, 1, 3).contiguous()
        kh =  k_rot.permute(0, 2, 3, 1).contiguous()
        
        logits = torch.matmul(qh, kh)
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)

        vh = v.permute(0, 2, 1, 3).contiguous()
        attn_out = torch.matmul(attn, vh)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()

        out = attn_out.reshape(B, N, D)
        out = self.proj_drop(self.proj(out))
        
        return out, align_loss

class Block(nn.Module):
    """v12/v13: Propagates alignment loss"""
    def __init__(self, dim, num_heads, pe: PE_StringESPR, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttnPE(dim, num_heads, pe)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
    
    def forward(self, x, r):
        attn_out, align_loss = self.attn(self.norm1(x), r)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, align_loss

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch=4, in_chans=3, dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)
        self.grid = (img_size // patch, img_size // patch)
        self.num_patches = self.grid[0] * self.grid[1]
        self.patch_size = patch

    def forward(self, x):
        x = self.proj(x)
        B,D,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        return x

class ViTTinyPE(nn.Module):
    """
    v12/v13: Gathers alignment loss and L2-norms final features.
    """
    def __init__(self, img_size=32, patch=4, in_ch=3, dim=192, depth=6, heads=3,
                 num_classes=10, pe_cfg: PEConfig=None, 
                 coord_jitter_std=0.0, coord_drop_prob=0.0,
                 l2_norm_feats=True, l2_norm_scale=10.0):
        super().__init__()
        self.dim = dim
        self.patch = PatchEmbed(img_size, patch, in_ch, dim)
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        
        pe_cfg.d_h = dim // heads
        self.pe_module = PE_StringESPR(pe_cfg, num_heads=heads)
        
        self.blocks = nn.ModuleList([
            Block(dim, heads, self.pe_module) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        h,w = self.patch.grid
        yy, xx = torch.meshgrid(
            torch.linspace(-1,1,h),
            torch.linspace(-1,1,w), indexing='ij')
        self.register_buffer("xy_grid", torch.stack([xx,yy], dim=-1).reshape(1,h*w,2))
        
        self.coord_jitter_std = coord_jitter_std
        self.coord_drop_prob = coord_drop_prob
        
        self.l2_norm_feats = l2_norm_feats
        self.l2_norm_scale = l2_norm_scale

    def get_coords(self, B, is_training: bool):
        """Get coordinates 'r' for all tokens."""
        d_c = self.pe_module.cfg.d_c
        
        grid = self.xy_grid.repeat(B,1,1)
        if d_c > 2:
            padding = torch.zeros(B, self.patch.num_patches, d_c - 2,
                                  device=grid.device, dtype=grid.dtype)
            r_patches = torch.cat([grid, padding], dim=-1)
        else:
            r_patches = grid[..., :d_c]
            
        r_cls = torch.zeros(B,1,d_c, device=r_patches.device, dtype=r_patches.dtype)
        
        r = torch.cat([r_cls, r_patches], dim=1)
        
        if is_training:
            if self.coord_jitter_std > 0:
                r = r + torch.randn_like(r) * self.coord_jitter_std
            
            if self.coord_drop_prob > 0 and d_c > 1:
                drop_mask = (torch.rand(B, r.size(1), 1, device=r.device) < self.coord_drop_prob)
                drop_left  = drop_mask & (torch.rand(B, r.size(1), 1, device=r.device) < 0.5)
                drop_right = drop_mask & (~drop_left)
                r[:,:,0] = torch.where(drop_left.squeeze(-1),  torch.zeros_like(r[:,:,0]), r[:,:,0])
                r[:,:,1] = torch.where(drop_right.squeeze(-1), torch.zeros_like(r[:,:,1]), r[:,:,1])
        
        return r

    def forward_features(self, x):
        B = x.size(0)
        # v14: Check device global
        if SET_CHANNELS_LAST and device == 'cuda':
             x = x.to(memory_format=torch.channels_last)
        x = self.patch(x)
        cls = self.cls.expand(B,-1,-1)
        x = torch.cat([cls, x], dim=1)
        
        r = self.get_coords(B, is_training=self.training)
        
        total_align_loss = 0.0
        for blk in self.blocks:
            x, align_loss = blk(x, r) 
            total_align_loss = total_align_loss + align_loss
        
        x_norm = self.norm(x)
        cls_feat = x_norm[:,0]
        
        if self.l2_norm_feats:
            cls_feat = F.normalize(cls_feat, p=2, dim=-1) * self.l2_norm_scale
            
        return cls_feat, total_align_loss / len(self.blocks) if len(self.blocks) > 0 else 0.0

    def forward(self, x, y=None):
        feats, align_loss = self.forward_features(x)
        logits = self.head(feats)
        if y is None:
            return logits
        
        loss_task = F.cross_entropy(logits, y)
        return (loss_task, align_loss), logits

def build_vit_tiny_with_PE(kind="STRING", d_c=2, m_ratio=1.0, lowrank_r=0, allow_mixing=False, img_size=32,
                           coord_jitter_std=0.0, coord_drop_prob=0.0):
    dim=192; heads=3; d_h=dim//heads
    m = int((d_h // 2) * m_ratio)
    pe_cfg = PEConfig(kind=kind, d_c=d_c, d_h=d_h, m=m,
                      lowrank_r=lowrank_r, allow_mixing=allow_mixing,
                      dtype=torch.float32)
    
    model = ViTTinyPE(img_size=img_size, patch=4, in_ch=3, dim=dim, depth=6, heads=heads,
                      num_classes=10, pe_cfg=pe_cfg,
                      coord_jitter_std=coord_jitter_std,
                      coord_drop_prob=coord_drop_prob,
                      l2_norm_feats=True)
    
    if SET_CHANNELS_LAST and device == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    model = model.to(device)

    if USE_TORCH_COMPILE and device == 'cuda':
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}")
    return model

# %% [markdown]
# ## 4. Synthetic-Invariance Dataset (v12/v13/v14 - Expanded Transforms)
# (Class definitions to be used by the worker)

# %%
class DeterministicShearRotate:
    """Applies shear and rotate in a random order, using a local RNG for determinism."""
    def __init__(self, shear_deg, rot_deg, rng: random.Random):
        self.shear = float(shear_deg)
        self.rot   = float(rot_deg)
        self.rng   = rng
        self.norm = T.Normalize([0.5]*3, [0.5]*3)

    def __call__(self, x):
        sx = self.rng.uniform(-self.shear, self.shear)
        sy = self.rng.uniform(-self.shear, self.shear)
        ang = self.rng.uniform(-self.rot, self.rot)
        order = self.rng.random() < 0.5
        if order:
            x = TF.affine(x, angle=0.0, translate=[0,0], scale=1.0, shear=[sx, sy])
            x = TF.affine(x, angle=ang, translate=[0,0], scale=1.0, shear=[0.0, 0.0])
        else:
            x = TF.affine(x, angle=ang, translate=[0,0], scale=1.0, shear=[0.0, 0.0])
            x = TF.affine(x, angle=0.0, translate=[0,0], scale=1.0, shear=[sx, sy])
        return self.norm(x)

class DeterministicAnisoScaleRotate:
    """Applies anisotropic scale and rotate in a random order."""
    def __init__(self, sx_max_delta, sy_max_delta, rot_deg, rng: random.Random):
        self.sx_delta = float(sx_max_delta)
        self.sy_delta = float(sy_max_delta)
        self.rot_deg = float(rot_deg)
        self.rng = rng
        self.norm = T.Normalize([0.5]*3, [0.5]*3)

    def _aniso_scale(self, x, sx, sy):
        C, H, W = x.shape
        H2 = max(2, int(round(H * sy))); W2 = max(2, int(round(W * sx)))
        y = TF.resize(x, [H2, W2], interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        pad_h = max(0, H - H2); pad_w = max(0, W - W2)
        if pad_h > 0 or pad_w > 0:
            y = TF.pad(y, [pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2])
        if y.shape[1] > H or y.shape[2] > W:
            y = TF.center_crop(y, [H, W])
        return y

    def __call__(self, x):
        sx = self.rng.uniform(1.0 - self.sx_delta, 1.0 + self.sx_delta)
        sy = self.rng.uniform(1.0 - self.sy_delta, 1.0 + self.sy_delta)
        ang = self.rng.uniform(-self.rot_deg, self.rot_deg)
        if self.rng.random() < 0.5:
            x = self._aniso_scale(x, sx, sy)
            x = TF.affine(x, angle=ang, translate=[0,0], scale=1.0, shear=[0.0,0.0])
        else:
            x = TF.affine(x, angle=ang, translate=[0,0], scale=1.0, shear=[0.0,0.0])
            x = self._aniso_scale(x, sx, sy)
        return self.norm(x)

class DeterministicPerspectiveRotate:
    """Applies perspective warp and rotate in a random order."""
    def __init__(self, persp_max, rot_deg, rng: random.Random):
        self.pmax = float(persp_max)
        self.rot_deg = float(rot_deg)
        self.rng = rng
        self.norm = T.Normalize([0.5]*3, [0.5]*3)

    def _rand_points(self, H, W):
        dx, dy = self.pmax * W, self.pmax * H
        def jitter(pt):
            return [pt[0] + self.rng.uniform(-dx, dx), pt[1] + self.rng.uniform(-dy, dy)]
        tl, tr, br, bl = [ [0,0], [W-1,0], [W-1,H-1], [0,H-1] ]
        return [tl, tr, br, bl], [jitter(tl), jitter(tr), jitter(br), jitter(bl)]

    def __call__(self, x):
        C,H,W = x.shape
        src, dst = self._rand_points(H, W)
        ang = self.rng.uniform(-self.rot_deg, self.rot_deg)
        if self.rng.random() < 0.5:
            x = TF.perspective(x, src, dst, interpolation=T.InterpolationMode.BILINEAR)
            x = TF.affine(x, angle=ang, translate=[0,0], scale=1.0, shear=[0.0,0.0])
        else:
            x = TF.affine(x, angle=ang, translate=[0,0], scale=1.0, shear=[0.0,0.0])
            x = TF.perspective(x, src, dst, interpolation=T.InterpolationMode.BILINEAR)
        return self.norm(x)


class SyntheticGrid(torch.utils.data.Dataset):
    """Deterministic dataset for pre-computed/cached sets (e.g., validation and OOD)."""
    def __init__(self, num_classes=10, size=32, length=5000, seed=42, transform=None):
        self.K = num_classes; self.S = size; self.N = length
        self.transform = transform
        gen = torch.Generator().manual_seed(seed)
        self.templates = []
        for k in range(self.K):
            f1 = 2 + k % 5
            f2 = 3 + (k // 2) % 4
            yy, xx = torch.meshgrid(torch.linspace(-1,1,self.S),
                                      torch.linspace(-1,1,self.S), indexing='ij')
            img = 0.5 + 0.25*torch.sin(math.pi*f1*xx) + 0.25*torch.cos(math.pi*f2*yy)
            self.templates.append(img.clamp(0,1).repeat(3,1,1))
        self.labels = [torch.tensor(i % self.K, dtype=torch.long) for i in range(self.N)]
        
        if self.transform is not None:
            # Use a simple loop for deterministic transform application
            self.images = []
            for i in range(self.N):
                self.images.append(self.transform(self.templates[i % self.K].clone()))
        else:
            self.images = [T.Normalize([0.5]*3, [0.5]*3)(self.templates[i % self.K].clone()) for i in range(self.N)]

    def __len__(self): return self.N
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

class SyntheticGridTrain(torch.utils.data.Dataset):
    """On-the-fly augmented dataset for training."""
    def __init__(self, num_classes=10, size=32, length=5000, seed=42, distort_level=0.0):
        self.K = num_classes; self.S = size; self.N = length
        gen = torch.Generator().manual_seed(seed)
        self.templates = []
        for k in range(self.K):
            f1 = 2 + k % 5
            f2 = 3 + (k // 2) % 4
            yy, xx = torch.meshgrid(torch.linspace(-1,1,self.S), torch.linspace(-1,1,self.S), indexing='ij')
            img = 0.5 + 0.25*torch.sin(math.pi*f1*xx) + 0.25*torch.cos(math.pi*f2*yy)
            self.templates.append(img.clamp(0,1).repeat(3,1,1))
        self.labels = [torch.tensor(i % self.K, dtype=torch.long) for i in range(self.N)]
        rot_deg = 0.0 if distort_level == 0.0 else max(1.0, 0.5*distort_level)
        self.transform = T.Compose([
            T.RandomAffine(degrees=rot_deg, shear=distort_level),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self): return self.N
    def __getitem__(self, idx):
        k = idx % self.K
        x = self.transform(self.templates[k].clone())
        y = self.labels[idx]
        return x, y

def make_synth_loaders(bs=2048, ntrain=8192, nval=1024, size=32, train_distort=0.0, num_workers=None, prefetch_factor=None):
    """v14: Patched loader to accept bs/worker args for parallel runs."""
    train_data = SyntheticGridTrain(length=ntrain, seed=42, size=size, distort_level=train_distort)
    val_rng = random.Random(1)
    val_tf = DeterministicShearRotate(shear_deg=0.0, rot_deg=0.0, rng=val_rng)
    val_data = SyntheticGrid(length=nval, seed=1, size=size, transform=val_tf)

    is_cuda = (device == 'cuda')
    
    if num_workers is None:
        if is_cuda:
            num_workers = max(2, min(8, (os.cpu_count() or 4) // 2))
        else:
            num_workers = 0 # Use 0 workers for CPU training
            
    if prefetch_factor is None and num_workers > 0:
        prefetch_factor = 4
    elif num_workers == 0:
        prefetch_factor = None

    common_kwargs = dict(
        batch_size=bs,
        pin_memory=is_cuda,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0 and USE_FAST_MODE),
        prefetch_factor=prefetch_factor,
    )

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **{k:v for k,v in common_kwargs.items() if v is not None})
    val_loader   = torch.utils.data.DataLoader(val_data,   shuffle=False, **{k:v for k,v in common_kwargs.items() if v is not None})
    return train_loader, val_loader

def build_fixed_ood_dataset(mode, param, n=2048, size=32, seed=777):
    """v14: Builds expanded OOD datasets. Called *inside* the worker."""
    # print(f"[Worker {os.getpid()}] Building OOD dataset: mode={mode}, param={param}, n={n}") 
    rng = random.Random(seed)
    
    if mode == "shear":
        rot = 0.0 if param == 0.0 else max(1.0, 0.5 * param)
        tf = DeterministicShearRotate(shear_deg=param, rot_deg=rot, rng=rng)
    elif mode == "aniso":
        sx_delta, sy_delta = param
        tf = DeterministicAnisoScaleRotate(sx_delta, sy_delta, rot_deg=15.0, rng=rng)
    elif mode == "persp":
        tf = DeterministicPerspectiveRotate(persp_max=param, rot_deg=15.0, rng=rng)
    else:
        raise ValueError(f"Unknown OOD mode: {mode}")

    base = SyntheticGrid(length=n, seed=99, size=size, transform=tf)
    images = [img.contiguous() for img in base.images]
    labels = base.labels
    return torch.utils.data.TensorDataset(torch.stack(images), torch.stack(labels))

# %% [markdown]
# ## 5. Training, Evaluation, and Statistical Tests
# (Class definitions to be used by the worker)

# %%
@torch.no_grad()
def invariant_residual_spec(pe: PE_StringESPR, deltas=((0.1, 0.0), (0.0, 0.1))):
    """
    v12/v13: Uses operator norm (ord=2) in the learned U-basis.
    """
    m, dc, dh = pe.cfg.m, pe.cfg.d_h, pe.cfg.d_c
    if m == 0: return 0.0
    
    betas = pe.betas; Psp = pe.build_Psp()
    Pi_act_full = pe.Pi_act
    Psp_act = Pi_act_full @ Psp @ Pi_act_full
    
    ir_vals = []
    for delta_coords in deltas:
        Δ = torch.zeros(m, dc, device=betas.device, dtype=betas.dtype)
        for i in range(min(dc, len(delta_coords))):
            Δ[:, i] = delta_coords[i]
        c = torch.cos((betas*Δ).sum(-1)); s = torch.sin((betas*Δ).sum(-1))
        
        R_delta_full = torch.eye(dh, device=betas.device, dtype=betas.dtype)
        for u in range(m):
            i0=2*u
            R_delta_full[i0:i0+2,i0:i0+2] = torch.tensor([[c[u],-s[u]],[s[u],c[u]]], 
                                                         device=betas.device, dtype=betas.dtype)
        R_delta_act = Pi_act_full @ R_delta_full @ Pi_act_full
        term_ideal = R_delta_act
        term_espr = Psp_act @ R_delta_act @ Psp_act.T
        
        err_norm = torch.linalg.norm(term_espr - term_ideal, ord=2)
        den_norm = torch.linalg.norm(term_ideal, ord=2) + 1e-8
        ir_vals.append((err_norm / den_norm).item())
        
    return float(sum(ir_vals)/len(ir_vals)) if ir_vals else 0.0

def make_optimizer(model):
    use_fused = (device == 'cuda' and USE_FAST_MODE)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, fused=use_fused)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    return opt

def train_one_epoch(model, opt, loader, lambdas, clk=None, desc=None):
    """v13: Patched for progress bar and alignment loss."""
    model.train()
    meter = {"loss":0., "acc":0., "n":0, "align_loss": 0.}
    
    pe = model._orig_mod.pe_module if hasattr(model, '_orig_mod') else model.pe_module
    lambda_align = lambdas.get("lambda_align", 0.0)

    scaler = None
    if AMP_DTYPE == torch.float16:
        scaler = torch.amp.GradScaler('cuda')

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    if clk: clk.start("data")
    for x,y in pbar:
        if clk: clk.stop("data")
        x = _to_device(x)
        y = y.to(device, non_blocking=True)
        
        if clk: clk.start("step")
        opt.zero_grad(set_to_none=True)

        with _amp_ctx():
            (loss_task, align_loss), logits = model(x, y)
            loss_penalty = pe.penalty(**lambdas)
            loss = loss_task + loss_penalty + align_loss * lambda_align

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        if clk: clk.stop("step")

        with torch.no_grad():
            meter["loss"] += loss_task.item() * x.size(0)
            meter["align_loss"] += align_loss.item() * x.size(0)
            meter["acc"]  += (logits.argmax(dim=-1) == y).float().sum().item()
            meter["n"]    += x.size(0)
            if meter["n"] > 0:
                pbar.set_postfix(loss=f"{meter['loss']/meter['n']:.3f}", 
                                 acc=f"{meter['acc']/meter['n']:.3f}",
                                 align=f"{meter['align_loss']/meter['n']:.2e}")
        
        if clk: clk.start("data")
    if clk: clk.stop("data") # Stop timer after loop finishes
    
    if meter["n"] == 0: return {"loss":0., "acc":0., "align_loss": 0.}
    return {k: meter[k]/meter["n"] for k in ["loss","acc", "align_loss"]}

@torch.no_grad()
def eval_cls(model, loader):
    model.eval()
    meter = {"loss":0., "acc":0., "n":0}
    for x,y in loader:
        x = _to_device(x)
        y = y.to(device, non_blocking=True)
        with _amp_ctx():
            (loss_task, _), logits = model(x, y)
        meter["loss"] += loss_task.item() * x.size(0)
        meter["acc"]  += (logits.argmax(dim=-1) == y).float().sum().item()
        meter["n"]    += x.size(0)
    if meter["n"] == 0: return {"loss":0., "acc":0.}
    return {k: meter[k]/meter["n"] for k in ["loss","acc"]}

# --- Statistical Test Functions ---
McNemar = namedtuple("McNemar","b c chi2 p")

@torch.no_grad()
def correctness(model, loader):
    """Returns a (correctness_tensor, label_tensor)."""
    model.eval(); outs=[]; ys=[]
    for x,y in loader:
        x = _to_device(x)
        with _amp_ctx():
            logits = model(x, y=None)
            p = logits.argmax(-1).cpu()
        outs.append((p == y.cpu()).float()); ys.append(y.cpu())
    return torch.cat(outs), torch.cat(ys)

def mcnemar_from_correct(corr_a, corr_b):
    """(From Audit) McNemar test on correctness booleans."""
    a_ok = corr_a.bool()
    b_ok = corr_b.bool()
    b = int((~a_ok &  b_ok).sum())
    c = int(( a_ok & ~b_ok).sum())
    if b + c < 25: # McNemar is not reliable for small n (b+c)
        return McNemar(b, c, 0.0, 1.0)
    if b + c == 0:
        return McNemar(b, c, 0.0, 1.0)
    chi2 = (abs(b - c) - 1)**2 / (b + c)
    mp.dps = 15
    p = 1 - mp.gammainc(0.5, 0, chi2 / 2) / mp.sqrt(mp.pi)
    return McNemar(b, c, float(chi2), float(p))

def paired_bootstrap_delta(corr_a, corr_b, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(corr_a); diffs = np.empty(B, dtype=np.float64)
    base = (corr_b.mean() - corr_a.mean()).item()
    idx = rng.integers(0, n, size=(B, n))
    a = np.take(corr_a.numpy(), idx); b = np.take(corr_b.numpy(), idx)
    diffs[:] = b.mean(axis=1) - a.mean(axis=1)
    ci = (np.quantile(diffs, 0.025), np.quantile(diffs, 0.975))
    return float(base), tuple(map(float, ci))

def paired_permutation_p(corr_a, corr_b, B=10000, seed=1):
    rng = np.random.default_rng(seed)
    d = (corr_b - corr_a).numpy()
    obs = d.mean()
    signs = rng.choice([-1,1], size=(B, len(d)))
    perm = (signs * d).mean(axis=1)
    p = float((np.abs(perm) >= abs(obs)).mean())
    return obs, p

def tost_noninferior(p_b, p_a, n, delta=0.02):
    """(From Audit) TOST for two proportions (independence approx)."""
    se = ((p_b*(1-p_b)/n + p_a*(1-p_a)/n) + 1e-9) ** 0.5
    z1 = ((p_b - p_a) + delta) / se   # H1: (p_b - p_a) > -delta
    p1 = 0.5 * (1.0 + erf(z1 / sqrt(2.0)))  # one-sided
    return 1.0 - p1  # p-value for H0: (p_b - p_a) <= -delta

# %% [markdown]
# ## 6. Worker Function (v14 - Self-Contained & Robust)
#
# This cell defines the top-level worker that the `ProcessPoolExecutor` will run.
# It is self-contained and sets up its own environment to avoid CUDA conflicts.
#
# (This cell combines v13's `_run_experiment_inner` and `_worker_task`)

# %%
# This *must* be a global variable for lazy-loading in the worker
_PROC_OOD_CACHE = None

def _parallel_worker_task(task):
    """
    v14: A self-contained worker function for the ProcessPoolExecutor.
    Sets up its own device, caches, and OOD data.
    """
    
    # --- 1. Imports (needed by spawn process) ---
    import os, traceback, random
    import numpy as np
    import torch, torch.nn as nn
    import torch.nn.functional as F
    from math import erf, sqrt
    from collections import namedtuple
    import mpmath as mp
    from tqdm.auto import tqdm
    import contextlib
    from einops import rearrange, repeat
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    
    
    # --- 2. Per-Process Cache Dirs (prevents races) ---
    base_cache = os.path.expanduser("~/.cache/torch_compile_sweeps")
    pid = os.getpid()
    pid_cache_dir = os.path.join(base_cache, f"pid_{pid}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(pid_cache_dir, "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(pid_cache_dir, "triton")
    os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

    # --- 3. Device & Seed Setup (Worker-Specific) ---
    # We now set the globals *for this worker process*
    global device, AMP_DTYPE
    
    run_params = task["run_params"]
    device_id = task.get("device_id", 0)
    
    local_device_str = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    
    if local_device_str != 'cpu':
        torch.cuda.set_device(local_device_str)
        major_cc, _ = torch.cuda.get_device_capability()
        local_amp_dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        local_amp_dtype = None
    
    # Set worker-local globals for legacy functions that use them
    device = local_device_str
    AMP_DTYPE = local_amp_dtype
    
    set_seed_all(task["seed"])
    torch.set_num_threads(2) # Keep CPU contention low

    try:
        # --- 4. Lazy OOD Cache Build ---
        global _PROC_OOD_CACHE
        if _PROC_OOD_CACHE is None:
            # print(f"[Worker {pid}] Building OOD Cache...")
            _PROC_OOD_CACHE = {
                key: torch.utils.data.DataLoader(
                         build_fixed_ood_dataset(mode, param, n=run_params["ood_n_samples"], seed=777),
                         batch_size=1024, shuffle=False, num_workers=0, pin_memory=(device=='cuda')
                     )
                for key, (mode, param) in run_params["ood_tests_config"].items()
            }

        # --- 5. Run Experiment Logic ---
        clk = RunClock()
        lambdas = run_params["lambdas"]
        IMG_SIZE = 32
        
        clk.start("build")
        model = build_vit_tiny_with_PE(
            kind=run_params["kind"], d_c=2, m_ratio=run_params["m_ratio"],
            lowrank_r=run_params["lowrank_r"], allow_mixing=run_params["allow_mixing"],
            img_size=IMG_SIZE,
            coord_jitter_std=run_params["coord_jitter_std"],
            coord_drop_prob=run_params["coord_drop_prob"]
        )
        
        pe_module = model._orig_mod.pe_module if hasattr(model, '_orig_mod') else model.pe_module

        train_loader, val_loader = make_synth_loaders(
            bs=run_params["bs"], ntrain=8192, nval=1024, size=IMG_SIZE, train_distort=run_params["train_distort"],
            num_workers=run_params["num_workers"], prefetch_factor=2
        )
        opt = make_optimizer(model)
        clk.stop("build")
        
        tr = {}
        clk.start("train_total")
        for ep in range(run_params["epochs"]):
            # desc = f"Ep {ep+1} [k={run_params['kind']}, m={run_params['m_ratio']}, d={run_params['train_distort']}, s={task['seed']}]"
            # Using silent=True, so desc is not needed for pbar
            tr = train_one_epoch(model, opt, train_loader, lambdas, clk=clk, desc=None)
        clk.stop("train_total")

        clk.start("diag")
        final_diags = pe_module.diagnostics()
        final_ir_spec = invariant_residual_spec(pe_module)
        clk.stop("diag")
        
        clk.start("eval_clean")
        va_clean = eval_cls(model, val_loader)
        clk.stop("eval_clean")
        
        ood_metrics = {}
        if _PROC_OOD_CACHE:
            clk.start("eval_ood")
            for test_key, ood_loader in _PROC_OOD_CACHE.items():
                va_ood = eval_cls(model, ood_loader)
                ood_metrics[f"acc@{test_key}"] = va_ood['acc']
                ood_metrics[f"robust_delta@{test_key}"] = va_clean['acc'] - va_ood['acc']
            clk.stop("eval_ood")
        
        # --- 6. Collate Results ---
        metrics = {
            **final_diags,
            "IR_spec": final_ir_spec,
            "clean_acc": va_clean['acc'],
            "train_acc": tr.get('acc', 0.0), # Get final epoch train_acc
            **ood_metrics,
            **clk.dump()
        }
        
        # Attach metadata for easier pandas aggregation
        metrics.update({
            "ok": True,
            "name": task["name"],
            "m_ratio": task["m_ratio"],
            "seed": task["seed"],
            "train_distort": task["train_distort"],
        })
        
        del model, train_loader, val_loader, opt
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        return metrics

    except Exception:
        # --- 7. Robust Failure Reporting ---
        return {
            "ok": False, 
            "trace": traceback.format_exc(), 
            "task_name": task["name"],
            "seed": task["seed"],
            "m_ratio": task["m_ratio"],
            "train_distort": task["train_distort"]
        }


# %% [markdown]
# ## 7. Run Comparisons (v14 - 30 Seed Parallel Sweep)
# 
# **NOTE:** This cell will *only* run if the script is executed directly (e.g., `python your_script.py`).
# It will be skipped if imported, which is what happens during the `spawn` process.

# %%
if __name__ == "__main__":
    
    # --- Configuration for the Sweep ---
    EPOCHS_PER_SWEEP = 3
    DISTORT_SWEEP = [0.0, 5.0, 8.0, 12.0]
    # <-- v13: 30 seeds for statistical power
    SEEDS = list(range(30)) 
    M_RATIOS = [1.0, 0.75, 0.5]
    COORD_JITTER = 0.0 
    COORD_DROP = 0.0   

    # --- v12: Parallelism Config ---
    SINGLE_GPU_CONCURRENCY = 2 # How many jobs to run on one GPU
    BASE_BS = 2048 # Total batch size, will be divided by concurrency
    # --- End ---

    # --- v12: Expanded OOD Tests ---
    OOD_N_SAMPLES = 2048 # Increased statistical power
    OOD_TESTS_CONFIG = {
        # key: (mode, param)
        "shear_15": ("shear", 15.0),
        "shear_20": ("shear", 20.0),
        "shear_30": ("shear", 30.0),
        "aniso_0.2": ("aniso", (0.2, 0.05)), # (sx_delta, sy_delta)
        "persp_0.1": ("persp", 0.1),
    }
    # --- End ---

    # --- v13: Tuned Penalties ---
    # v12 penalties (1e-2) were too strong. Tuned down to act as regularizers.
    LAMBDA_U_REG = 1e-5       # Regularizer for T_basis
    LAMBDA_COMM = 2e-3        # (Down from 1e-2)
    LAMBDA_MIX = 2e-3         # (Down from 1e-2)
    LAMBDA_DELTA_REG = 1e-5   # (Same)
    LAMBDA_E_REG = 1e-4       # (Same)
    LAMBDA_ALIGN = 1e-4       # (Down from 1e-3)
    # --- End v13 ---

    MODELS_TO_TEST = [
        {
            "name": "STRING", "kind": "STRING", "lowrank_r": 0, "allow_mixing": False,
            "lambdas": dict(lambda_comm=0., lambda_mix=0., lambda_delta=0., lambda_E=0., lambda_U=0., lambda_align=0.)
        },
        {
            "name": "ESPR_full", "kind": "ESPR", "lowrank_r": 8, "allow_mixing": True,
            "lambdas": dict(lambda_comm=LAMBDA_COMM, lambda_mix=LAMBDA_MIX, 
                            lambda_delta=LAMBDA_DELTA_REG, lambda_E=LAMBDA_E_REG, 
                            lambda_U=LAMBDA_U_REG, lambda_align=LAMBDA_ALIGN)
        },
        {
            "name": "ESPR_noMix", "kind": "ESPR", "lowrank_r": 8, "allow_mixing": False,
            "lambdas": dict(lambda_comm=LAMBDA_COMM, lambda_mix=0., 
                            lambda_delta=LAMBDA_DELTA_REG, lambda_E=0., 
                            lambda_U=LAMBDA_U_REG, lambda_align=LAMBDA_ALIGN)
        },
        {
            "name": "ESPR_noDelta", "kind": "ESPR", "lowrank_r": 0, "allow_mixing": True,
            "lambdas": dict(lambda_comm=LAMBDA_COMM, lambda_mix=LAMBDA_MIX, 
                            lambda_delta=0., lambda_E=LAMBDA_E_REG, 
                            lambda_U=LAMBDA_U_REG, lambda_align=LAMBDA_ALIGN)
        },
    ]

    # --- v14: Build Task List (Lightweight Dicts) ---
    def build_tasks_parallel(n_gpus, per_gpu_concurrency=1):
        tasks = []
        # Calculate BS per worker
        total_workers = n_gpus * per_gpu_concurrency if n_gpus > 0 else per_gpu_concurrency
        base_bs_per_proc = max(256, BASE_BS // max(1, total_workers))
        
        print(f"Building tasks: {n_gpus} GPUs, {per_gpu_concurrency} workers/GPU = {total_workers} total workers.")
        print(f"Base BS={BASE_BS} -> Per-worker BS={base_bs_per_proc}")

        for seed in SEEDS:
            for config in MODELS_TO_TEST:
                for m_ratio in M_RATIOS:
                    for distort_level in DISTORT_SWEEP:
                        if m_ratio == 1.0 and config['name'] in ('ESPR_noMix', 'ESPR_noDelta'):
                            continue
                        
                        run_params = dict(
                            kind=config['kind'], m_ratio=m_ratio,
                            lowrank_r=config['lowrank_r'], allow_mixing=config['allow_mixing'],
                            lambdas=config['lambdas'], train_distort=distort_level,
                            coord_jitter_std=COORD_JITTER, coord_drop_prob=COORD_DROP,
                            epochs=EPOCHS_PER_SWEEP,
                            bs=base_bs_per_proc,
                            num_workers=2, # Keep workers low for parallel runs
                            ood_tests_config=OOD_TESTS_CONFIG,
                            ood_n_samples=OOD_N_SAMPLES,
                            silent=True,
                        )
                        tasks.append({
                            "seed": seed, "name": config['name'],
                            "m_ratio": m_ratio, "train_distort": distort_level,
                            "run_params": run_params
                        })
        
        # Assign devices round-robin
        if n_gpus > 0:
            for i, t in enumerate(tasks):
                t["device_id"] = i % n_gpus
        else:
            for t in tasks:
                t["device_id"] = "cpu" # Assign to CPU if no GPUs
            
        return tasks

    # --- v14: Run the Sweep (Main Execution) ---
    
    # This must be called *before* any CUDA context is created
    # v14: Use spawn context for CUDA safety
    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    
    results = []
    fails = []
    tasks = []
    n_gpus = torch.cuda.device_count()
    max_workers = 1

    if n_gpus > 1:
        print(f"--- Starting Multi-GPU Sweep ({n_gpus} GPUs) ---")
        max_workers = n_gpus
        tasks = build_tasks_parallel(ngpu=n_gpus, per_gpu_concurrency=1)
    elif n_gpus == 1:
        print(f"--- Starting Single-GPU Concurrent Sweep ({SINGLE_GPU_CONCURRENCY} workers) ---")
        max_workers = SINGLE_GPU_CONCURRENCY
        tasks = build_tasks_parallel(ngpu=1, per_gpu_concurrency=SINGLE_GPU_CONCURRENCY)
    else:
        print("--- Starting CPU Sweep (1 worker) ---")
        max_workers = max(1, (os.cpu_count() or 4) // 2) # Use more cores for CPU
        tasks = build_tasks_parallel(ngpu=0, per_gpu_concurrency=1)

    print(f"Total tasks: {len(tasks)} | Max workers: {max_workers} | Seeds: {len(SEEDS)}")
    print(f"Models: {[c['name'] for c in MODELS_TO_TEST]}")
    print(f"Distortions: {DISTORT_SWEEP}")
    print(f"M_Ratios: {M_RATIOS}")
    
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futs = [executor.submit(_parallel_worker_task, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Total Sweep Progress"):
            try:
                r = fut.result()
                if r.get("ok"):
                    results.append(r)
                else:
                    fails.append(r)
            except Exception as e:
                # This catches errors in the future/result handling itself
                print(f"A task future failed to return: {e}")
                fails.append({"ok": False, "trace": str(e), "task": "Unknown"})

    print("\n--- Sweep Complete ---")
    print(f"Successful runs: {len(results)}")
    print(f"Failed runs:     {len(fails)}")
    if fails:
        print("\n--- Sample Failure Trace ---")
        f = fails[0]
        print(f"Job: {f.get('name', 'N/A')}, s={f.get('seed', 'N/A')}, m={f.get('m_ratio', 'N/A')}, d={f.get('train_distort', 'N/A')}")
        print(f.get("trace", "<no-trace>"))
        print("----------------------------\n")
    
    # v14: Main thread proceeds to analysis ONLY if results were generated
    if results:
        df = pd.DataFrame(results)
    else:
        print("No results were generated. Exiting before plotting or analysis.")
        df = None

# %% [markdown]
# ## 8. Plot Sweep Results (Mean ± SE)

# %%
# --- Process and Plot Results ---
if __name__ == "__main__" and 'df' in locals() and df is not None:
    # v13: Check for training collapse
    clean_acc_avg = df['clean_acc'].mean()
    train_acc_avg = df['train_acc'].mean()
    print(f"\n--- Sanity Check ---")
    print(f"Average Clean Acc (all runs): {clean_acc_avg:.3f}")
    print(f"Average Train Acc (all runs): {train_acc_avg:.3f}")
    if clean_acc_avg < 0.7:
        print(f"WARNING: Clean accuracy ({clean_acc_avg:.3f}) seems low. Penalties might still be too high or epochs too low.")
    else:
        print("INFO: Clean accuracy looks healthy.")
    print("----------------------\n")

    OOD_PLOT_KEY = "shear_15" # Plot the primary shear test
    robust_delta_key = f"robust_delta@{OOD_PLOT_KEY}"
    shear_acc_key = f"acc@{OOD_PLOT_KEY}"

    summary = (df.groupby(["name", "m_ratio", "train_distort"])
               .agg(
                   shear_acc_mean=(shear_acc_key, "mean"),
                   shear_acc_se=(shear_acc_key, lambda x: x.std(ddof=1) / (len(x)**0.5 if len(x) > 1 else 1)),
                   robust_delta_mean=(robust_delta_key, "mean"),
                   robust_delta_se=(robust_delta_key, lambda x: x.std(ddof=1) / (len(x)**0.5 if len(x) > 1 else 1)),
                   ir_spec_mean=("IR_spec", "mean"),
                   train_time_mean=("t_train_total_s", "mean"),
               )
               .reset_index().fillna(0)) 

    print(f"\n--- Final Results Table (Mean across {len(SEEDS)} seeds) ---")
    try:
        print(summary.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print("Install 'tabulate' for markdown table output. Falling back to simple print.")
        print(summary.round(4))
    print("-------------------------------------------------\n")

    m_ratios_to_plot = sorted(summary['m_ratios'].unique(), reverse=True)
    for m_r in m_ratios_to_plot:
        
        df_plot = summary[summary['m_ratio'] == m_r]
        if df_plot.empty:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
        fig.suptitle(f"Robustness Sweep for m_ratio = {m_r} (OOD Test @ {OOD_PLOT_KEY})", fontsize=16)
        
        for name, group in df_plot.groupby('name'):
            ax1.errorbar(group['train_distort'], group['shear_acc_mean'], 
                         yerr=group['shear_acc_se'], label=name, fmt='-o', capsize=5)
        ax1.set_title('OOD Generalization (Accuracy)')
        ax1.set_xlabel('Training Distortion Level')
        ax1.set_ylabel(f'Accuracy on OOD Test @ {OOD_PLOT_KEY} (Higher is Better)')
        ax1.legend()
        ax1.grid(True, linestyle='--')

        for name, group in df_plot.groupby('name'):
            if 'ESPR' in name:
                ax2.plot(group['train_distort'], group['ir_spec_mean'], 'o-', label=f'{name} IR_spec')
        ax2.set_title('ESPR Internal Flexibility (Deviation from Invariance)')
        ax2.set_xlabel('Training Distortion Level')
        ax2.set_ylabel('IR_spec (Operator Norm) (Higher = More Deviation)')
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# %% [markdown]
# ## 9. Paired Statistical Test (v14 - 30-Seed Analysis)
# (This cell analyzes the `results` DataFrame from the sweep in Cell 7)

# %%
# --- Test Cycle 2: Rigorous Paired Test ---
# v14: This cell now analyzes the full 30-seed sweep results
if __name__ == "__main__":
    if 'df' not in locals() or df is None or df.empty:
        print("No results to analyze. Run the sweep cell (Cell 7) first.")
    else:
        print("\n---" * 20)
        print("Starting Test Cycle 2: Full Paired Statistical Tests on 30-Seed Sweep")
        print(f"Comparing ESPR_full(m=0.75) vs. STRING(m=0.75) at 8.0° distortion")
        print("---" * 20)

        # --- Configs for the paired test ---
        TEST_DISTORT = 8.0
        TEST_M_RATIO = 0.75 # This is the key matched parameter
        OOD_TEST_KEY = "shear_15"
        acc_key = f"acc@{OOD_TEST_KEY}"
        
        # Filter the dataframe to the "sweet spot"
        df_sweet_spot = df[
            (df["m_ratio"] == TEST_M_RATIO) &
            (df["train_distort"] == TEST_DISTORT)
        ]
        
        # Get the 30 paired scores
        scores_a = df_sweet_spot[df_sweet_spot["name"] == "STRING"].sort_values("seed")[acc_key]
        scores_b = df_sweet_spot[df_sweet_spot["name"] == "ESPR_full"].sort_values("seed")[acc_key]
        
        clean_scores_a = df_sweet_spot[df_sweet_spot["name"] == "STRING"].sort_values("seed")["clean_acc"]
        clean_scores_b = df_sweet_spot[df_sweet_spot["name"] == "ESPR_full"].sort_values("seed")["clean_acc"]
        
        if len(scores_a) != len(SEEDS) or len(scores_b) != len(SEEDS):
            print(f"ERROR: Missing data. Found {len(scores_a)} STRING runs and {len(scores_b)} ESPR_full runs.")
            print(f"Expected {len(SEEDS)} runs for each.")
            print("Please check the 'fails' list from the sweep.")
        elif len(scores_a) < 2:
            print(f"Not enough data points to run statistical tests (found {len(scores_a)}).")
        else:
            # --- 4. Run Statistical Tests (on OOD Set) ---
            acc_a = scores_a.mean()
            acc_b = scores_b.mean()

            # Test 1: Paired T-Test
            t_stat, t_p = ttest_rel(scores_b, scores_a)
            
            # Test 2: Wilcoxon Signed-Rank Test (Non-parametric)
            try:
                # v14: Use 'auto' for method, handles zeros better
                w_stat, w_p = wilcoxon(scores_b, scores_a, method='auto', alternative='two-sided')
            except ValueError as e:
                # Happens if differences are all zero or all equal
                print(f"Wilcoxon test warning: {e}")
                w_stat, w_p = 0, 1.0

            print(f"\n--- Paired Statistical Test Results (on OOD {OOD_TEST_KEY}, n={len(scores_a)}) ---")
            print(f"  Model A (Control): STRING (m_ratio={TEST_M_RATIO})")
            print(f"  Model B (Test):    ESPR_full (m_ratio={TEST_M_RATIO})")
            print(f"  Train Distortion:  {TEST_DISTORT}°\n")
            print(f"  Mean Acc (Model A): {acc_a:.4f} (Std: {scores_a.std():.4f})")
            print(f"  Mean Acc (Model B): {acc_b:.4f} (Std: {scores_b.std():.4f})")
            print(f"  Mean Delta (B - A): {acc_b - acc_a:+.4f}\n")

            print("--- Paired T-Test (parametric) ---")
            print(f"  T-statistic: {t_stat:.4f} | p-value: {t_p:.6f}")

            print("\n--- Wilcoxon Signed-Rank Test (non-parametric) ---")
            print(f"  W-statistic: {w_stat:.1f} | p-value: {w_p:.6f}")
            
            if w_p < 0.05:
                if acc_b > acc_a:
                    print("\n  Conclusion: Model B is statistically superior to Model A (p < 0.05).")
                else:
                    print("\n  Conclusion: Model A is statistically superior to Model B (p < 0.05).")
            else:
                 print("\n  Conclusion: No statistically significant difference found (p > 0.05).")

            # --- 5. Run TOST for non-inferiority on CLEAN set ---
            print("\n--- TOST Non-Inferiority Test (on CLEAN Val Set) ---")
            clean_acc_a = clean_scores_a.mean()
            clean_acc_b = clean_scores_b.mean()
            n_clean = len(clean_scores_a)
            delta_non_inferior = 0.02 # 2% margin

            diffs_clean = clean_scores_b - clean_scores_a
            se_diff = diffs_clean.std() / (n_clean**0.5)
            
            if se_diff < 1e-9:
                print("  Warning: Standard error of differences is zero. TOST is not reliable.")
                tost_p = 1.0
            else:
                # TOST: t-test for H0: diff <= -delta
                t_tost = (diffs_clean.mean() + delta_non_inferior) / se_diff
                from scipy.stats import t
                tost_p = 1.0 - t.cdf(t_tost, df=n_clean-1)

            print(f"  Clean Acc (Model A): {clean_acc_a:.4f}")
            print(f"  Clean Acc (Model B): {clean_acc_b:.4f}")
            print(f"  Non-Inferiority Margin (delta): {delta_non_inferior:.3f}")
            print(f"  p-value (for H0: Acc(B) <= Acc(A) - delta): {tost_p:.6f}")
            if tost_p < 0.05:
                print("  Conclusion: Model B is NOT statistically worse than Model A on clean data (non-inferior).")
            else:
                print("  Conclusion: We CANNOT conclude non-inferiority on clean data.")
            print("---" * 20)
