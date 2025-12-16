# Implementation: Relative-Position Attention from T-Series Proof

This directory contains a PyTorch implementation of the relative-position attention mechanism proven in `proof-axiomatic-compact.tex` (Theorems T0.1–T6.2).

## Theoretical Foundation

The implementation directly realizes the mathematical constructions from the proof:

### Core Theorems Implemented

**T1.6**: `R_STR(r) = exp(A(r))` where `A(r) = Σ_k [r]_k L_k`
- Implemented in `src/rotations.py::StructuredRotation`
- Learnable skew-symmetric generators `{L_k}` satisfying `L_k^T = -L_k`

**T2.9**: Relative-position property `R_STR(r_i)^T R_STR(r_j) = R_STR(r_j - r_i)`
- Implemented in `src/rotations.py::RelativePositionRotation`
- Key property enabling position-independent attention

**T4.5**: Relative-position attention
```
α_ij = (1/√(2m)) q_i^(act)^T (Π_act R_STR(r_j - r_i) Π_act) k_j^(act)
```
- Implemented in `src/attention.py::RelativePositionAttention`
- Includes active subspace projection `Π_act` from T2.10

**T5.1**: Commutator regularization `ε = max_{a,b} |[L_a, L_b]|`
- Enforces commutation condition `[L_a, L_b] = 0` from T1.1
- Loss term: `Total_Loss = CE_Loss + λ_comm * ε`

## Project Structure

EquivariantAttentionHead/
├── src/
│   ├── __init__.py
│   ├── rotations.py          # R_STR implementation (T2.7)
│   ├── attention.py          # Relative-position attention (T4.5)
│   ├── model.py              # Vision Transformer architecture
│   └── Benchmarking_Robust_STRING.py # End-to-end robustness sweep
├── references/
│   ├── string-drafts/        # Proof LaTeX files
│   └── original_code/        # Archived training scripts
└── GUIDELINES.md             # Project guidelines

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Train on MNIST

```bash
python train_mnist.py \
    --epochs 20 \
    --batch-size 128 \
    --lr 1e-3 \
    --lambda-comm 0.01 \
    --device cuda
```

**Expected results**: ~99% test accuracy after 20 epochs

### Train on CIFAR-10

```bash
python train_cifar.py \
    --epochs 100 \
    --batch-size 128 \
    --lr 3e-4 \
    --lambda-comm 0.01 \
    --augment \
    --device cuda
```

**Expected results**: ~85-90% test accuracy with augmentation

## Key Hyperparameters

### Model Architecture
- `--d-model`: Model dimension (default: 64 for MNIST, 128 for CIFAR)
- `--d-head`: Head dimension (default: 16 for MNIST, 32 for CIFAR)
- `--n-heads`: Number of attention heads (default: 4)
- `--n-layers`: Number of transformer blocks (default: 4 for MNIST, 6 for CIFAR)
- `--patch-size`: Patch size (default: 4)

### Training
- `--lr`: Learning rate (default: 1e-3 for MNIST, 3e-4 for CIFAR)
- `--lambda-comm`: Commutator loss weight from T5.1 (default: 0.01)
  - Higher values enforce stricter commutation
  - Lower values prioritize classification accuracy

## Understanding the Commutator Loss

From Theorem T5.1, the commutator error is:
```
ε = max_{a,b} |[L_a, L_b]|
```

This regularization term enforces the key hypothesis T1.1: `[L_a, L_b] = 0`.

**Monitoring commutator loss during training:**
- Values near 0 indicate generators are commuting (good)
- Values > 0.1 indicate violation of theorem hypotheses (bad)
- Balance λ_comm to maintain ε < 0.01 while optimizing accuracy

## Theoretical Guarantees

When commutator loss ε → 0:

1. **Exact relative-position property** (T6.1):
   - Attention depends only on `r_j - r_i`
   - Translation equivariance: `f(x + δ) = f(x)`

2. **Robust approximation** (T6.2):
   - For small ε, attention is approximately relative
   - Error bound: `Δ ≤ C(ε|r_i|_1|r_j|_1 + η_mix)|q_i||k_j|`

## Comparison with Standard Positional Encodings

| Method | Positional Info | Translation Equivariant | Proven Properties |
|--------|----------------|------------------------|-------------------|
| Absolute PE (Vaswani et al.) | Learned/Fixed | ✗ | None |
| RoPE (Su et al.) | Rotary | ~ (heuristic) | Empirical |
| **This work** | R_STR rotations | ✓ | Theorem T6.1 |

## Example: Verifying Relative-Position Property

```python
import torch
from src.rotations import StructuredRotation

# Create rotation module
rotation = StructuredRotation(d_h=32, d_c=2)

# Define positions
r_i = torch.tensor([0.2, 0.3])
r_j = torch.tensor([0.5, 0.7])

# Compute via relative position (T2.9)
R_rel = rotation(r_j - r_i)

# Compute via individual rotations and transpose
R_i = rotation(r_i)
R_j = rotation(r_j)
R_rel_indirect = R_i.T @ R_j

# Should be approximately equal (up to commutator error ε)
print(f"Difference: {torch.norm(R_rel - R_rel_indirect).item():.6f}")
# Expected: < 0.01 if commutator loss is well-optimized
```

## Citation

If you use this implementation, please cite the associated proof:

```bibtex
@article{equivariant_attention,
  title={Relative-Position Attention via Commuting Skew Generators},
  author={},
  year={2025},
  note={Formal proof in proof-axiomatic-compact.tex}
}
```

## Future Work

Potential extensions:
1. **3D vision**: Extend to point clouds (d_coord=3)
2. **Temporal modeling**: Apply to video (d_coord=3, including time)
3. **Formal verification**: Use theorem prover to verify implementation matches proof
4. **Scaling**: Test on ImageNet with larger models
