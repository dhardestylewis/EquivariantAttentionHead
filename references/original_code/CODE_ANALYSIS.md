# Code Analysis: STRING Implementation Scripts

**Date:** 2025-12-13
**Location:** `references/original_code/`
**Files:**
- `test_implementation.py`
- `train_mnist.py`
- `train_cifar.py`

This document provides a detailed analysis of the original codebase to guide refactoring.

---

## 1. `test_implementation.py`

**Purpose:** Unit and integration testing for the algebraic properties of the STRING method (Structured Equivariant Attention).

### Key Test Cases
| Function | Description | Theorem Ref | Check/Constraint |
| :--- | :--- | :--- | :--- |
| `test_skew_symmetric` | Verifies generators $L$ are skew-symmetric. | T1.1 | $\|L^T + L\| < 10^{-5}$ |
| `test_relative_position` | Checks relative rotation property: $R(r_i)^T R(r_j) \approx R(r_j - r_i)$. | T2.9 | Reports error; notes that it may be non-zero before training (commutator loss). |
| `test_orthogonality` | Verifies rotation matrices are orthogonal. | T2.8 | $\|R^T R - I\| < 10^{-4}$ |
| `test_model_forward` | Runs a dummy forward pass on `MNISTModel`. | - | Checks output shape `(B, 10)` and existence of `commutator_loss`. |
| `test_commutator_loss`| Verifies gradients flow from commutator loss. | - | Checks `p.grad` is non-zero after backward. |
| `test_pairwise_rotations`| Checks `RelativePositionRotation` module. | - | Verifies $R(0) = I$ (identity check). |

**Dependencies:**
- `src.rotations` (`StructuredRotation`, `RelativePositionRotation`)
- `src.model` (`create_mnist_model`)

---

## 2. `train_mnist.py`

**Purpose:** Training script for MNIST dataset using STRING with commutator regularization.

### Architecture
- **Model:** `RelativePositionViT` (via `create_mnist_model`)
- **Defaults:** `d_model=64`, `d_head=16`, `n_heads=4`, `layers=4`, `d_ff=256`, `patch=4`.

### Training Logic
1.  **Objective:**
    $$ \mathcal{L} = \mathcal{L}_{CE} + \lambda_{comm} \cdot \mathcal{L}_{comm} $$
    - $\mathcal{L}_{comm}$ enforces commutativity of generators (T5.1).
2.  **Optimizer:** AdamW (`lr=1e-3`, `wd=0.01`).
3.  **Scheduler:** CosineAnnealingLR (`T_max=epochs`).
4.  **Early Stopping:** Patience-based (default 2 epochs without test acc improvement).
5.  **Metrics:** Tracks CE loss, Commutator loss, total loss, and accuracy.

### Key Functions
- `train_epoch(...)`: Handles forward/backward pass, loss combination, stats logging.
- `evaluate(...)`: Handles evaluation loop (no training, checks CE/Comm loss/Accuracy).

---

## 3. `train_cifar.py`

**Purpose:** Training script for CIFAR-10 dataset. Structurally similar to MNIST but with dataset-specific adjustments.

### Differences from MNIST
| Feature | `train_mnist.py` | `train_cifar.py` |
| :--- | :--- | :--- |
| **Model Params** | Small (`d_model=64`, `layers=4`) | Medium (`d_model=128`, `layers=6`) |
| **Data Augmentation**| Normalization only | RandomCrop + Flip + Norm |
| **Gradient Clipping** | None | `clip_grad_norm_(..., 1.0)` |
| **Early Stopping** | **Yes** (Patience-based) | **No** (Runs full epochs) |
| **Optimizer** | AdamW (`wd=0.01`) | AdamW (`wd=0.05`) |

### Boilerplate Duplication
These files share ~80% identical code:
- Argument parsing (almost identical flags).
- `train_epoch` logic (except clipping).
- `evaluate` function (identical).
- Metrics logging/JSON saving (identical).
- Checkpointing logic (identical).

---

## 4. Refactoring Plan

To reduce duplication and improve maintainability, the following changes are proposed:

### A. Shared `src/trainer.py`
Create a `Trainer` class (or helper functions) to encapsulate:
- `train_epoch`: With optional gradient clipping hook.
- `evaluate`: Shared logic.
- `fit`: Main loop handling epochs, patience, scheduling, and checkpointing.

### B. Specialized Scripts
Refactor `train_mnist.py` and `train_cifar.py` to be thin configuration wrappers that:
1.  Parse arguments.
2.  Setup DataLoaders (dataset specific).
3.  Instantiate Model (dataset specific).
4.  Call `Trainer.fit()`.

### C. Reference Storage
The original files are preserved in `references/original_code/` for regression testing and historical context.
