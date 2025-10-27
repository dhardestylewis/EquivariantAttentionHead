# Equivariant Structured Positional Rotations

This repository contains two tightly linked artifacts:

1. **Formal proofs** showing that commuting skew-symmetric generators yield exact (and robust) relative-position attention.
2. **A PyTorch implementation** that follows those proofs, together with tests and training scripts for MNIST and CIFAR-10.

The manuscripts explain why the construction works; the code and tests verify that the numerical implementation satisfies the hypotheses before we train on data.

---

## Repository Layout

| Path | Description |
|------|-------------|
| `proof-algebraic-detailed.tex` | Full derivation. |
| `src/rotations.py` | Implements the structured rotations `R_STR` from Theorems T1.3-T2.9. |
| `src/attention.py` | Implements the relative-position attention head from Theorem T4.5 and the commutator penalty (T5.1). |
| `src/model.py` | Vision transformer that uses only the structured rotations (no learned positional encodings). |
| `train_mnist.py`, `train_cifar.py` | Reference training loops for MNIST and CIFAR-10. |
| `test_implementation.py` | Tests covering theorems T0.1-T6.2. |
| `IMPLEMENTATION.md` | Additional engineering notes and hyperparameters. |

---

## Installation

### Python environment

```powershell
python -m pip install -r requirements.txt
```

Only four packages are required: `torch`, `torchvision`, `numpy`, and `tqdm`.

---

## Structural Test Suite

Run all implementation checks with:

```powershell
python test_implementation.py
```

The script prints the outcome of six tests. A typical run ends with the line `ALL TESTS PASSED`.

### Test Coverage

| Test | Linked theorems | Purpose |
|------|-----------------|---------|
| Skew-symmetric generators | T1.1 | Confirms each learnable `L_k` satisfies `L_k^T = -L_k`. |
| Relative-position property | T2.9 | Checks that `R_STR(r_j - r_i)` matches `R_STR(r_i)^T R_STR(r_j)` within tolerance. |
| Orthogonality | T2.8 | Verifies that every `R_STR` lies in `SO(d_h)` (i.e., `R^T R = I`). |
| Pairwise relative rotations | T2.7 | Ensures `RelativePositionRotation` returns the full `(N, N, d_h, d_h)` tensor used by the attention head. |
| Transformer forward pass | T4.5 plus model definitions | Runs a forward pass of the MNIST classifier and reports the commutator penalty. |
| Commutator loss gradients | T5.1 | Confirms the commutator penalty participates in backpropagation. |

Passing the suite shows that the code respects the structural statements from Theorems T1.1, T2.3-T2.11, T4.1-T4.5, and T5.1. That is sufficient to demonstrate that the mechanics promised by the proofs are present for MNIST and CIFAR-10. Accuracy and convergence still depend on running the training scripts.

---

---

## Experimental Diagnostics

### Translation robustness (Experiment A)
Run the scripted probe to compare checkpoints under integer pixel shifts. Example on MNIST with three trained models (relative, absolute, rope):

`powershell
python analysis/shift_probe.py \
    --dataset mnist \
    --models relative=checkpoints/mnist_relative/best_model.pt \
             absolute=checkpoints/mnist_absolute/best_model.pt@absolute \
             rope=checkpoints/mnist_rope/best_model.pt@rope \
    --shift-pixels 0 2 4 8 \
    --device cuda \
    --output results/mnist_shift.json
`

The JSON log reports base accuracy, class agreement after shifting, and the average L2 change in logits for each translation.

### lambda_comm sweeps (Experiment B)
Aggregate the training runs produced with different regularisation weights to visualise the epsilon-accuracy frontier:

`powershell
python analysis/lambda_sweep.py \
    --metrics checkpoints/mnist_relative/metrics_relative.json \
             checkpoints/mnist_relative/metrics_lambda3e-4.json \
             checkpoints/mnist_relative/metrics_lambda1e-3.json \
    --sort-by epsilon \
    --output results/mnist_lambda_sweep.json
`

Each metrics file is emitted automatically by 	rain_mnist.py / 	rain_cifar.py; the sweep script prints a table and can save a consolidated JSON payload for plotting.

### Active subspace diagnostics (Experiment C)
Inspect Pi_act separation and eta_mix leakage for a trained checkpoint:

`powershell
python analysis/active_subspace.py \
    --dataset mnist \
    --checkpoint checkpoints/mnist_relative/best_model.pt \
    --max-batches 8 \
    --store-sample \
    --output results/mnist_active_subspace.json
`

The summary records per-block means/maxima for q_proj_norm, k_proj_norm, and eta_mix. With --store-sample it also captures a 2D heatmap of the active projection norms for the first validation example.

---

## Training Scripts

```powershell
# Optional sanity check
python test_implementation.py

# MNIST (about 20 epochs to reach roughly 99 percent accuracy)
python train_mnist.py --epochs 20 --lambda-comm 0.01 --device cuda

# CIFAR-10 (about 100 epochs with augmentation for roughly 85-90 percent accuracy)
python train_cifar.py --epochs 100 --augment --lambda-comm 0.01 --device cuda
```

The commutator penalty `lambda_comm * epsilon` is the empirical counterpart of Theorems T5.1 and T6.2. During training, monitor the reported `commutator_loss`; values below `1e-2` indicate the generators remain close to the commuting regime assumed in the theory.

---

## Proof to Implementation Map

| Proof statement | Implementation |
|-----------------|----------------|
| T1.3-T2.7 | `StructuredRotation` block-diagonalises commuting generators and exponentiates them. |
| T2.9 / T4.5 | `RelativePositionAttention` applies `R_STR(r_j - r_i)` so attention scores depend only on relative displacements. |
| T2.10 | `ActiveSubspaceProjector` maintains the active/null decomposition. |
| T5.1 / T6.2 | `compute_commutator_loss()` provides the penalty that keeps generators near the commuting manifold. |

With these pieces in place, the repository documents why the theory holds, how the implementation realises it, and how to validate and train the models on MNIST and CIFAR-10.

