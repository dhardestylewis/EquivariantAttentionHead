# STRING Robustness Verification

This repository contains the **theoretical proofs and empirical verification** for the robustness properties of **STRING** (Structured Equivariant Position Encodings).

## Core Results

We verify two key theoretical claims regarding STRING's approximate equivariance:
1.  **Metric Stability (Fragility)**: Relaxing the commutativity constraint ($[L_a, L_b] \neq 0$) leads to quadratic error growth in relative position encoding.
2.  **Zero Generalization Gap**: Exact STRING constraints guarantee a zero OOD generalization gap for translational shifts in the idealized limit.

## Key Files

*   **`STRING-Robustness.pdf`**: The technical report detailing the full algebraic derivations and empirical results.
*   **`demo_mnist_robustness.py`**: The self-contained Python script (PyTorch) that:
    *   Explicitly constructs Lie algebra generators.
    *   Verifies commutativity and mixing constraints (Metric A).
    *   Tests the Relative Operator Identity (Metric A').
    *   Demonstrates logit stability (Metric B) and generalization gap (Metric C) on MNIST.
*   **`mnist_robustness_verified.png`**: The resulting plot from the verification script.

## Directory Structure

*   `proofs_reference/`: Source text files for the mathematical proofs.
*   `references/`: Archived code and prior benchmarking scripts.
*   `project_docs/`: Project history, guidelines, and todo lists.
*   `tex.d/`: LaTeX style files.

## Usage

To run the verification sweep:

```bash
python demo_mnist_robustness.py
```

This will output the constraint verification metrics and generate `mnist_robustness_verified.png`.

## References

*   Schenck et al., "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING", arXiv:2502.02562 (2025).
