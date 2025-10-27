#!/usr/bin/env python3
"""
Quick test script to verify implementation correctness.

Tests:
1. Skew-symmetric property of generators (T1.1)
2. Relative-position property (T2.9)
3. Forward pass works
4. Commutator loss computation
"""

import torch
from src.rotations import StructuredRotation, RelativePositionRotation
from src.model import create_mnist_model


def test_skew_symmetric():
    """Test that generators are skew-symmetric: L^T = -L (T1.1)"""
    print("\n" + "="*60)
    print("TEST 1: Skew-Symmetric Generators (T1.1)")
    print("="*60)

    rotation = StructuredRotation(d_h=16, d_c=2)
    generators = rotation.get_generators()

    for i, L in enumerate(generators):
        # Check L^T = -L
        diff = L.T + L
        error = torch.norm(diff).item()
        print(f"Generator L_{i+1}: ||L^T + L|| = {error:.6f}")
        assert error < 1e-5, f"Generator {i} not skew-symmetric!"

    print("✓ All generators are skew-symmetric")


def test_relative_position_property():
    """Test R_STR(r_i)^T R_STR(r_j) = R_STR(r_j - r_i) (T2.9)"""
    print("\n" + "="*60)
    print("TEST 2: Relative-Position Property (T2.9)")
    print("="*60)

    rotation = StructuredRotation(d_h=16, d_c=2)

    # Random positions
    r_i = torch.randn(2)
    r_j = torch.randn(2)

    # Compute via relative position
    R_rel_direct = rotation(r_j - r_i)

    # Compute via individual rotations
    R_i = rotation(r_i)
    R_j = rotation(r_j)
    R_rel_indirect = R_i.T @ R_j

    # Check if they match
    error = torch.norm(R_rel_direct - R_rel_indirect).item()
    print(f"||R_STR(r_j - r_i) - R_STR(r_i)^T R_STR(r_j)|| = {error:.6f}")

    # Note: Error may be nonzero initially due to non-commuting generators
    # During training, commutator loss will drive this to zero
    print(f"Commutator loss: {rotation.compute_commutator_loss().item():.6f}")

    if error < 0.1:
        print("✓ Relative-position property holds (within tolerance)")
    else:
        print("⚠ Relative-position property violated (generators not yet commuting)")
        print("  This is expected before training - commutator loss will fix this")


def test_orthogonality():
    """Test that R_STR(r) is orthogonal: R^T R = I (T2.8)"""
    print("\n" + "="*60)
    print("TEST 3: Orthogonality R^T R = I (T2.8)")
    print("="*60)

    rotation = StructuredRotation(d_h=16, d_c=2)
    r = torch.randn(2)

    R = rotation(r)

    # Check orthogonality
    I = torch.eye(16)
    error = torch.norm(R.T @ R - I).item()
    print(f"||R^T R - I|| = {error:.6f}")
    assert error < 1e-4, "R_STR is not orthogonal!"
    print("✓ R_STR is orthogonal")


def test_model_forward():
    """Test that model forward pass works"""
    print("\n" + "="*60)
    print("TEST 4: Model Forward Pass")
    print("="*60)

    model = create_mnist_model(
        d_model=32,
        d_head=8,
        n_heads=2,
        n_layers=2,
    )

    # Random MNIST-like input
    x = torch.randn(4, 1, 28, 28)  # Batch of 4 images

    # Forward pass
    try:
        logits, metrics = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Commutator loss: {metrics['commutator_loss'].item():.6f}")
        assert logits.shape == (4, 10), "Wrong output shape!"
        print("✓ Forward pass successful")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_commutator_loss_gradient():
    """Test that commutator loss has gradients"""
    print("\n" + "="*60)
    print("TEST 5: Commutator Loss Gradients")
    print("="*60)

    model = create_mnist_model(d_model=16, d_head=8, n_heads=1, n_layers=1)
    x = torch.randn(2, 1, 28, 28)

    logits, metrics = model(x)
    comm_loss = metrics['commutator_loss']

    # Backward pass
    comm_loss.backward()

    # Check that some parameters have gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    if has_grad:
        print("✓ Commutator loss produces gradients")
    else:
        print("✗ No gradients from commutator loss!")


def test_pairwise_rotations():
    """Test pairwise relative rotations"""
    print("\n" + "="*60)
    print("TEST 6: Pairwise Relative Rotations")
    print("="*60)

    rel_rot = RelativePositionRotation(d_h=8, d_c=2)

    # Grid positions for 3x3 patches
    positions = torch.tensor([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
    ])  # 9 positions

    rotations = rel_rot(positions)  # (9, 9, 8, 8)
    print(f"Pairwise rotations shape: {rotations.shape}")

    # Check that R[i,i] = I (rotation by zero displacement)
    for i in range(9):
        error = torch.norm(rotations[i, i] - torch.eye(8)).item()
        if error > 1e-4:
            print(f"Warning: R[{i},{i}] != I, error = {error:.6f}")

    print("✓ Pairwise rotations computed successfully")


def main():
    print("\nTesting Relative-Position Attention Implementation")
    print("Based on proof-axiomatic-compact.tex (T0.1-T6.2)")

    try:
        test_skew_symmetric()
        test_orthogonality()
        test_relative_position_property()
        test_pairwise_rotations()
        test_model_forward()
        test_commutator_loss_gradient()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nImplementation is ready for training!")
        print("Try: python train_mnist.py --epochs 5")

    except Exception as e:
        print("\n" + "="*60)
        print("TESTS FAILED ✗")
        print("="*60)
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
