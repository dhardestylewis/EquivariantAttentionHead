#!/usr/bin/env python3
"""
Train Relative-Position ViT on MNIST

Implements training with commutator regularization from Theorem T5.1:
    Total Loss = CE_loss + λ_comm * ε
where ε = max_{a,b} |[L_a, L_b]| enforces commutation condition T1.1
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.model import create_mnist_model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_comm: float = 0.01,
) -> dict:
    """
    Train for one epoch.

    Args:
        model: RelativePositionViT
        loader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        lambda_comm: Weight for commutator loss (Theorem T5.1)

    Returns:
        Dictionary with average losses
    """
    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_comm_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, metrics = model(data)

        # Cross-entropy loss
        ce_loss = nn.functional.cross_entropy(logits, target)

        # Commutator regularization (Theorem T5.1)
        comm_loss = metrics['commutator_loss']

        # Total loss
        loss = ce_loss + lambda_comm * comm_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_comm_loss += comm_loss.item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'comm': f'{comm_loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return {
        'loss': total_loss / len(loader),
        'ce_loss': total_ce_loss / len(loader),
        'comm_loss': total_comm_loss / len(loader),
        'accuracy': 100. * correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate model on validation/test set.

    Returns:
        Dictionary with losses and accuracy
    """
    model.eval()

    total_ce_loss = 0.0
    total_comm_loss = 0.0
    correct = 0
    total = 0

    for data, target in tqdm(loader, desc='Evaluating'):
        data, target = data.to(device), target.to(device)

        logits, metrics = model(data)

        ce_loss = nn.functional.cross_entropy(logits, target)
        comm_loss = metrics['commutator_loss']

        total_ce_loss += ce_loss.item()
        total_comm_loss += comm_loss.item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return {
        'ce_loss': total_ce_loss / len(loader),
        'comm_loss': total_comm_loss / len(loader),
        'accuracy': 100. * correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description='Train RelativePositionViT on MNIST')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda-comm', type=float, default=0.01, help='Commutator loss weight (T5.1)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')

    # Model hyperparameters
    parser.add_argument('--d-model', type=int, default=64, help='Model dimension')
    parser.add_argument('--d-head', type=int, default=16, help='Head dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--d-ff', type=int, default=256, help='Feedforward dimension')
    parser.add_argument('--patch-size', type=int, default=4, help='Patch size')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        args.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        args.data_dir,
        train=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )

    # Create model
    device = torch.device(args.device)
    model = create_mnist_model(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        patch_size=args.patch_size,
    ).to(device)

    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training on {args.device}\n")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.lambda_comm
        )

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)

        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"CE: {train_metrics['ce_loss']:.4f}, "
              f"Comm: {train_metrics['comm_loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Test  - CE: {test_metrics['ce_loss']:.4f}, "
              f"Comm: {test_metrics['comm_loss']:.4f}, "
              f"Acc: {test_metrics['accuracy']:.2f}%")

        # Save best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args,
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            print(f"✓ Saved best model (acc: {best_acc:.2f}%)")

        # Step scheduler
        scheduler.step()

    print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
