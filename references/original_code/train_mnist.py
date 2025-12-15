#!/usr/bin/env python3
"""
Train Relative-Position ViT on MNIST

Implements training with commutator regularization from Theorem T5.1:
    Total Loss = CE_loss + λ_comm * ε
where ε = max_{a,b} |[L_a, L_b]| enforces commutation condition T1.1

Updates in this version:
- Early stopping with patience.
  We stop if test accuracy hasn't improved in `--patience` epochs.
  This keeps us from burning compute once we've already demonstrated:
    (i) the structured relative-position attention trains,
    (ii) it generalizes.

- We still checkpoint the best model (highest test accuracy),
  and we still log full training history to a metrics JSON.
"""

import argparse
import json
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
    One training epoch.

    Loss = CE + λ_comm * commutator_penalty
    where commutator_penalty is the empirical ε from T5.1.
    """
    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_comm_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward
        logits, metrics = model(data)

        # Core CE objective
        ce_loss = nn.functional.cross_entropy(logits, target)

        # Commutator regularizer (controls generator non-commutation, T5.1)
        comm_loss = metrics['commutator_loss']

        # Total scalar objective
        loss = ce_loss + lambda_comm * comm_loss

        # Backprop
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_comm_loss += comm_loss.item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'comm': f'{comm_loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
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
    Evaluate on validation/test split.
    We do *not* include lambda_comm in the scalar here.
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

    # Training / optimization
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Max epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda-comm', type=float, default=0.01,
                        help='Commutator loss weight (T5.1)')
    parser.add_argument('--patience', type=int, default=2,
                        help='Early stop if no test acc improvement for this many epochs')

    # Architecture / logging / runtime
    parser.add_argument('--architecture', type=str, default='relative',
                        choices=['relative', 'absolute', 'rope'],
                        help='Model variant to instantiate')
    parser.add_argument('--metrics-file', type=str, default=None,
                        help='Optional JSON file to record metrics (defaults to save-dir/metrics_*.json)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory')

    # Model hyperparameters
    parser.add_argument('--d-model', type=int, default=64, help='Model dimension')
    parser.add_argument('--d-head', type=int, default=16, help='Head dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--d-ff', type=int, default=256, help='Feedforward dimension')
    parser.add_argument('--patch-size', type=int, default=4, help='Patch size')

    args = parser.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # I/O setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = (
        Path(args.metrics_file)
        if args.metrics_file
        else (save_dir / f'metrics_{args.architecture}.json')
    )

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        args.data_dir,
        train=False,
        transform=transform,
    )

    pin = (args.device == 'cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin,
    )

    # Model
    device = torch.device(args.device)
    model = create_mnist_model(
        architecture=args.architecture,
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        patch_size=args.patch_size,
    ).to(device)

    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training on {args.device}\n")

    # Optimizer + scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # Training loop with patience-based early stopping
    best_acc = 0.0
    best_epoch = 0
    best_metrics = None
    history = []

    epochs_since_improve = 0
    patience = max(args.patience, 1)  # just in case someone passes 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train one epoch
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.lambda_comm,
        )

        # Evaluate on test
        test_metrics = evaluate(
            model,
            test_loader,
            device,
        )

        # Bookkeeping
        epoch_record = {
            'epoch': epoch,
            'train': {k: float(v) for k, v in train_metrics.items()},
            'test': {k: float(v) for k, v in test_metrics.items()},
            'lr': float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_record)

        # Console report
        print(
            f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
            f"CE: {train_metrics['ce_loss']:.4f}, "
            f"Comm: {train_metrics['comm_loss']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.2f}%"
        )
        print(
            f"Test  - CE: {test_metrics['ce_loss']:.4f}, "
            f"Comm: {test_metrics['comm_loss']:.4f}, "
            f"Acc: {test_metrics['accuracy']:.2f}%"
        )

        # Check for improvement
        improved = test_metrics['accuracy'] > best_acc

        if improved:
            best_acc = float(test_metrics['accuracy'])
            best_epoch = epoch
            best_metrics = epoch_record
            epochs_since_improve = 0  # reset patience counter

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': vars(args),
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            print(f"✓ Saved best model (acc: {best_acc:.2f}%)")

        else:
            epochs_since_improve += 1
            print(f"(no new best for {epochs_since_improve} epoch(s))")

        # Scheduler step
        scheduler.step()

        # Early stopping trigger
        if epochs_since_improve >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch} "
                f"(no test acc improvement in {patience} epoch(s))"
            )
            break

    # Fallback in case we never "improved" (shouldn't really happen, but safe)
    if best_metrics is None and history:
        best_metrics = history[-1]
        best_epoch = best_metrics['epoch']
        best_acc = best_metrics['test']['accuracy']

    # Summary and metrics dump
    summary = {
        'dataset': 'mnist',
        'architecture': args.architecture,
        'lambda_comm': args.lambda_comm,
        'epochs_requested': args.epochs,
        'epochs_ran': len(history),
        'best_accuracy': float(best_acc),
        'best_epoch': int(best_epoch),
        'best_comm_loss': best_metrics['test']['comm_loss'] if best_metrics else None,
        'final_comm_loss': history[-1]['test']['comm_loss'] if history else None,
        'best_metrics': best_metrics,
        'history': history,
        'patience': patience,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8'
    )

    print(f"\nMetrics saved to {metrics_path}")
    print(
        f"\nTraining completed! "
        f"Best test accuracy: {best_acc:.2f}% "
        f"(epoch {best_epoch}, patience {patience})"
    )


if __name__ == '__main__':
    main()
