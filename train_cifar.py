#!/usr/bin/env python3
"""
Train Relative-Position ViT on CIFAR-10

More challenging dataset than MNIST to demonstrate theorem's effectiveness.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.model import create_cifar_model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_comm: float = 0.01,
) -> dict:
    """Train for one epoch with commutator regularization."""
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

        logits, metrics = model(data)

        ce_loss = nn.functional.cross_entropy(logits, target)
        comm_loss = metrics['commutator_loss']
        loss = ce_loss + lambda_comm * comm_loss

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

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
            'acc': f'{100.*correct/total:.2f}%'
        })

    return {
        'loss': total_loss / len(loader),
        'ce_loss': total_ce_loss / len(loader),
        'comm_loss': total_comm_loss / len(loader),
        'accuracy': 100. * correct / total,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Evaluate model."""
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
    parser = argparse.ArgumentParser(description='Train RelativePositionViT on CIFAR-10')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lambda-comm', type=float, default=0.01, help='Commutator loss weight (T5.1)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='checkpoints_cifar')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')

    # Model hyperparameters
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--d-head', type=int, default=32)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--patch-size', type=int, default=4)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data augmentation for CIFAR-10
    if args.augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(
        args.data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        args.data_dir,
        train=False,
        transform=test_transform
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

    device = torch.device(args.device)
    model = create_cifar_model(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        patch_size=args.patch_size,
    ).to(device)

    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training on {args.device}\n")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05
    )

    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.lambda_comm
        )

        test_metrics = evaluate(model, test_loader, device)

        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"CE: {train_metrics['ce_loss']:.4f}, "
              f"Comm: {train_metrics['comm_loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Test  - CE: {test_metrics['ce_loss']:.4f}, "
              f"Comm: {test_metrics['comm_loss']:.4f}, "
              f"Acc: {test_metrics['accuracy']:.2f}%")

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

        scheduler.step()

    print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
