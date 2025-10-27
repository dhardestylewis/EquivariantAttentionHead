#!/usr/bin/env python3
"""
Experiment C: active subspace diagnostics.

Collects statistics about the learned active subspace projectors Π_act by
streaming batches through the model and aggregating projection norms, mixing
ratios (η_mix), and optional spatial maps for qualitative inspection.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.model import create_cifar_model, create_mnist_model


@dataclass
class DatasetConfig:
    create_model: callable
    dataset_cls: type
    transform: transforms.Compose
    fill_value: Tuple[float, ...]


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    'mnist': DatasetConfig(
        create_model=create_mnist_model,
        dataset_cls=datasets.MNIST,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        fill_value=(-0.1307 / 0.3081,),
    ),
    'cifar10': DatasetConfig(
        create_model=create_cifar_model,
        dataset_cls=datasets.CIFAR10,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]),
        fill_value=(
            -0.4914 / 0.2470,
            -0.4822 / 0.2435,
            -0.4465 / 0.2616,
        ),
    ),
}


@dataclass
class BlockAccumulator:
    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    maxima: Dict[str, float] = field(default_factory=dict)
    sample_maps: Dict[str, List[List[float]]] = field(default_factory=dict)
    grid_size: Optional[int] = None

    def update(self, key: str, tensor: torch.Tensor) -> None:
        value_sum = float(tensor.sum().item())
        value_count = int(tensor.numel())
        value_max = float(tensor.max().item())
        self.sums[key] = self.sums.get(key, 0.0) + value_sum
        self.counts[key] = self.counts.get(key, 0) + value_count
        current_max = self.maxima.get(key, float('-inf'))
        self.maxima[key] = max(current_max, value_max)

    def mean(self, key: str) -> Optional[float]:
        if key not in self.sums or key not in self.counts or self.counts[key] == 0:
            return None
        return self.sums[key] / self.counts[key]


def load_checkpoint_model(
    dataset: str,
    checkpoint_path: Path,
    device: torch.device,
    architecture_override: Optional[str],
) -> Tuple[torch.nn.Module, str, Optional[float]]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    ckpt_args = checkpoint.get('args')
    if ckpt_args is not None:
        if hasattr(ckpt_args, '__dict__'):
            ckpt_args = vars(ckpt_args)
        else:
            ckpt_args = dict(ckpt_args)
    else:
        ckpt_args = {}

    architecture = architecture_override or ckpt_args.get('architecture', 'relative')
    lambda_comm = ckpt_args.get('lambda_comm')

    model_kwargs_keys = [
        'd_model',
        'd_head',
        'n_heads',
        'n_layers',
        'd_ff',
        'patch_size',
        'dropout',
        'd_coord',
        'num_classes',
        'img_size',
        'in_channels',
    ]
    model_kwargs = {key: ckpt_args[key] for key in model_kwargs_keys if key in ckpt_args}

    config = DATASET_CONFIGS[dataset]
    model = config.create_model(architecture=architecture, **model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, architecture, lambda_comm


def build_dataloader(
    dataset: str,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    config = DATASET_CONFIGS[dataset]
    ds = config.dataset_cls(
        data_dir,
        train=False,
        download=True,
        transform=config.transform,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def maybe_extract_map(tensor: torch.Tensor) -> Tuple[Optional[List[List[float]]], Optional[int]]:
    """
    Convert per-token scalar tensor with CLS token at index 0 into a 2D grid (if square).
    """
    # tensor shape: (N, H) or (H, N); expect first dimension tokens.
    if tensor.dim() != 2:
        return None, None
    token_dim = tensor.size(0)
    if token_dim < 2:
        return None, None
    patch_tokens = token_dim - 1
    grid_size = int(round(math.sqrt(patch_tokens)))
    if grid_size * grid_size != patch_tokens:
        return None, None
    spatial = tensor[1:, :].mean(dim=-1).reshape(grid_size, grid_size)
    return spatial.tolist(), grid_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Π_act diagnostics from saved checkpoints.")
    parser.add_argument('--dataset', choices=DATASET_CONFIGS.keys(), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('./data'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-batches', type=int, default=4,
                        help="Number of batches to process for statistics (default: 4).")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--architecture', type=str, default=None,
                        help="Override architecture if checkpoint metadata is missing.")
    parser.add_argument('--store-sample', action='store_true',
                        help="Store spatial heatmaps for the first sample in the output JSON.")
    parser.add_argument('--output', type=Path, required=True,
                        help="Destination JSON file for diagnostic statistics.")
    args = parser.parse_args()

    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset '{args.dataset}'.")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint '{args.checkpoint}' does not exist.")

    device = torch.device(args.device)
    model, architecture, lambda_comm = load_checkpoint_model(
        dataset=args.dataset,
        checkpoint_path=args.checkpoint,
        device=device,
        architecture_override=args.architecture,
    )

    dataloader = build_dataloader(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    accumulators: List[BlockAccumulator] = []
    total_batches = 0
    commutator_losses: List[float] = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Collecting", total=args.max_batches)):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            data = data.to(device)
            _, metrics = model.forward_with_diagnostics(data)

            total_batches += 1
            commutator_losses.append(float(metrics['commutator_loss']))

            diagnostics = metrics.get('diagnostics')
            if diagnostics is None:
                continue

            if not accumulators:
                accumulators = [BlockAccumulator() if diag is not None else BlockAccumulator()
                                for diag in diagnostics]

            for block_idx, diag in enumerate(diagnostics):
                if diag is None:
                    continue
                accumulator = accumulators[block_idx]
                for key in ['q_proj_norm', 'q_eta_mix', 'k_proj_norm', 'k_eta_mix',
                            'q_residual_norm', 'k_residual_norm']:
                    tensor = diag[key]  # shape: (B, N, H)
                    accumulator.update(key, tensor)

                if args.store_sample and not accumulator.sample_maps:
                    sample_q = diag['q_proj_norm'][0]  # (N, H)
                    sample_k = diag['k_proj_norm'][0]
                    q_map, grid = maybe_extract_map(sample_q)
                    k_map, _ = maybe_extract_map(sample_k)
                    if q_map is not None:
                        accumulator.sample_maps['q_proj_norm'] = q_map
                        accumulator.grid_size = grid
                    if k_map is not None:
                        accumulator.sample_maps['k_proj_norm'] = k_map

    summary = {
        'dataset': args.dataset,
        'architecture': architecture,
        'lambda_comm': lambda_comm,
        'checkpoint': str(args.checkpoint),
        'num_batches': total_batches,
        'batch_size': args.batch_size,
        'commutator_loss_mean': (sum(commutator_losses) / len(commutator_losses)) if commutator_losses else None,
        'blocks': [],
    }

    for idx, accumulator in enumerate(accumulators):
        block_entry = {
            'index': idx,
            'means': {
                key: accumulator.mean(key)
                for key in accumulator.sums.keys()
            },
            'maxima': accumulator.maxima,
        }
        if accumulator.sample_maps:
            block_entry['sample_maps'] = accumulator.sample_maps
            block_entry['grid_size'] = accumulator.grid_size
        summary['blocks'].append(block_entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Saved diagnostics to {args.output}")


if __name__ == '__main__':
    main()
