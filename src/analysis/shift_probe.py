#!/usr/bin/env python3
"""
Experiment A: translation-shift robustness probe.

Evaluates trained models on MNIST or CIFAR-10 under integer translations and
records how stable the logits/predictions remain relative to the unshifted
input. Results can be written to JSON and are printed in a compact table.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.model import create_cifar_model, create_mnist_model


@dataclass
class ModelSpec:
    label: str
    checkpoint: Path
    architecture: Optional[str] = None


DATASET_FACTORIES = {
    'mnist': {
        'create_model': create_mnist_model,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        'dataset': datasets.MNIST,
        'fill_value': (-0.1307 / 0.3081,),  # normalized background intensity
    },
    'cifar10': {
        'create_model': create_cifar_model,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]),
        'dataset': datasets.CIFAR10,
        'fill_value': (
            -0.4914 / 0.2470,
            -0.4822 / 0.2435,
            -0.4465 / 0.2616,
        ),
    },
}


def parse_model_specs(specs: Sequence[str]) -> List[ModelSpec]:
    parsed: List[ModelSpec] = []
    for spec in specs:
        if '=' not in spec:
            raise ValueError(
                f"Model specification '{spec}' must be in LABEL=PATH[@architecture] format."
            )
        label, remainder = spec.split('=', 1)
        if '@' in remainder:
            path_str, architecture = remainder.rsplit('@', 1)
        else:
            path_str, architecture = remainder, None
        parsed.append(ModelSpec(label=label, checkpoint=Path(path_str), architecture=architecture))
    return parsed


def build_offsets(shift_values: Sequence[int]) -> List[Tuple[int, int]]:
    unique = sorted({int(v) for v in shift_values})
    if 0 not in unique:
        unique.insert(0, 0)

    offsets = set()
    for dx in unique:
        dx_options = [dx] if dx == 0 else [dx, -dx]
        for dy in unique:
            dy_options = [dy] if dy == 0 else [dy, -dy]
            for ox in dx_options:
                for oy in dy_options:
                    offsets.add((ox, oy))
    # Remove the zero shift from final list but keep it at the front for reference
    ordered = sorted(offsets)
    return ordered


def translate(images: torch.Tensor, dx: int, dy: int, fill: torch.Tensor) -> torch.Tensor:
    """
    Translate a batch of image tensors by integer displacements with zero padding.
    Positive dx shifts to the right, positive dy shifts downward.
    """
    if dx == 0 and dy == 0:
        return images.clone()

    b, c, h, w = images.shape
    shifted = images.new_empty((b, c, h, w))
    shifted[:] = fill.view(1, c, 1, 1)

    if dx >= 0:
        src_x_start = 0
        src_x_end = w - dx
        dst_x_start = dx
        dst_x_end = w
    else:
        src_x_start = -dx
        src_x_end = w
        dst_x_start = 0
        dst_x_end = w + dx

    if dy >= 0:
        src_y_start = 0
        src_y_end = h - dy
        dst_y_start = dy
        dst_y_end = h
    else:
        src_y_start = -dy
        src_y_end = h
        dst_y_start = 0
        dst_y_end = h + dy

    if src_x_end <= src_x_start or src_y_end <= src_y_start:
        # Shift completely outside the frame – return the filled tensor.
        return shifted

    shifted[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = images[
        :, :, src_y_start:src_y_end, src_x_start:src_x_end
    ]
    return shifted


def load_model(
    dataset: str,
    spec: ModelSpec,
    device: torch.device,
    default_architecture: str,
) -> Tuple[torch.nn.Module, str]:
    checkpoint = torch.load(spec.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    ckpt_args = checkpoint.get('args')
    if ckpt_args is not None:
        if hasattr(ckpt_args, '__dict__'):
            ckpt_args = vars(ckpt_args)
        else:
            ckpt_args = dict(ckpt_args)
    else:
        ckpt_args = {}

    architecture = spec.architecture or ckpt_args.get('architecture') or default_architecture
    builder = DATASET_FACTORIES[dataset]['create_model']

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
    model = builder(architecture=architecture, **model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, architecture


def make_dataloader(
    dataset: str,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    factory = DATASET_FACTORIES[dataset]
    ds = factory['dataset'](
        data_dir,
        train=False,
        download=True,
        transform=factory['transform'],
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    offsets: Iterable[Tuple[int, int]],
    fill: torch.Tensor,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    offsets = list(offsets)
    non_zero_offsets = [offset for offset in offsets if offset != (0, 0)]
    totals = {
        'base_correct': 0.0,
        'total': 0,
        'shift_stats': {
            offset: {
                'agreement': 0.0,
                'correct': 0.0,
                'logit_l2_sum': 0.0,
            }
            for offset in non_zero_offsets
        },
    }

    with torch.no_grad():
        for batch in loader:
            data, target = batch
            batch_size = data.size(0)
            remaining = None if max_samples is None else max_samples - totals['total']
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and remaining < batch_size:
                data = data[:remaining]
                target = target[:remaining]
                batch_size = data.size(0)

            data = data.to(device)
            target = target.to(device)

            logits, _ = model(data)
            base_pred = logits.argmax(dim=-1)

            totals['base_correct'] += base_pred.eq(target).sum().item()
            totals['total'] += batch_size

            base_logits = logits
            for dx, dy in non_zero_offsets:
                shifted = translate(data, dx, dy, fill)
                shift_logits, _ = model(shifted)
                shift_pred = shift_logits.argmax(dim=-1)

                agreement = shift_pred.eq(base_pred).sum().item()
                correct = shift_pred.eq(target).sum().item()
                diff = (shift_logits - base_logits).norm(p=2, dim=-1).sum().item()

                stats = totals['shift_stats'][(dx, dy)]
                stats['agreement'] += agreement
                stats['correct'] += correct
                stats['logit_l2_sum'] += diff

            if remaining is not None and totals['total'] >= max_samples:
                break

    results = {
        'base_accuracy': totals['base_correct'] / totals['total'] if totals['total'] else 0.0,
        'num_samples': totals['total'],
        'shifts': [],
    }

    for dx, dy in non_zero_offsets:
        stats = totals['shift_stats'][(dx, dy)]
        results['shifts'].append({
            'dx': dx,
            'dy': dy,
            'agreement': stats['agreement'] / totals['total'] if totals['total'] else 0.0,
            'shift_accuracy': stats['correct'] / totals['total'] if totals['total'] else 0.0,
            'avg_logit_l2': stats['logit_l2_sum'] / totals['total'] if totals['total'] else 0.0,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Translation shift robustness probe.")
    parser.add_argument('--dataset', choices=DATASET_FACTORIES.keys(), required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('./data'))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--models', nargs='+', required=True,
                        help="Model specs in LABEL=checkpoint.pt[@architecture] format. "
                             "Use multiple --models entries for comparisons.")
    parser.add_argument('--shift-pixels', type=int, nargs='+', default=[0, 2, 4, 8],
                        help="Set of integer translations to test (evaluates ± for non-zero entries).")
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Limit evaluation to a subset of the test set.")
    parser.add_argument('--output', type=Path, default=None,
                        help="Optional path to write aggregated JSON metrics.")
    parser.add_argument('--default-architecture', type=str, default='relative',
                        help="Architecture fallback when checkpoint metadata is missing.")
    args = parser.parse_args()

    if args.dataset not in DATASET_FACTORIES:
        raise ValueError(f"Unsupported dataset '{args.dataset}'.")

    specs = parse_model_specs(args.models)
    offsets = build_offsets(args.shift_pixels)
    device = torch.device(args.device)

    dataloader = make_dataloader(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    fill_values = torch.tensor(
        DATASET_FACTORIES[args.dataset]['fill_value'],
        dtype=torch.float32,
        device=device,
    )

    summary = {
        'dataset': args.dataset,
        'shift_offsets': [offset for offset in offsets if offset != (0, 0)],
        'models': {},
    }

    for spec in specs:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint '{spec.checkpoint}' not found.")
        model, resolved_arch = load_model(args.dataset, spec, device, args.default_architecture)
        fill = fill_values
        arch_label = spec.architecture or resolved_arch
        print(f"\nEvaluating '{spec.label}' ({arch_label})...")

        results = evaluate_model(
            model=model,
            loader=tqdm(dataloader, desc=f"{spec.label}", leave=False),
            device=device,
            offsets=offsets,
            fill=fill,
            max_samples=args.max_samples,
        )

        summary['models'][spec.label] = {
            'architecture': arch_label,
            'base_accuracy': results['base_accuracy'],
            'num_samples': results['num_samples'],
            'metrics': results['shifts'],
        }

        print(f"  Base accuracy: {results['base_accuracy'] * 100:.2f}% "
              f"on {results['num_samples']} samples")
        print("  Shift results (dx, dy -> agreement | shift acc | avg L2):")
        for entry in results['shifts']:
            dx, dy = entry['dx'], entry['dy']
            agreement = entry['agreement'] * 100
            shift_acc = entry['shift_accuracy'] * 100
            l2 = entry['avg_logit_l2']
            print(f"    ({dx:>2}, {dy:>2}) -> {agreement:6.2f}% | {shift_acc:6.2f}% | {l2:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f"\nSaved metrics to {args.output}")


if __name__ == '__main__':
    main()
