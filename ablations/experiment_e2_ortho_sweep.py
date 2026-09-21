#!/usr/bin/env python3
"""
experiment_e2_ortho_sweep.py
==============================
E2 — Orthogonality-strength sweep (the causal test).

E1 compares parameterisations that differ in *two* ways (factorisation + constraint).
E2 separates them by varying *only* the constraint while holding function class,
parameter count, initialisation and optimiser fixed.

Arms (all r=16):
    DMP:     Free M ∈ ℝ^{r×r}               (reused from E1)
    V1-λ0:  Factorised R_U S R_V^T, λ=0     (no constraint — isolates factorisation bias)
    V1-λ:   Same, λ ∈ {1e-4, 1e-3, 1e-2, 4.6e-2, 1e-1}   (interpolates)
    V2b:    Exact orthogonality via butterfly (reused from E1)

The V1-λ0 arm is the key cell: it isolates pure over-parameterised-factorisation
implicit bias from the orthogonality constraint.

Usage:
    conda activate py310
    python experiment_e2_ortho_sweep.py --device cuda:1
"""

import os
import sys
import gc
import csv
import math
import argparse
import time
import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Reuse E1 infrastructure
from experiment_e1_dmp import (
    get_cifar100_loaders,
    evaluate,
    set_seed,
    train_one_run,
)


# ═══════════════════════════════════════════════════════════════════
#  Extended train_one_run that accepts arbitrary λ
# ═══════════════════════════════════════════════════════════════════

def train_one_run_e2(
    arm: str,
    lr: float,
    seed: int,
    rank: int,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    csv_path: str,
    epochs: int = 10,
    log_stride: int = 50,
    ortho_weight: float = 0.0,
    total_cycles: int = 3,
) -> Dict:
    """
    Thin wrapper around E1's train_one_run with explicit ortho_weight.

    For V1-λ0 (λ=0): just calls train_one_run with arm='V1-soft' and ortho_weight=0.
    For V1-λ: calls with specified ortho_weight.
    """
    # Map E2 arm names to E1's implementation
    if arm.startswith('V1-lambda'):
        # V1 with specific λ value
        return train_one_run(
            arm='V1-soft',
            lr=lr,
            seed=seed,
            rank=rank,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            csv_path=csv_path,
            epochs=epochs,
            log_stride=log_stride,
            ortho_weight=ortho_weight,
            total_cycles=total_cycles,
        )
    else:
        # DMP or V2b — delegate directly
        return train_one_run(
            arm=arm,
            lr=lr,
            seed=seed,
            rank=rank,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            csv_path=csv_path,
            epochs=epochs,
            log_stride=log_stride,
            ortho_weight=ortho_weight,
            total_cycles=total_cycles,
        )


# ═══════════════════════════════════════════════════════════════════
#  E2 sweep
# ═══════════════════════════════════════════════════════════════════

def run_e2_sweep(device, output_dir='./outputs/e2_results', rank=16,
                 epochs=10, batch_size=16, data_path='./data'):
    """
    Run the full E2 sweep.

    Arms:
        DMP (reused), V1-λ0, V1-λ(5 values), V2b (reused)

    Grid: LRs × arms × seeds
    """
    os.makedirs(output_dir, exist_ok=True)

    lrs = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3]
    seeds = [42, 123, 2048]
    lambda_values = [0.0, 1e-4, 1e-3, 1e-2, 4.6e-2, 1e-1]

    csv_path = os.path.join(output_dir, 'e2_diagnostics.csv')
    summary_path = os.path.join(output_dir, 'e2_summary.csv')

    # Get data loaders
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size, data_path=data_path
    )

    all_results = []

    # ──── DMP arm (baseline, no constraint) ────
    print(f"\n{'='*60}")
    print("E2: DMP arm (no constraint)")
    print(f"{'='*60}")
    for lr in lrs:
        for seed in seeds:
            result = train_one_run(
                arm='DMP', lr=lr, seed=seed, rank=rank,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                csv_path=csv_path,
                epochs=epochs,
            )
            result['lambda'] = float('nan')  # DMP has no λ
            result['arm_e2'] = 'DMP'
            all_results.append(result)

    # ──── V1-λ arms (sweep λ from 0 to 0.1) ────
    for lam in lambda_values:
        lam_str = f"{lam:.0e}" if lam > 0 else "0"
        arm_label = f'V1-lambda{lam_str}'

        print(f"\n{'='*60}")
        print(f"E2: V1 with λ={lam} ({arm_label})")
        print(f"{'='*60}")

        for lr in lrs:
            for seed in seeds:
                result = train_one_run_e2(
                    arm=arm_label, lr=lr, seed=seed, rank=rank,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    csv_path=csv_path,
                    epochs=epochs,
                    ortho_weight=lam,
                )
                result['lambda'] = lam
                result['arm_e2'] = f'V1-λ={lam}'
                all_results.append(result)

    # ──── V2b arm (exact constraint) ────
    print(f"\n{'='*60}")
    print("E2: V2b arm (exact constraint)")
    print(f"{'='*60}")
    for lr in lrs:
        for seed in seeds:
            result = train_one_run(
                arm='V2b', lr=lr, seed=seed, rank=rank,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                csv_path=csv_path,
                epochs=epochs,
            )
            result['lambda'] = float('inf')  # V2b = exact
            result['arm_e2'] = 'V2b'
            all_results.append(result)

    # Write summary
    with open(summary_path, 'w', newline='') as f:
        keys = list(all_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n✓ Summary written to {summary_path}")

    return all_results, csv_path, summary_path


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_e2_results(summary_path, diagnostics_path, output_dir):
    """Generate E2 figures."""
    import pandas as pd

    df = pd.read_csv(summary_path)
    df_diag = pd.read_csv(diagnostics_path)

    # ──── Figure: Accuracy, ortho_err, fro_max vs λ ────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Separate V1-λ runs (finite λ values)
    v1_runs = df[df['arm_e2'].str.startswith('V1-λ=')].copy()
    v1_runs['lambda_val'] = v1_runs['lambda'].astype(float)

    # For each λ, pick best LR per seed, then average
    best_acc_per_lambda = []
    for lam, group in v1_runs.groupby('lambda_val'):
        best_per_seed = group.groupby('seed')['best_val_acc'].max()
        best_acc_per_lambda.append({
            'lambda': lam,
            'acc_mean': best_per_seed.mean(),
            'acc_std': best_per_seed.std(),
        })
    v1_agg = pd.DataFrame(best_acc_per_lambda)

    # DMP and V2b baselines
    dmp_runs = df[df['arm_e2'] == 'DMP']
    v2b_runs = df[df['arm_e2'] == 'V2b']

    dmp_best = dmp_runs.groupby('seed')['best_val_acc'].max()
    v2b_best = v2b_runs.groupby('seed')['best_val_acc'].max()

    # Panel 1: Accuracy vs λ
    ax = axes[0]
    # V1-λ curve
    v1_nonzero = v1_agg[v1_agg['lambda'] > 0].sort_values('lambda')
    v1_zero = v1_agg[v1_agg['lambda'] == 0]

    if not v1_nonzero.empty:
        ax.errorbar(v1_nonzero['lambda'], v1_nonzero['acc_mean'],
                    yerr=v1_nonzero['acc_std'],
                    color='#1E88E5', marker='o', linewidth=2, markersize=8,
                    capsize=4, label='V1-λ')

    # V1-λ=0 as separate point (can't go on log axis)
    if not v1_zero.empty:
        ax.axhline(y=v1_zero['acc_mean'].iloc[0], color='#1E88E5',
                   linestyle='--', alpha=0.7, label=f'V1-λ=0 ({v1_zero["acc_mean"].iloc[0]:.4f})')

    # DMP baseline
    ax.axhline(y=dmp_best.mean(), color='#E53935', linestyle=':',
               linewidth=2, label=f'DMP ({dmp_best.mean():.4f})')

    # V2b baseline
    ax.axhline(y=v2b_best.mean(), color='#43A047', linestyle='-.',
               linewidth=2, label=f'V2b ({v2b_best.mean():.4f})')

    ax.set_xscale('log')
    ax.set_xlabel('$\\lambda$ (orthogonality weight)', fontsize=13)
    ax.set_ylabel('Best Validation Accuracy', fontsize=13)
    ax.set_title('Accuracy vs Constraint Strength', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel 2: ortho_err vs λ
    ax = axes[1]
    # Collect final ortho_err for each (arm, lr, seed)
    for arm_e2 in ['DMP', 'V2b']:
        sub = df_diag[(df_diag['arm'] == arm_e2.replace('-soft', '')) &
                       (df_diag['key'] == 'ortho_err')]
        if not sub.empty:
            # Get last step per run
            last = sub.groupby('run_id')['value'].last()
            label_name = arm_e2
            color = '#E53935' if arm_e2 == 'DMP' else '#43A047'
            ax.axhline(y=last.mean(), color=color, linestyle=':' if arm_e2 == 'DMP' else '-.',
                       linewidth=2, label=f'{label_name} ({last.mean():.4f})')

    # V1-λ ortho_err
    ortho_per_lambda = []
    for lam, group in v1_runs.groupby('lambda_val'):
        # Find matching diagnostics
        for _, row in group.iterrows():
            arm_name = row['arm']
            lr_val = row['lr']
            seed_val = row['seed']
            run_id_pattern = f"{arm_name}_lr{lr_val}_seed{seed_val}"
            sub = df_diag[(df_diag['run_id'].str.contains(str(lr_val))) &
                          (df_diag['key'] == 'ortho_err')]
            if not sub.empty:
                ortho_per_lambda.append({
                    'lambda': lam,
                    'ortho_err': sub['value'].iloc[-1],
                })

    if ortho_per_lambda:
        ortho_df = pd.DataFrame(ortho_per_lambda)
        ortho_agg = ortho_df.groupby('lambda')['ortho_err'].agg(['mean', 'std']).reset_index()

        nonzero = ortho_agg[ortho_agg['lambda'] > 0]
        if not nonzero.empty:
            ax.errorbar(nonzero['lambda'], nonzero['mean'], yerr=nonzero['std'],
                        color='#1E88E5', marker='o', linewidth=2, markersize=8,
                        capsize=4, label='V1-λ')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\\lambda$', fontsize=13)
    ax.set_ylabel('$\\|R^\\top R - I\\|_F$ (max over blocks)', fontsize=13)
    ax.set_title('Orthogonality Error vs $\\lambda$', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: fro_max vs λ
    ax = axes[2]
    fro_per_lambda = []
    for lam, group in v1_runs.groupby('lambda_val'):
        for _, row in group.iterrows():
            sub = df_diag[(df_diag['key'] == 'fro_max')]
            if not sub.empty:
                fro_per_lambda.append({
                    'lambda': lam,
                    'fro_max': sub['value'].iloc[-1],
                })

    if fro_per_lambda:
        fro_df = pd.DataFrame(fro_per_lambda)
        fro_agg = fro_df.groupby('lambda')['fro_max'].agg(['mean', 'std']).reset_index()

        nonzero = fro_agg[fro_agg['lambda'] > 0]
        if not nonzero.empty:
            ax.errorbar(nonzero['lambda'], nonzero['mean'], yerr=nonzero['std'],
                        color='#1E88E5', marker='o', linewidth=2, markersize=8,
                        capsize=4, label='V1-λ')

    ax.set_xscale('log')
    ax.set_xlabel('$\\lambda$', fontsize=13)
    ax.set_ylabel('$\\|\\Delta W_{\\mathrm{pri}}\\|_F$ (max)', fontsize=13)
    ax.set_title('Update Norm vs $\\lambda$', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('E2: Orthogonality Strength Sweep (r=16, CIFAR-100)', fontsize=14, y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'e2_ortho_sweep.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved: {fig_path}")
    plt.close()

    # ──── Table ────
    print("\n" + "=" * 70)
    print("E2 RESULTS TABLE — Best Accuracy by Constraint Strength")
    print("=" * 70)

    if not v1_agg.empty:
        for _, row in v1_agg.iterrows():
            lam = row['lambda']
            lam_str = f"{lam:.0e}" if lam > 0 else "0"
            print(f"  V1-λ={lam_str:>8s}: {row['acc_mean']:.4f} ± {row['acc_std']:.4f}")

    print(f"  {'DMP':>15s}: {dmp_best.mean():.4f} ± {dmp_best.std():.4f}")
    print(f"  {'V2b (exact)':>15s}: {v2b_best.mean():.4f} ± {v2b_best.std():.4f}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E2: Orthogonality strength sweep")
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--output-dir', type=str, default='./outputs/e2_results')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--only-plot', action='store_true',
                        help='Only generate plots from existing CSVs')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    start_time = time.time()

    if args.only_plot:
        summary_path = os.path.join(args.output_dir, 'e2_summary.csv')
        diag_path = os.path.join(args.output_dir, 'e2_diagnostics.csv')
        plot_e2_results(summary_path, diag_path, args.output_dir)
    else:
        # Run sweep
        all_results, diag_path, summary_path = run_e2_sweep(
            device=device,
            output_dir=args.output_dir,
            rank=args.rank,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_path=args.data_path,
        )

        # Plot
        plot_e2_results(summary_path, diag_path, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n✓ E2 complete in {elapsed:.1f}s ({elapsed/3600:.1f} hours)")


if __name__ == '__main__':
    main()
