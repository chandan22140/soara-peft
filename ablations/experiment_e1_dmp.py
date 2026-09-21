#!/usr/bin/env python3
"""
experiment_e1_dmp.py
=====================
E1 — Direct Matrix Parameterisation (DMP) vs SOARA ablation.

This is Requested Change 2: rank-matched comparison of:
    - DMP: free M ∈ ℝ^{r×r}, init = diag(S₁[:r])
    - V1-soft: R_U S R_V^T with orthogonality regularization (λ=4.6e-2)
    - V2b: butterfly, exact orthogonality

All arms satisfy W' = W* at step 0, start from identical initial adaptation
matrix diag(S₁[:r]).

Grid: 6 LRs × 3 arms × 3 seeds = 54 runs at r=16.
Backbone: ViT-B/16, ImageNet-21k pretrained, CIFAR-100, 72 matrices, 10 epochs.

Usage:
    conda activate py310
    python experiment_e1_dmp.py --device cuda:1
"""

import os
import sys
import gc
import copy
import csv
import math
import argparse
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from transformers import ViTForImageClassification
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from rotational_pissa_unified import (
    SOARAConfig, SOARALinearLayer, replace_linear_with_soara, SOARATrainer,
)
from experiment_instrumentation import (
    store_initial_singular_values,
    compute_block_diagnostics,
    append_diagnostics_to_csv,
    collect_soara_layers,
)


# ═══════════════════════════════════════════════════════════════════
#  DMP (Direct Matrix Parameterisation) Layer
# ═══════════════════════════════════════════════════════════════════

class DMPLinearLayer(nn.Module):
    """
    Direct Matrix Parameterisation: W' = U[:,:r] @ M @ V[:,:r]^T + W_residual.

    M ∈ ℝ^{r×r} is a FREE trainable parameter (no orthogonality constraint).
    Initialised at M₀ = diag(S₁[:r]) so W' = W* at step 0.

    This is the reviewer's suggested ablation: what happens when you drop
    the rotation constraint and just train a free r×r core?
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 16, freeze_singular_values: bool = False):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = min(rank, min(self.in_features, self.out_features))

        with torch.no_grad():
            W = base_layer.weight.data.float()
            U, S, V = torch.linalg.svd(W, full_matrices=False)

            # Principal components
            U_r = U[:, :self.r].contiguous()
            S_r = S[:self.r].contiguous()
            V_r = V[:self.r, :].contiguous()

            # Residual
            U_res = U[:, self.r:]
            S_res = S[self.r:]
            V_res = V[self.r:, :]
            if U_res.shape[1] > 0:
                W_res = U_res @ torch.diag(S_res) @ V_res
            else:
                W_res = torch.zeros_like(W)

            # Store residual as the base layer weight (frozen)
            base_layer.weight.data = W_res.to(base_layer.weight.dtype)
            base_layer.weight.requires_grad = False

            self.base_layer = base_layer
            self.register_buffer('U', U_r.to(base_layer.weight.dtype))
            self.register_buffer('V', V_r.to(base_layer.weight.dtype))

            # Store S_init for instrumentation
            self.register_buffer('S_init', S_r.clone())

            # M: free r×r trainable matrix, initialised at diag(S₁[:r])
            self.M = nn.Parameter(torch.diag(S_r).to(torch.float32))

            del U, S, V, U_res, S_res, V_res, W_res, W
            gc.collect()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)

        target_dtype = x.dtype
        U = self.U.to(target_dtype) if self.U.dtype != target_dtype else self.U
        V = self.V.to(target_dtype) if self.V.dtype != target_dtype else self.V
        M = self.M.to(target_dtype) if self.M.dtype != target_dtype else self.M

        # x @ V^T @ M^T @ U^T  (since W_pri = U @ M @ V)
        x_adapted = x @ V.T        # [batch, r]
        x_adapted = x_adapted @ M.T  # [batch, r]
        x_adapted = x_adapted @ U.T  # [batch, out]

        return result + x_adapted


def replace_linear_with_dmp(
    model: nn.Module,
    rank: int = 16,
    target_modules: List[str] = None,
    exclude_modules: List[str] = None,
) -> Dict[str, nn.Module]:
    """Replace linear layers with DMP adapters."""
    if target_modules is None:
        target_modules = ['query', 'key', 'value', 'dense']
    if exclude_modules is None:
        exclude_modules = ['classifier', 'head']

    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(excl in name for excl in exclude_modules):
                continue
            if any(target in name for target in target_modules):
                layers_to_replace.append((name, module))

    adapters = {}
    for name, module in layers_to_replace:
        adapted = DMPLinearLayer(module, rank=rank)

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]

        if parent_name:
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, adapted)
        else:
            setattr(model, child_name, adapted)

        adapters[name] = adapted

    # Freeze everything except DMP adapters and classifier
    for param in model.parameters():
        param.requires_grad = False

    for adapter in adapters.values():
        adapter.M.requires_grad = True

    # Unfreeze classifier
    for name, param in model.named_parameters():
        if 'classifier' in name or ('head' in name and 'encoder' not in name):
            param.requires_grad = True

    return adapters


# ═══════════════════════════════════════════════════════════════════
#  Data loading (shared across arms)
# ═══════════════════════════════════════════════════════════════════

def get_cifar100_loaders(batch_size=16, data_path='./data', num_workers=4):
    """Get CIFAR-100 train/val/test loaders."""
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_train = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                transform=transform_train, download=True)

    # Stratified val split (10%)
    from torch.utils.data import Subset
    from collections import defaultdict
    cls_idx = defaultdict(list)
    for idx, label in enumerate(full_train.targets):
        cls_idx[label].append(idx)

    rng = random.Random(42)
    train_idx, val_idx = [], []
    for cls, inds in cls_idx.items():
        rng.shuffle(inds)
        n_val = max(1, int(len(inds) * 0.1))
        val_idx.extend(inds[:n_val])
        train_idx.extend(inds[n_val:])

    train_ds = Subset(full_train, train_idx)
    val_ds_base = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                 transform=transform_test, download=True)
    val_ds = Subset(val_ds_base, val_idx)
    test_ds = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                             transform=transform_test, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"  Data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        logits = output.logits if hasattr(output, 'logits') else output
        loss = F.cross_entropy(logits, target)
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        total_loss += loss.item() * target.size(0)

    acc = correct / total
    avg_loss = total_loss / total
    return acc, avg_loss


# ═══════════════════════════════════════════════════════════════════
#  Training one run
# ═══════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_run(
    arm: str,
    lr: float,
    seed: int,
    rank: int,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    csv_path: str,
    epochs: int = 10,
    log_stride: int = 50,
    ortho_weight: float = 4.6e-2,
    total_cycles: int = 3,
) -> Dict:
    """
    Train one arm for one (lr, seed) setting.

    Arms:
        'DMP':     Free M ∈ ℝ^{r×r}
        'V1-soft': SOARA-V1 with λ=ortho_weight
        'V2b':     SOARA-V2b butterfly sequential
    """
    set_seed(seed)

    run_id = f"{arm}_lr{lr}_seed{seed}_r{rank}"
    print(f"\n{'─'*50}")
    print(f"  RUN: {run_id}")
    print(f"{'─'*50}")

    # Create fresh model
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=100,
        ignore_mismatched_sizes=True,
    ).to(device)

    rotational_trainer = None
    params_per_matrix = 0

    if arm == 'DMP':
        adapters = replace_linear_with_dmp(model, rank=rank)
        n_adapters = len(adapters)
        params_per_matrix = rank * rank
        print(f"  DMP: {n_adapters} adapters, {params_per_matrix} params/matrix")

    elif arm == 'V1-soft':
        soara_config = SOARAConfig(
            r=rank,
            lora_alpha=16.0,
            method='v1',
            orthogonality_reg_weight=ortho_weight,
            regularization_type='frobenius',
            s_dtype_fp32=True,
        )
        adapters = replace_linear_with_soara(
            model, soara_config,
            target_modules=['query', 'key', 'value', 'dense'],
            exclude_modules=['classifier', 'head'],
            freeze_base_model=True,
        )
        # Unfreeze classifier
        for name, param in model.named_parameters():
            if 'classifier' in name or ('head' in name and 'encoder' not in name):
                param.requires_grad = True
        rotational_trainer = SOARATrainer(model, soara_config)
        n_adapters = len(adapters)
        # V1 at r=16: 2 × r² (R_U, R_V) + r (S) = 528 + 16 = 544
        # But: R_U r², R_V r², S r => 2*16² + 16 = 528
        params_per_matrix = 2 * rank * rank + rank
        print(f"  V1-soft: {n_adapters} adapters, {params_per_matrix} params/matrix, λ={ortho_weight}")

    elif arm == 'V2b':
        # Compute steps_per_phase
        total_steps = len(train_loader) * epochs
        d_padded = 2 ** math.ceil(math.log2(rank)) if rank > 0 else 1
        num_phases_per_cycle = int(math.log2(d_padded))
        total_phases = num_phases_per_cycle * total_cycles
        steps_per_phase = max(1, total_steps // total_phases)

        soara_config = SOARAConfig(
            r=rank,
            lora_alpha=16.0,
            method='V2',
            use_butterfly=True,
            butterfly_sequential=True,
            butterfly_block_size=1,
            steps_per_phase=steps_per_phase,
            total_cycles=total_cycles,
            s_dtype_fp32=True,
        )
        adapters = replace_linear_with_soara(
            model, soara_config,
            target_modules=['query', 'key', 'value', 'dense'],
            exclude_modules=['classifier', 'head'],
            freeze_base_model=True,
        )
        for name, param in model.named_parameters():
            if 'classifier' in name or ('head' in name and 'encoder' not in name):
                param.requires_grad = True
        rotational_trainer = SOARATrainer(model, soara_config)
        n_adapters = len(adapters)
        # V2b at r=16: d_bf (butterfly angles per side) + r (S)
        # d_padded = 16, log2(16) = 4 components, each has d_padded/2 = 8 angles
        # Total per side: 4 × 8 = 32 angles, but only 1 component active at a time
        # Actual trainable per matrix: 2 × (d_padded/2) + r = 2*8 + 16 = 32
        bf_params = d_padded // 2  # angles per component per side
        params_per_matrix = 2 * bf_params + rank  # 2 sides × angles + S
        print(f"  V2b: {n_adapters} adapters, {params_per_matrix} params/matrix, "
              f"steps_per_phase={steps_per_phase}")

    else:
        raise ValueError(f"Unknown arm: {arm}")

    # Store S_init for diagnostics
    s_init_dict = store_initial_singular_values(model)

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable_params:,} trainable / {total_params:,} total")

    # Optimizer: AdamW with LoRA+ style LR for S
    lr_ratio_s = 10.0
    rotation_params = []
    s_params = []
    head_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or ('head' in name and 'encoder' not in name):
            head_params.append(param)
        elif '.S' in name and '.S_init' not in name and 'thetas' not in name:
            s_params.append(param)
        elif '.M' in name:
            # DMP's M — use base LR (it's the main param)
            rotation_params.append(param)
        elif 'thetas' in name or 'R_U' in name or 'R_V' in name:
            rotation_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if rotation_params:
        param_groups.append({'params': rotation_params, 'lr': lr})
    if s_params:
        param_groups.append({'params': s_params, 'lr': lr * lr_ratio_s})
    if head_params:
        param_groups.append({'params': head_params, 'lr': lr})
    if other_params:
        param_groups.append({'params': other_params, 'lr': lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)

    # Cosine schedule with 1-epoch warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            logits = output.logits if hasattr(output, 'logits') else output
            loss = F.cross_entropy(logits, target)

            # Orthogonality regularization for V1-soft
            if arm == 'V1-soft' and rotational_trainer is not None:
                ortho_loss = rotational_trainer.get_orthogonality_loss()
                loss = loss + ortho_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # V2b phase stepping
            if arm == 'V2b' and rotational_trainer is not None:
                if rotational_trainer.should_step_phase(global_step):
                    rotational_trainer.step_phase()
                    # Re-store S_init after phase step (S doesn't change, but structure might)
                    s_init_dict = store_initial_singular_values(model)

            pred = logits.argmax(dim=1)
            epoch_correct += pred.eq(target).sum().item()
            epoch_total += target.size(0)
            epoch_loss += loss.item()

            # Logging every log_stride steps
            if global_step % log_stride == 0:
                diag = compute_block_diagnostics(model, s_init_dict)
                # Quick val accuracy (use a subset for speed during training)
                val_acc_quick, _ = evaluate(model, val_loader, device)
                model.train()

                diag['val_acc'] = val_acc_quick
                diag['train_loss'] = loss.item()

                append_diagnostics_to_csv(
                    csv_path, run_id, arm, lr, seed, global_step, epoch, diag
                )

            global_step += 1

        # End of epoch evaluation
        train_acc = epoch_correct / epoch_total
        val_acc, val_loss = evaluate(model, val_loader, device)
        model.train()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"  Epoch {epoch+1}/{epochs}: Train Acc {train_acc:.4f}, "
              f"Val Acc {val_acc:.4f}, Best {best_val_acc:.4f}")

    # Final test evaluation
    test_acc, test_loss = evaluate(model, test_loader, device)

    result = {
        'arm': arm,
        'lr': lr,
        'seed': seed,
        'rank': rank,
        'params_per_matrix': params_per_matrix,
        'trainable_params': trainable_params,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
    }
    print(f"  ✓ {run_id}: best_val={best_val_acc:.4f}, test={test_acc:.4f}")

    # Cleanup
    del model, optimizer, scheduler, rotational_trainer
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════
#  E1 sweep
# ═══════════════════════════════════════════════════════════════════

def run_e1_sweep(device, output_dir='./outputs/e1_results', rank=16,
                 epochs=10, batch_size=16, data_path='./data'):
    """Run the full E1 sweep: 6 LRs × 3 arms × 3 seeds."""
    os.makedirs(output_dir, exist_ok=True)

    lrs = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3]
    arms = ['DMP', 'V1-soft', 'V2b']
    seeds = [42, 123, 2048]

    csv_path = os.path.join(output_dir, 'e1_diagnostics.csv')
    summary_path = os.path.join(output_dir, 'e1_summary.csv')

    # Get data loaders (shared across all runs)
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size, data_path=data_path
    )

    all_results = []
    total_runs = len(lrs) * len(arms) * len(seeds)
    run_count = 0

    print(f"\n{'='*60}")
    print(f"E1 SWEEP: {total_runs} runs ({len(lrs)} LRs × {len(arms)} arms × {len(seeds)} seeds)")
    print(f"  Rank: {rank}, Epochs: {epochs}, Batch size: {batch_size}")
    print(f"{'='*60}")

    for arm in arms:
        for lr in lrs:
            for seed in seeds:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] arm={arm}, lr={lr}, seed={seed}")

                result = train_one_run(
                    arm=arm, lr=lr, seed=seed, rank=rank,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    csv_path=csv_path,
                    epochs=epochs,
                )
                all_results.append(result)

    # Write summary CSV
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n✓ Summary written to {summary_path}")

    return all_results, csv_path, summary_path


# ═══════════════════════════════════════════════════════════════════
#  V2b at r=768 (paper setting, not rank-matched)
# ═══════════════════════════════════════════════════════════════════

def run_v2b_full_rank(device, output_dir='./outputs/e1_results',
                      epochs=10, batch_size=16, data_path='./data'):
    """
    Run V2b at r=768 (paper setting) for reference.
    Uses paper LR = 2e-4.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'e1_diagnostics.csv')

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size, data_path=data_path
    )

    result = train_one_run(
        arm='V2b', lr=2e-4, seed=42, rank=768,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        csv_path=csv_path,
        epochs=epochs,
    )

    print(f"\n✓ V2b r=768: best_val={result['best_val_acc']:.4f}, "
          f"test={result['test_acc']:.4f}, "
          f"params/matrix={result['params_per_matrix']}")
    return result


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_e1_results(summary_path, diagnostics_path, output_dir):
    """Generate E1 figures."""
    import pandas as pd

    df_summary = pd.read_csv(summary_path)
    df_diag = pd.read_csv(diagnostics_path)

    # ──── Figure 1: Accuracy vs LR (robustness curve) ────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    arms = [a for a in df_summary['arm'].unique() if a != 'V2b']
    colors = {'DMP': '#E53935', 'V1-soft': '#1E88E5', 'V2b': '#43A047'}
    markers = {'DMP': 'o', 'V1-soft': 's', 'V2b': '^'}

    for arm in arms:
        sub = df_summary[df_summary['arm'] == arm]
        agg = sub.groupby('lr')['best_val_acc'].agg(['mean', 'std']).reset_index()
        ax.errorbar(agg['lr'], agg['mean'], yerr=agg['std'],
                    label=arm, color=colors.get(arm, 'gray'),
                    marker=markers.get(arm, 'o'), linewidth=2, markersize=8, capsize=4)

    lrs = sorted(df_summary['lr'].unique())
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.set_xticks(lrs)
    labels = []
    for lr in lrs:
        if math.isclose(lr, 1e-4):
            labels.append(r'$10^{-4}$')
        elif math.isclose(lr, 1e-3):
            labels.append(r'$10^{-3}$')
        else:
            base = int(round(lr / (10 ** math.floor(math.log10(lr)))))
            exp = int(math.floor(math.log10(lr)))
            labels.append(rf'${base}\times 10^{{{exp}}}$')
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel('Learning Rate', fontsize=13)
    ax.set_ylabel('Best Validation Accuracy', fontsize=13)
    ax.set_title(f'E1: Accuracy vs LR (r=16, CIFAR-100)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='major', alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'e1_accuracy_vs_lr.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved: {fig_path}")
    plt.close()

    # ──── Figure 2: 2×2 panels (fro, spec) × (LR=2e-4, LR=5e-3) ────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    representative_lrs = [2e-4, 5e-3]
    metric_pairs = [
        ('fro', '$\\|S\\|_2$ (Frobenius norm)', 'fro_max', 'fro_mean'),
        ('spec', '$\\|S\\|_\\infty$ (spectral norm)', 'spec_max', 'spec_mean'),
    ]

    for col_idx, lr_val in enumerate(representative_lrs):
        for row_idx, (m_name, m_label, k_max, k_mean) in enumerate(metric_pairs):
            ax = axes[row_idx, col_idx]

            for arm_name in arms:
                arm_color = colors.get(arm_name, 'gray')

                # Plot max curve (solid)
                sub_max = df_diag[(df_diag['arm'] == arm_name) &
                                  (df_diag['lr'] == lr_val) &
                                  (df_diag['key'] == k_max)]
                if not sub_max.empty:
                    agg_max = sub_max.groupby('step')['value'].agg(['mean', 'std']).reset_index()
                    ax.plot(agg_max['step'], agg_max['mean'],
                            label=f'{arm_name} (max)', color=arm_color,
                            linestyle='-', linewidth=1.8)
                    ax.fill_between(agg_max['step'],
                                    agg_max['mean'] - agg_max['std'],
                                    agg_max['mean'] + agg_max['std'],
                                    alpha=0.12, color=arm_color)

                # Plot mean curve (dashed)
                sub_mean = df_diag[(df_diag['arm'] == arm_name) &
                                   (df_diag['lr'] == lr_val) &
                                   (df_diag['key'] == k_mean)]
                if not sub_mean.empty:
                    agg_mean = sub_mean.groupby('step')['value'].agg(['mean', 'std']).reset_index()
                    ax.plot(agg_mean['step'], agg_mean['mean'],
                            label=f'{arm_name} (mean)', color=arm_color,
                            linestyle='--', linewidth=1.8)
                    ax.fill_between(agg_mean['step'],
                                    agg_mean['mean'] - agg_mean['std'],
                                    agg_mean['mean'] + agg_mean['std'],
                                    alpha=0.08, color=arm_color)

            ax.set_xlabel('Step', fontsize=11)
            ax.set_ylabel(m_label, fontsize=11)
            ax.set_title(f'LR = {lr_val}', fontsize=12)
            ax.legend(fontsize=10, loc='upper left')
            ax.grid(True, alpha=0.3)

    fig.suptitle('E1: Principal-Block Update Norms (r=16)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'e1_norm_panels.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved: {fig_path}")
    plt.close()

    # ──── Table: Best accuracy per arm ────
    print("\n" + "=" * 70)
    print("E1 RESULTS TABLE — Best-over-LR Accuracy per Arm (r=16)")
    print("=" * 70)
    for arm_name in df_summary['arm'].unique():
        sub = df_summary[df_summary['arm'] == arm_name]
        # Best over LR: for each seed, pick best LR, then average
        best_per_seed = sub.groupby('seed')['best_val_acc'].max()
        mean_acc = best_per_seed.mean()
        std_acc = best_per_seed.std()
        ppm = sub['params_per_matrix'].iloc[0]
        total_p = sub['trainable_params'].iloc[0]
        print(f"  {arm_name:>10s}: {mean_acc:.4f} ± {std_acc:.4f}  "
              f"(params/matrix={ppm}, total={total_p:,})")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E1: DMP vs SOARA ablation")
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--output-dir', type=str, default='./outputs/e1_results')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--run-full-rank', action='store_true',
                        help='Also run V2b at r=768 (paper setting)')
    parser.add_argument('--only-plot', action='store_true',
                        help='Only generate plots from existing CSVs')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    start_time = time.time()

    if args.only_plot:
        summary_path = os.path.join(args.output_dir, 'e1_summary.csv')
        diag_path = os.path.join(args.output_dir, 'e1_diagnostics.csv')
        plot_e1_results(summary_path, diag_path, args.output_dir)
    else:
        # Run sweep
        all_results, diag_path, summary_path = run_e1_sweep(
            device=device,
            output_dir=args.output_dir,
            rank=args.rank,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_path=args.data_path,
        )

        # Optionally run V2b at full rank
        if args.run_full_rank:
            r768_result = run_v2b_full_rank(
                device=device,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                data_path=args.data_path,
            )

        # Plot
        plot_e1_results(summary_path, diag_path, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n✓ E1 complete in {elapsed:.1f}s ({elapsed/3600:.1f} hours)")


if __name__ == '__main__':
    main()
