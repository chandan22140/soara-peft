#!/usr/bin/env python3
"""
experiment_e3_oracle.py
========================
E3 — Support-geometry oracle projection (no training; run this first).

Explains why a dense leading block beats a diagonal band *without* any optimiser,
learning-rate or module-coverage confound.

Procedure:
    For each of the 72 adapted matrices:
    1. Δ = W_FFT - W*
    2. SVD the pretrained weight: W* = U S₁ Vᵀ
    3. C = Uᵀ Δ V ∈ ℝ^{768×768}  — true update in pretrained singular basis
    4. For budgets k, compute retained energy ρ(Ω) for band, block, top-k, random-k

Usage:
    conda activate py310
    python experiment_e3_oracle.py --device cuda:1
"""

import os
import sys
import gc
import argparse
import time
import csv
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ─────────────────────────────────────────────────────────────────
#  Step 0: Full fine-tuning to get W_FFT checkpoint
# ─────────────────────────────────────────────────────────────────

def train_full_finetune(device, data_path='./data', output_dir='./outputs/e3_fft',
                        epochs=10, batch_size=16, lr=1e-4, seed=42):
    """
    Train a full fine-tuning checkpoint on CIFAR-100 with ViT-B/16.
    Returns path to saved checkpoint.
    """
    from transformers import ViTForImageClassification
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    ckpt_path = os.path.join(output_dir, 'fft_best.pth')
    if os.path.exists(ckpt_path):
        print(f"✓ FFT checkpoint already exists: {ckpt_path}")
        return ckpt_path

    print("=" * 60)
    print("TRAINING FULL FINE-TUNING CHECKPOINT")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Data
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

    train_ds = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                              transform=transform_train, download=True)
    test_ds = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                             transform=transform_test, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model — ALL parameters trainable
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=100,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine schedule
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader)  # 1 epoch warmup
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(progress * math.pi))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            logits = output.logits if hasattr(output, 'logits') else output
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss {loss.item():.4f}, Acc {correct/total:.4f}")

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                logits = output.logits if hasattr(output, 'logits') else output
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        test_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (acc={test_acc:.4f})")

    print(f"\n✓ FFT training complete. Best test acc: {best_acc:.4f}")
    print(f"  Checkpoint: {ckpt_path}")

    # Cleanup
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return ckpt_path


# ─────────────────────────────────────────────────────────────────
#  Step 1: Oracle projection analysis
# ─────────────────────────────────────────────────────────────────

TARGET_MODULES = ['query', 'key', 'value', 'dense']


def get_target_layers(model):
    """Get names and modules of the 72 target linear layers."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in TARGET_MODULES):
                # Exclude classifier
                if 'classifier' not in name and 'head' not in name:
                    layers.append((name, module))
    return layers


def compute_band_mask(d, b):
    """
    Create band mask of half-width b: Ω = {|i-j| ≤ b}.
    Returns boolean mask of shape (d, d).
    Budget k = (2b+1)*d - b*(b+1).
    """
    row_idx = torch.arange(d).unsqueeze(1)
    col_idx = torch.arange(d).unsqueeze(0)
    return (row_idx - col_idx).abs() <= b


def band_budget(d, b):
    """Number of non-zero entries in a band mask of half-width b."""
    return (2 * b + 1) * d - b * (b + 1)


def find_band_halfwidth(d, k):
    """Find the largest half-width b such that band_budget(d, b) ≤ k."""
    for b in range(d):
        if band_budget(d, b) > k:
            return max(0, b - 1)
    return d - 1


def compute_block_mask(d, r):
    """Create leading-block mask: Ω = [r]×[r]. Returns boolean mask (d, d)."""
    mask = torch.zeros(d, d, dtype=torch.bool)
    mask[:r, :r] = True
    return mask


def compute_topk_mask(C_abs, k):
    """Create top-k mask by absolute value. Returns boolean mask (d, d)."""
    d = C_abs.shape[0]
    flat = C_abs.flatten()
    _, indices = flat.topk(min(k, flat.numel()))
    mask = torch.zeros(flat.numel(), dtype=torch.bool)
    mask[indices] = True
    return mask.view(d, d)


def compute_random_mask(d, k, seed=42):
    """Create random-k mask. Returns boolean mask (d, d)."""
    rng = np.random.RandomState(seed)
    total = d * d
    indices = rng.choice(total, size=min(k, total), replace=False)
    mask = torch.zeros(total, dtype=torch.bool)
    mask[indices] = True
    return mask.view(d, d)


def retained_energy(C, mask):
    """
    Compute retained energy ρ(Ω) = Σ_{(i,j)∈Ω} C²_ij / ||C||²_F.
    """
    C_sq = C ** 2
    total = C_sq.sum().item()
    if total < 1e-30:
        return 0.0
    selected = C_sq[mask].sum().item()
    return selected / total


def run_oracle_projection(device, fft_ckpt_path, output_dir='./outputs/e3_results'):
    """
    Main E3 analysis: oracle projection for each of the 72 layers.
    """
    from transformers import ViTForImageClassification

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("E3 — ORACLE PROJECTION ANALYSIS")
    print("=" * 60)

    # Load pretrained model
    print("Loading pretrained ViT-B/16...")
    model_pretrained = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=100,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Load FFT model
    print("Loading FFT checkpoint...")
    model_fft = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=100,
        ignore_mismatched_sizes=True,
    ).to(device)
    ckpt = torch.load(fft_ckpt_path, map_location=device, weights_only=False)
    model_fft.load_state_dict(ckpt['model_state_dict'])
    fft_acc = ckpt.get('test_acc', 'N/A')
    print(f"  FFT test accuracy: {fft_acc}")

    # Get target layers
    layers_pretrained = get_target_layers(model_pretrained)
    layers_fft = get_target_layers(model_fft)
    assert len(layers_pretrained) == len(layers_fft), \
        f"Layer count mismatch: {len(layers_pretrained)} vs {len(layers_fft)}"
    print(f"  Found {len(layers_pretrained)} target layers")

    # Budgets matching Table 2
    budgets = [256, 528, 1792, 6892]

    # Results storage
    csv_path = os.path.join(output_dir, 'e3_results.csv')
    row_energy_csv = os.path.join(output_dir, 'e3_row_marginal.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer_name', 'layer_idx', 'budget_k', 'support', 'retained_energy',
                          'actual_nnz', 'd', 'out_feat', 'in_feat'])

    with open(row_energy_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer_name', 'layer_idx', 'row_i', 'row_energy_fraction'])

    all_results = []
    all_row_marginals = []

    for layer_idx, ((name_pre, mod_pre), (name_fft, mod_fft)) in enumerate(
            zip(layers_pretrained, layers_fft)):
        assert name_pre == name_fft, f"Name mismatch: {name_pre} vs {name_fft}"

        print(f"\n  [{layer_idx+1}/{len(layers_pretrained)}] {name_pre}")

        with torch.no_grad():
            W_pre = mod_pre.weight.data.float()
            W_fft = mod_fft.weight.data.float()
            out_feat, in_feat = W_pre.shape

            # Δ = W_FFT - W*
            Delta = W_fft - W_pre

            delta_fro = torch.norm(Delta, p='fro').item()
            print(f"    shape={out_feat}x{in_feat}, ||Δ||_F = {delta_fro:.6f}")

            if delta_fro < 1e-10:
                print(f"    WARNING: Δ ≈ 0 (layer unchanged by FFT), skipping")
                continue

            # SVD of pretrained weight W ∈ ℝ^{out×in}
            # torch.linalg.svd with full_matrices=False:
            #   U  : [out, k],  Vh : [k, in],  k = min(out, in)
            # So C = U^T @ Δ @ Vh^T  ∈  ℝ^{k×k}  (always square)
            # For square 768×768 layers: k=768; for 3072×768 layers: k=768.
            U, S1, V = torch.linalg.svd(W_pre, full_matrices=False)

            # C = U^T @ Δ @ Vh^T  (note: V here is Vh, so V.T is V_original)
            C = U.T @ Delta @ V.T  # [k, k]  where k = min(out_feat, in_feat)

            # d is the dimension of C (= min(out, in)), used for all mask sizes
            d = C.shape[0]

            # Verify: ||C||_F should equal ||Δ||_F (orthogonal transform preserves norms)
            # This holds exactly since U and V are orthonormal (full_matrices=False SVD)
            c_fro = torch.norm(C, p='fro').item()
            print(f"    ||C||_F = {c_fro:.6f} (should ≈ ||Δ||_F = {delta_fro:.6f})")

            # Row-marginal energy: Σ_j C²_ij for each row i
            C_sq = C ** 2
            row_energies = C_sq.sum(dim=1)  # [d]
            total_energy = row_energies.sum().item()
            row_energy_fracs = (row_energies / total_energy).cpu().numpy()

            for i, frac in enumerate(row_energy_fracs):
                all_row_marginals.append([name_pre, layer_idx, i, frac])

            # For each budget, compute retained energy under each support type
            for k in budgets:
                results_row = {}

                # Band
                b = find_band_halfwidth(d, k)
                band_mask = compute_band_mask(d, b).to(device)
                actual_nnz_band = band_mask.sum().item()
                rho_band = retained_energy(C, band_mask)

                # Block
                r_block = int(np.floor(np.sqrt(k)))
                block_mask = compute_block_mask(d, r_block).to(device)
                actual_nnz_block = block_mask.sum().item()
                rho_block = retained_energy(C, block_mask)

                # Top-k
                topk_mask = compute_topk_mask(C.abs(), k).to(device)
                actual_nnz_topk = topk_mask.sum().item()
                rho_topk = retained_energy(C, topk_mask)

                # Random-k
                random_mask = compute_random_mask(d, k, seed=42).to(device)
                actual_nnz_random = random_mask.sum().item()
                rho_random = retained_energy(C, random_mask)

                print(f"    k={k:>5d}: band(b={b})={rho_band:.4f}({actual_nnz_band}), "
                      f"block(r={r_block})={rho_block:.4f}({actual_nnz_block}), "
                      f"top-k={rho_topk:.4f}, random={rho_random:.4f}")

                all_results.extend([
                    [name_pre, layer_idx, k, 'band', rho_band, actual_nnz_band, d, out_feat, in_feat],
                    [name_pre, layer_idx, k, 'block', rho_block, actual_nnz_block, d, out_feat, in_feat],
                    [name_pre, layer_idx, k, 'topk', rho_topk, actual_nnz_topk, d, out_feat, in_feat],
                    [name_pre, layer_idx, k, 'random', rho_random, actual_nnz_random, d, out_feat, in_feat],
                ])

            del Delta, U, S1, V, C, C_sq, row_energies
            torch.cuda.empty_cache()

    # Write results
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for row in all_results:
            writer.writerow(row)
    print(f"\n✓ Results written to {csv_path}")

    with open(row_energy_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for row in all_row_marginals:
            writer.writerow(row)
    print(f"✓ Row-marginal energies written to {row_energy_csv}")

    # Cleanup models
    del model_pretrained, model_fft
    gc.collect()
    torch.cuda.empty_cache()

    return csv_path, row_energy_csv


# ─────────────────────────────────────────────────────────────────
#  Step 2: Plotting
# ─────────────────────────────────────────────────────────────────

def plot_results(csv_path, row_energy_csv, output_dir):
    """Generate figures from E3 results."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    df_row = pd.read_csv(row_energy_csv)

    # ──── Figure 1: Retained energy vs budget (aggregated over layers) ────
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    supports = ['block', 'band', 'topk', 'random']
    labels = ['Leading block (SOARA)', 'Band (SVFT$^b$)', 'Top-$k$ (oracle)', 'Random-$k$ (floor)']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9E9E9E']
    markers = ['o', 's', '^', 'x']

    for support, label, color, marker in zip(supports, labels, colors, markers):
        sub = df[df['support'] == support]
        agg = sub.groupby('budget_k')['retained_energy'].agg(['mean', 'std']).reset_index()
        ax.errorbar(agg['budget_k'], agg['mean'], yerr=agg['std'],
                     label=label, color=color, marker=marker, linewidth=2,
                     markersize=8, capsize=4)

    ax.set_xscale('log')
    ax.set_xlabel('Budget $k$ (parameters per matrix)', fontsize=13)
    ax.set_ylabel('Retained energy $\\rho(\\Omega)$', fontsize=13)
    ax.set_title('Oracle Projection: Support Geometry vs Retained Energy\n'
                 '(CIFAR-100 FFT → ViT-B/16, 72 layers, mean ± std)', fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'e3_retained_energy.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved: {fig_path}")
    plt.close()

    # ──── Figure 2: Row-marginal energy profile ────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Aggregate across layers
    row_agg = df_row.groupby('row_i')['row_energy_fraction'].agg(['mean', 'std']).reset_index()
    ax.plot(row_agg['row_i'], row_agg['mean'], color='#2196F3', linewidth=1.5)
    ax.fill_between(row_agg['row_i'],
                    row_agg['mean'] - row_agg['std'],
                    row_agg['mean'] + row_agg['std'],
                    alpha=0.2, color='#2196F3')
    ax.set_xlabel('Singular direction index $i$', fontsize=13)
    ax.set_ylabel('Row-marginal energy fraction $\\sum_j C_{ij}^2 / ||C||_F^2$', fontsize=13)
    ax.set_title('Row-Marginal Energy Profile\n'
                 '(Decay confirms adaptation concentrates in leading singular directions)',
                 fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, 'e3_row_marginal.pdf')
    plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path2.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved: {fig_path2}")
    plt.close()

    # ──── Table: Retained energy at each budget ────
    print("\n" + "=" * 70)
    print("E3 RESULTS TABLE — Retained Energy ρ(Ω) (mean ± std over 72 layers)")
    print("=" * 70)

    pivot = df.groupby(['budget_k', 'support'])['retained_energy'].agg(['mean', 'std'])
    for k in sorted(df['budget_k'].unique()):
        print(f"\n  Budget k = {k}:")
        for support in supports:
            if (k, support) in pivot.index:
                m, s = pivot.loc[(k, support)]
                print(f"    {support:>10s}: {m:.4f} ± {s:.4f}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E3: Oracle projection analysis")
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./outputs/e3_results')
    parser.add_argument('--fft-epochs', type=int, default=10)
    parser.add_argument('--fft-batch-size', type=int, default=16)
    parser.add_argument('--fft-lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-fft', action='store_true',
                        help='Skip FFT training (assume checkpoint exists)')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    start_time = time.time()

    # Step 0: Get FFT checkpoint
    fft_output_dir = os.path.join(args.output_dir, 'fft_checkpoint')
    fft_ckpt_path = os.path.join(fft_output_dir, 'fft_best.pth')

    if args.skip_fft and os.path.exists(fft_ckpt_path):
        print(f"✓ Skipping FFT training, using existing checkpoint: {fft_ckpt_path}")
    else:
        fft_ckpt_path = train_full_finetune(
            device=device,
            data_path=args.data_path,
            output_dir=fft_output_dir,
            epochs=args.fft_epochs,
            batch_size=args.fft_batch_size,
            lr=args.fft_lr,
            seed=args.seed,
        )

    # Step 1: Oracle projection analysis
    csv_path, row_energy_csv = run_oracle_projection(
        device=device,
        fft_ckpt_path=fft_ckpt_path,
        output_dir=args.output_dir,
    )

    # Step 2: Plotting
    plot_results(csv_path, row_energy_csv, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n✓ E3 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
