#!/usr/bin/env python3
"""
vision_table1_stats.py
======================
Reports mean ± standard deviation over 10 trials for each SOARA method on the
vision benchmarks (Table 1), and tests statistical significance against the
strongest relevant baseline.

Two modes of operation
----------------------
1. PARSE MODE  (default)
   Pass log files produced by train_vit_rotational.py via --log-dir.
   The script discovers all per-(method, dataset, seed) logs, collects test
   accuracies, and prints the table.

2. LAUNCH MODE  (--launch)
   Generates and optionally runs the shell commands needed to train each
   method × dataset × seed combination.  Useful when you still need to
   collect the 10-seed results.

Usage examples
--------------
# Parse existing logs and show stats:
python vision_table1_stats.py --log-dir ./outputs

# Print the 10-seed launch commands (dry run):
python vision_table1_stats.py --launch --dry-run

# Launch all training jobs sequentially:
python vision_table1_stats.py --launch
"""

import argparse
import itertools
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────────
#  Experiment configuration
# ─────────────────────────────────────────────────────────────────

# Datasets in Table 1 order
TABLE1_DATASETS = ["cifar100", "dtd", "sun397", "fer2013", "fgvc_aircraft"]
DATASET_DISPLAY = {
    "cifar100":      "CIFAR-100",
    "dtd":           "DTD",
    "sun397":        "SUN397",
    "fer2013":       "FER2013",
    "fgvc_aircraft": "FGVCAircraft",
}

# SOARA methods in Table 1 order
SOARA_METHODS = ["V2b_bfly_seq", "V1_r16", "V2a_givens_r16"]
METHOD_DISPLAY = {
    "V2b_bfly_seq":   "SOARA-V2b (BF seq)",
    "V1_r16":         "SOARA-V1 (r=16)",
    "V2a_givens_r16": "SOARA-V2a (Givens, r=16)",
}

# Strongest baselines from Table 1 (single numbers – these are the
# published results we compare against via one-sample tests when we only
# have SOARA samples).
BASELINE_SINGLE = {
    # Strongest published competitor per dataset:
    # KAdaptation is the top baseline overall.
    "cifar100":      91.2,
    "dtd":           71.4,
    "sun397":        75.1,
    "fer2013":       63.8,
    "fgvc_aircraft": 55.5,
    # name for display
    "_name":         "KAdaptation",
    "_avg":          71.40,
}

# 10 seeds to use
SEEDS = [2048]
# 256, 512, 1024, 2048
# ─────────────────────────────────────────────────────────────────
#  Per-method training configs — exact hyperparams from paper Table
#  (tab:vision_hyperparams).  All 15 Table-1 cells are covered;
#  food101 and resisc45 are included for the additional-results table.
# ─────────────────────────────────────────────────────────────────

METHOD_CONFIGS = {
    # ── SOARA-V2b (BF seq) ──────────────────────────────────────────
    # Paper columns: WD=--  (not used → 0.0),  λ_ortho=-- (not used)
    "V2b_bfly_seq": {
        "method": "V2",
        "extra_flags": "--use-butterfly --butterfly-sequential",
        "per_dataset": {
            # LR=2.0e-4,  r=768, Epochs=10, BS=16,  cycles=3
            "cifar100":      {"rank": 768, "epochs": 10,  "batch_size": 16,
                              "learning_rate": 2.0e-4, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=9.0e-4,  r=768, Epochs=40, BS=32,  cycles=3
            "dtd":           {"rank": 768, "epochs": 40,  "batch_size": 32,
                              "learning_rate": 9.0e-4, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=2.6e-4,  r=768, Epochs=20, BS=32,  cycles=3
            "sun397":        {"rank": 768, "epochs": 20,  "batch_size": 32,
                              "learning_rate": 2.6e-4, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=4.3e-4,  r=768, Epochs=10, BS=32,  cycles=3
            "fer2013":       {"rank": 768, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 4.3e-4, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=1.9e-3,  r=768, Epochs=30, BS=32,  cycles=3
            "fgvc_aircraft": {"rank": 768, "epochs": 30,  "batch_size": 32,
                              "learning_rate": 1.9e-3, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=1.9e-3,  r=768, Epochs=10, BS=16,  cycles=3
            "food101":       {"rank": 768, "epochs": 10,  "batch_size": 16,
                              "learning_rate": 1.9e-3, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=3.2e-3,  r=768, Epochs=10, BS=16,  cycles=3
            "resisc45":      {"rank": 768, "epochs": 10,  "batch_size": 16,
                              "learning_rate": 3.2e-3, "weight_decay": 0.0,
                              "total_cycles": 3},
        },
    },

    # ── SOARA-V1 (r=16) ─────────────────────────────────────────────
    # Paper columns: λ_ortho, WD, LR, r=16, Epochs, BS=32, cycles=--
    "V1_r16": {
        "method": "v1",
        "extra_flags": "",
        "per_dataset": {
            # λ=4.6e-2, WD=2.7e-4, LR=1.5e-4, Epochs=10
            "cifar100":      {"rank": 16, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 1.5e-4,
                              "weight_decay":  2.7e-4,
                              "orthogonality_weight": 4.6e-2},
            # λ=1.7e-3, WD=1.4e-4, LR=9.8e-4, Epochs=40
            "dtd":           {"rank": 16, "epochs": 40,  "batch_size": 32,
                              "learning_rate": 9.8e-4,
                              "weight_decay":  1.4e-4,
                              "orthogonality_weight": 1.7e-3},
            # λ=4.8e-2, WD=6.9e-4, LR=1.9e-4, Epochs=10
            "sun397":        {"rank": 16, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 1.9e-4,
                              "weight_decay":  6.9e-4,
                              "orthogonality_weight": 4.8e-2},
            # λ=4.6e-2, WD=1.1e-4, LR=6.0e-4, Epochs=10
            "fer2013":       {"rank": 16, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 6.0e-4,
                              "weight_decay":  1.1e-4,
                              "orthogonality_weight": 4.6e-2},
            # λ=1.9e-3, WD=1.9e-5, LR=9.1e-4, Epochs=30
            "fgvc_aircraft": {"rank": 16, "epochs": 30,  "batch_size": 32,
                              "learning_rate": 9.1e-4,
                              "weight_decay":  1.9e-5,
                              "orthogonality_weight": 1.9e-3},
            # λ=4.3e-2, WD=9.2e-4, LR=5.8e-4, Epochs=10
            "food101":       {"rank": 16, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 5.8e-4,
                              "weight_decay":  9.2e-4,
                              "orthogonality_weight": 4.3e-2},
            # λ=2.7e-2, WD=7.2e-4, LR=9.9e-4, Epochs=10
            "resisc45":      {"rank": 16, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 9.9e-4,
                              "weight_decay":  7.2e-4,
                              "orthogonality_weight": 2.7e-2},
        },
    },

    # ── SOARA-V2a (Givens, r=16) ────────────────────────────────────
    # Paper columns: WD=-- (not used → 0.0), λ_ortho=-- (not used)
    "V2a_givens_r16": {
        "method": "V2",
        "extra_flags": "",
        "per_dataset": {
            # LR=4.3e-4, Epochs=10, BS=32, cycles=3
            "cifar100":      {"rank": 16, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 4.3e-4, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=4.7e-4, Epochs=40, BS=32, cycles=3
            "dtd":           {"rank": 16, "epochs": 40,  "batch_size": 32,
                              "learning_rate": 4.7e-4, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=2.6e-4, Epochs=10, BS=32, cycles=3
            "sun397":        {"rank": 16, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 2.6e-4, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=6.0e-3, Epochs=10, BS=32, cycles=3
            "fer2013":       {"rank": 16, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 6.0e-3, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=3.3e-3, Epochs=30, BS=32, cycles=3
            "fgvc_aircraft": {"rank": 16, "epochs": 30,  "batch_size": 32,
                              "learning_rate": 3.3e-3, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=3.4e-4, Epochs=10, BS=32, cycles=3
            "food101":       {"rank": 16, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 3.4e-4, "weight_decay": 0.0,
                              "total_cycles": 3},
            # LR=4.1e-2, Epochs=10, BS=32, cycles=3
            "resisc45":      {"rank": 16, "epochs": 10,  "batch_size": 32,
                              "learning_rate": 4.1e-2, "weight_decay": 0.0,
                              "total_cycles": 3},
        },
    },
}


# ─────────────────────────────────────────────────────────────────
#  Log parsing
# ─────────────────────────────────────────────────────────────────

def parse_test_accuracy_from_log(log_path: Path) -> Optional[float]:
    """Extract final test accuracy from a train_vit_rotational.py log file."""
    try:
        content = log_path.read_text(errors="replace")
    except OSError:
        return None

    # Primary pattern (post-training analysis block)
    m = re.search(r"Test Accuracy:\s+([\d.]+)", content)
    if m:
        val = float(m.group(1))
        return val * 100 if val <= 1.0 else val   # normalise if needed

    # Fallback: last epoch val acc
    m_all = re.findall(r"Val Acc:\s+([\d.]+)", content)
    if m_all:
        val = float(m_all[-1])
        return val * 100 if val <= 1.0 else val

    return None


def discover_logs(log_dir: Path) -> Dict[Tuple[str, str, int], float]:
    """
    Walk log_dir and return {(method_key, dataset, seed): test_acc}.

    Expected filename convention (produced by generate_launch_commands):
        <method_key>_<dataset>_seed<seed>.log
    e.g.:
        V1_r16_cifar100_seed42.log
        V2b_bfly_seq_fgvc_aircraft_seed123.log
    """
    results: Dict[Tuple[str, str, int], float] = {}
    if not log_dir.exists():
        return results

    for f in sorted(log_dir.glob("*.log")):
        stem = f.stem  # filename without extension
        # Try to match pattern: <method>_<dataset>_seed<N>
        m = re.match(
            r"^(.+?)_(cifar100|dtd|sun397|fer2013|fgvc_aircraft|food101|resisc45)_seed(\d+)$",
            stem,
        )
        if not m:
            continue
        method_key = m.group(1)
        dataset = m.group(2)
        seed = int(m.group(3))
        if method_key not in SOARA_METHODS:
            continue
        acc = parse_test_accuracy_from_log(f)
        if acc is not None:
            results[(method_key, dataset, seed)] = acc

    return results


def load_results_from_json(json_path: Path) -> Dict[Tuple[str, str, int], float]:
    """Load previously saved results from a JSON file."""
    if not json_path.exists():
        return {}
    with json_path.open() as fh:
        data = json.load(fh)
    out = {}
    for entry in data:
        key = (entry["method"], entry["dataset"], entry["seed"])
        out[key] = entry["test_acc"]
    return out


def save_results_to_json(
    results: Dict[Tuple[str, str, int], float], json_path: Path
) -> None:
    data = [
        {"method": m, "dataset": d, "seed": s, "test_acc": acc}
        for (m, d, s), acc in sorted(results.items())
    ]
    json_path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(data)} result entries to {json_path}")


# ─────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────

def compute_stats(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std) using ddof=1 (sample std)."""
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def significance_vs_baseline(
    values: List[float], baseline_value: float, alpha: float = 0.05
) -> Tuple[float, bool, str]:
    """
    Test whether 'values' is significantly different from 'baseline_value'.

    Since we have samples for SOARA but only a point estimate for the baseline,
    we use a one-sample t-test (H0: mean(SOARA) == baseline).

    Returns
    -------
    p_value : float
    significant : bool  (True if p < alpha)
    direction : str     (">" or "<" or "=")
    """
    if len(values) < 2:
        return float("nan"), False, "?"
    arr = np.array(values, dtype=float)
    t_stat, p_value = scipy_stats.ttest_1samp(arr, popmean=baseline_value)
    significant = float(p_value) < alpha
    mean_soara = float(arr.mean())
    if mean_soara > baseline_value:
        direction = ">"
    elif mean_soara < baseline_value:
        direction = "<"
    else:
        direction = "="
    return float(p_value), significant, direction


def significance_paired(
    soara_values: List[float],
    baseline_values: List[float],
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Paired t-test + Wilcoxon signed-rank test when we have sample lists for
    both methods (e.g., two SOARA variants).

    Returns (p_ttest, p_wilcoxon, significant_at_alpha)
    """
    if len(soara_values) < 2 or len(baseline_values) < 2:
        return float("nan"), float("nan"), False
    n = min(len(soara_values), len(baseline_values))
    a = np.array(soara_values[:n], dtype=float)
    b = np.array(baseline_values[:n], dtype=float)
    _, p_t = scipy_stats.ttest_rel(a, b)
    try:
        _, p_w = scipy_stats.wilcoxon(a - b, alternative="two-sided")
    except ValueError:
        p_w = float("nan")
    return float(p_t), float(p_w), float(p_t) < alpha


# ─────────────────────────────────────────────────────────────────
#  Table printing
# ─────────────────────────────────────────────────────────────────

def print_console_table(
    results: Dict[Tuple[str, str, int], float],
    alpha: float = 0.05,
) -> None:
    """Print a formatted table with mean ± std and significance markers."""

    print("\n" + "=" * 100)
    print("TABLE 1  –  Vision Benchmarks: Top-1 Accuracy (%) for ViT-B/16")
    print(f"           Mean ± Std over {len(SEEDS)} trials; "
          f"significance vs. {BASELINE_SINGLE['_name']} "
          f"(α={alpha}, one-sample t-test)")
    print("=" * 100)

    col_w = 20
    ds_w  = 16

    header = f"{'Method':<{col_w}}" + "".join(
        f"{DATASET_DISPLAY[d]:>{ds_w}}" for d in TABLE1_DATASETS
    ) + f"{'Avg':>{ds_w}}"
    print(header)
    print("-" * len(header))

    # Baseline row
    bl_name = BASELINE_SINGLE["_name"]
    bl_row = f"{bl_name:<{col_w}}"
    bl_accs = []
    for ds in TABLE1_DATASETS:
        bl_val = BASELINE_SINGLE.get(ds, float("nan"))
        bl_accs.append(bl_val)
        bl_row += f"{bl_val:>{ds_w}.2f}"
    bl_avg = np.nanmean(bl_accs)
    bl_row += f"{bl_avg:>{ds_w}.2f}"
    print(bl_row)
    print("-" * len(header))

    method_avgs: Dict[str, List[float]] = {m: [] for m in SOARA_METHODS}

    for method_key in SOARA_METHODS:
        display = METHOD_DISPLAY[method_key]
        row = f"{display:<{col_w}}"
        method_avg_vals: List[float] = []

        for ds in TABLE1_DATASETS:
            values = [
                v for (mk, dk, _), v in results.items()
                if mk == method_key and dk == ds
            ]
            mean, std = compute_stats(values)
            bl_val = BASELINE_SINGLE.get(ds, float("nan"))
            p_val, sig, direction = significance_vs_baseline(values, bl_val, alpha)

            n_filled = len(values)
            if not values:
                cell = f"{'N/A':>{ds_w}}"
            else:
                sig_marker = "†" if sig else " "
                cell_str = f"{mean:.2f}±{std:.2f}{sig_marker}  (n={n_filled})"
                cell = f"{cell_str:>{ds_w}}"
                method_avg_vals.append(mean)
            row += cell

        # Average column
        if method_avg_vals:
            avg_mean = np.mean(method_avg_vals)
            row += f"{avg_mean:>{ds_w}.2f}"
            method_avgs[method_key] = method_avg_vals
        else:
            row += f"{'N/A':>{ds_w}}"

        print(row)

    print("-" * len(header))
    print(f"\n† = significantly different from {bl_name} at α={alpha} "
          "(one-sample t-test against published point estimate).")
    print()

    # ── Pairwise significance between SOARA methods ──────────────────────
    print("─" * 100)
    print("Pairwise significance tests between SOARA methods "
          "(paired t-test + Wilcoxon over ALL datasets & seeds)")
    print("─" * 100)

    method_pairs = list(itertools.combinations(SOARA_METHODS, 2))
    for m1, m2 in method_pairs:
        vals1, vals2 = [], []
        for ds in TABLE1_DATASETS:
            for seed in SEEDS:
                v1 = results.get((m1, ds, seed))
                v2 = results.get((m2, ds, seed))
                if v1 is not None and v2 is not None:
                    vals1.append(v1)
                    vals2.append(v2)
        p_t, p_w, sig = significance_paired(vals1, vals2, alpha)
        mu1 = np.mean(vals1) if vals1 else float("nan")
        mu2 = np.mean(vals2) if vals2 else float("nan")
        direction = ">" if mu1 > mu2 else "<"
        print(
            f"  {METHOD_DISPLAY[m1]}  {direction}  {METHOD_DISPLAY[m2]}"
            f"  |  t-test p={p_t:.4f}  |  Wilcoxon p={p_w:.4f}"
            f"  |  {'SIGNIFICANT' if sig else 'not significant'} at α={alpha}"
            f"  |  pairs n={len(vals1)}"
        )
    print()


def print_latex_table(
    results: Dict[Tuple[str, str, int], float],
    alpha: float = 0.05,
) -> None:
    """Print a LaTeX table snippet ready to paste into the paper."""

    print("\n" + "%" * 70)
    print("% LaTeX table (replace Table 1 in the paper)")
    print("%" * 70)
    print(r"\begin{table*}[t]")
    print(r"  \caption{Performance Comparison on Vision Benchmarks."
          r"  Top-1 accuracy (\%) for ViT-B/16 across five diverse datasets,"
          r"  reported as mean $\pm$ std over 10 trials."
          r"  $\dagger$: significantly different from KAdaptation ($\alpha=0.05$,"
          r"  one-sample $t$-test).}")
    print(r"  \label{tab:vision_main}")
    print(r"  \begin{tabular}{l" + "c" * (len(TABLE1_DATASETS) + 2) + "}")
    print(r"    \toprule")

    # header
    ds_headers = " & ".join(
        r"\textbf{" + DATASET_DISPLAY[d] + "}" for d in TABLE1_DATASETS
    )
    print(r"    \textbf{Method} & \textbf{\# Params ($\downarrow$)} & "
          + ds_headers
          + r" & \textbf{Avg.~($\uparrow$)} \\")
    print(r"    \midrule")

    # Baseline
    bl_vals = [BASELINE_SINGLE.get(d, float("nan")) for d in TABLE1_DATASETS]
    bl_avg  = np.nanmean(bl_vals)
    bl_cells = " & ".join(f"{v:.2f}" for v in bl_vals)
    print(
        f"    KAdaptation~\\cite{{...}} & 114,079 & "
        + bl_cells
        + f" & {bl_avg:.2f} \\\\"
    )
    print(r"    \midrule")

    PARAM_COUNTS = {
        "V2b_bfly_seq":   "129,024",
        "V1_r16":          " 38,016",
        "V2a_givens_r16":  "  2,304",
    }

    for method_key in SOARA_METHODS:
        display = METHOD_DISPLAY[method_key]
        cells = []
        avg_vals = []
        for ds in TABLE1_DATASETS:
            values = [
                v for (mk, dk, _), v in results.items()
                if mk == method_key and dk == ds
            ]
            mean, std = compute_stats(values)
            bl_val = BASELINE_SINGLE.get(ds, float("nan"))
            _, sig, direction = significance_vs_baseline(values, bl_val, alpha)
            if math.isnan(mean):
                cells.append("--")
            else:
                marker = r"$^\dagger$" if sig else ""
                cells.append(f"{mean:.2f}$\\pm${std:.2f}{marker}")
                avg_vals.append(mean)

        avg_str = f"{np.mean(avg_vals):.2f}" if avg_vals else "--"
        cell_str = " & ".join(cells)
        params = PARAM_COUNTS.get(method_key, "--")
        print(f"    {display} & {params} & {cell_str} & {avg_str} \\\\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table*}")
    print()


# ─────────────────────────────────────────────────────────────────
#  Launch-command generation
# ─────────────────────────────────────────────────────────────────

def build_command(
    method_key: str,
    dataset: str,
    seed: int,
    script: str = "train_vit_rotational.py",
    output_base: str = "./outputs",
) -> str:
    """Return a shell command string for one (method, dataset, seed) run."""
    cfg = METHOD_CONFIGS[method_key]
    ds_cfg = cfg["per_dataset"][dataset]

    out_dir = os.path.join(output_base, f"{method_key}_{dataset}_seed{seed}")
    log_file = os.path.join(
        output_base, f"{method_key}_{dataset}_seed{seed}.log"
    )

    parts = [
        f"python {script}",
        f"--method {cfg['method']}",
        f"--dataset {dataset}",
        f"--rank {ds_cfg['rank']}",
        f"--epochs {ds_cfg['epochs']}",
        f"--batch-size {ds_cfg['batch_size']}",
        f"--learning-rate {ds_cfg['learning_rate']}",
        f"--weight-decay {ds_cfg['weight_decay']}",
        f"--seed {seed}",
        f"--output-dir {out_dir}",
        "--no-wandb",
    ]

    # Optional per-method flags
    if cfg["extra_flags"]:
        parts.append(cfg["extra_flags"])
    if "total_cycles" in ds_cfg:
        parts.append(f"--total-cycles {ds_cfg['total_cycles']}")
    if "orthogonality_weight" in ds_cfg:
        parts.append(
            f"--orthogonality-weight {ds_cfg['orthogonality_weight']}"
        )

    cmd = " ".join(parts)
    cmd += f" 2>&1 | tee {log_file}"
    return cmd


def generate_launch_commands(
    output_base: str = "./outputs",
    script: str = "train_vit_rotational.py",
) -> List[str]:
    """Return list of shell commands for all method×dataset×seed combos."""
    cmds = []
    for method_key in SOARA_METHODS:
        for dataset in TABLE1_DATASETS:
            for seed in SEEDS:
                cmds.append(
                    build_command(method_key, dataset, seed, script, output_base)
                )
    return cmds


# ─────────────────────────────────────────────────────────────────
#  Summary stats helper & File Generator
# ─────────────────────────────────────────────────────────────────

def print_data_summary(results: Dict[Tuple[str, str, int], float]) -> None:
    print("\n── Data coverage ──────────────────────────────────────────")
    for method_key in SOARA_METHODS:
        for ds in TABLE1_DATASETS:
            collected = [
                seed for seed in SEEDS
                if (method_key, ds, seed) in results
            ]
            missing = set(SEEDS) - set(collected)
            status = (
                f"✓ {len(collected)}/10"
                if len(collected) == 10
                else f"⚠ {len(collected)}/10 (missing seeds: "
                     + ",".join(map(str, sorted(missing)))
                     + ")"
            )
            print(f"  {METHOD_DISPLAY[method_key]:<35} | {DATASET_DISPLAY[ds]:<15} | {status}")
    print()


def create_empty_logs(log_dir: Path) -> None:
    """Create empty log files pre-filled with 'Test Accuracy: ' for manual pasting."""
    log_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for method_key in SOARA_METHODS:
        for dataset in TABLE1_DATASETS:
            for seed in SEEDS:
                filename = f"{method_key}_{dataset}_seed{seed}.log"
                filepath = log_dir / filename
                if not filepath.exists():
                    filepath.write_text("Test Accuracy: ")
                    count += 1
    print(f"\nGenerated {count} empty log files in {log_dir}/")
    print("You can now open these files and paste your test acc numbers next to 'Test Accuracy: '.")
    print("Once done, run this script normally (without --create-empty-logs) to get the average and std.\n")


# ─────────────────────────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vision Table 1: mean±std over 10 trials + significance tests"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./outputs_vision_table1_stats"),
        help="Directory where per-seed log files are stored "
             "(filename pattern: <method>_<dataset>_seed<N>.log)",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional JSON file to load / save collected results. "
             "If provided and exists, results are loaded from it first.",
    )
    parser.add_argument(
        "--create-empty-logs",
        action="store_true",
        help="Generate empty .log files inside --log-dir so you can manually paste your outputs into them.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Generate (and optionally run) the 10-seed training commands.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --launch: only print commands, do not execute.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for hypothesis tests (default: 0.05).",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also print a LaTeX table snippet.",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="train_vit_rotational.py",
        help="Path to the training script (used by --launch).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Base output directory for training runs (used by --launch).",
    )
    args = parser.parse_args()

    # ── LAUNCH MODE ────────────────────────────────────────────────
    if args.launch:
        cmds = generate_launch_commands(args.output_dir, args.script)
        if args.dry_run:
            print(f"\n{'─'*70}")
            print(f"DRY RUN – {len(cmds)} commands to execute:")
            print(f"{'─'*70}")
            for i, cmd in enumerate(cmds, 1):
                print(f"\n[{i}/{len(cmds)}] {cmd}")
            print(f"\n{'─'*70}")
            print("Re-run without --dry-run to actually execute.")
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"\nLaunching {len(cmds)} training runs sequentially ...")
            for i, cmd in enumerate(cmds, 1):
                print(f"\n{'='*70}")
                print(f"[{i}/{len(cmds)}] {cmd}")
                print(f"{'='*70}")
                ret = subprocess.run(cmd, shell=True)
                if ret.returncode != 0:
                    print(
                        f"  ⚠  Command exited with code {ret.returncode}. "
                        "Continuing ..."
                    )
        return

    # ── GENERATE MODE ──────────────────────────────────────────────
    if args.create_empty_logs:
        create_empty_logs(args.log_dir)
        return

    # ── PARSE MODE ─────────────────────────────────────────────────
    results: Dict[Tuple[str, str, int], float] = {}

    # Load from JSON if given
    if args.results_json and args.results_json.exists():
        results.update(load_results_from_json(args.results_json))
        print(f"Loaded {len(results)} entries from {args.results_json}")

    # Discover additional logs
    if args.log_dir.exists():
        discovered = discover_logs(args.log_dir)
        new_count = sum(1 for k in discovered if k not in results)
        results.update(discovered)
        print(f"Discovered {len(discovered)} entries from {args.log_dir} "
              f"({new_count} new).")

    # Save merged results
    if args.results_json:
        save_results_to_json(results, args.results_json)

    if not results:
        print(
            "\n⚠  No results found. "
            "Run with --launch (or --launch --dry-run) to generate training "
            "commands, then re-run in parse mode with --log-dir pointing at "
            "where the logs were saved."
        )
        sys.exit(0)

    print_data_summary(results)
    print_console_table(results, alpha=args.alpha)

    if args.latex:
        print_latex_table(results, alpha=args.alpha)


if __name__ == "__main__":
    main()
