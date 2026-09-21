"""
GLUE Benchmark Training with DeBERTaV3-base + SOARA
================================================================

Trains DeBERTaV3-base on all GLUE tasks using SOARA (all 4 ways).
Reports results on development set with 5-seed averaging.

Tasks: MNLI, SST-2, CoLA, QQP, QNLI, RTE, MRPC, STS-B
Methods: v1, V2, v3, V4

Usage:
    python train_glue_deberta.py --task sst2 --method v1 --seed 42
    python train_glue_deberta.py --task all --method all --seeds 42,1234,2024,7890,5555
"""

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import sys
import argparse
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoConfig,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import wandb
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

# Import SOARA modules
from rotational_pissa_unified import (
    SOARAConfig,
    SOARALinearLayer,
    replace_linear_with_soara,
    SOARATrainer,
)

# ============================================================================
# GLUE TASK CONFIGURATIONS
# ============================================================================

GLUE_TASKS = {
    "mnli": {
        "num_labels": 3,
        "metric": "accuracy",  # Report m/mm accuracy
        "keys": ("premise", "hypothesis"),
        "is_regression": False,
        "validation_key": "validation_matched",  # Also has validation_mismatched
    },
    "sst2": {
        "num_labels": 2,
        "metric": "accuracy",
        "keys": ("sentence", None),
        "is_regression": False,
        "validation_key": "validation",
    },
    "cola": {
        "num_labels": 2,
        "metric": "matthews",
        "keys": ("sentence", None),
        "is_regression": False,
        "validation_key": "validation",
    },
    "qqp": {
        "num_labels": 2,
        "metric": "acc_f1",
        "keys": ("question1", "question2"),
        "is_regression": False,
        "validation_key": "validation",
    },
    "qnli": {
        "num_labels": 2,
        "metric": "accuracy",
        "keys": ("question", "sentence"),
        "is_regression": False,
        "validation_key": "validation",
    },
    "rte": {
        "num_labels": 2,
        "metric": "accuracy",
        "keys": ("sentence1", "sentence2"),
        "is_regression": False,
        "validation_key": "validation",
    },
    "mrpc": {
        "num_labels": 2,
        "metric": "acc_f1",
        "keys": ("sentence1", "sentence2"),
        "is_regression": False,
        "validation_key": "validation",
    },
    "stsb": {
        "num_labels": 1,
        "metric": "spearman",
        "keys": ("sentence1", "sentence2"),
        "is_regression": True,
        "validation_key": "validation",
    },
    "boolq": {
        "dataset_name": ("super_glue", "boolq"),
        "num_labels": 2,
        "metric": "accuracy",
        "keys": ("passage", "question"),
        "is_regression": False,
        "is_multiple_choice": False,
        "validation_key": "validation",
    },
    "piqa": {
        "dataset_name": ("piqa", None),
        "num_labels": 2,
        "metric": "accuracy",
        "is_regression": False,
        "is_multiple_choice": True,
        "validation_key": "validation",
    },
    "siqa": {
        "dataset_name": ("social_i_qa", None),
        "num_labels": 3,
        "metric": "accuracy",
        "is_regression": False,
        "is_multiple_choice": True,
        "validation_key": "validation",
    },
    "arc-c": {
        "dataset_name": ("ai2_arc", "ARC-Challenge"),
        "num_labels": 4,
        "metric": "accuracy",
        "is_regression": False,
        "is_multiple_choice": True,
        "validation_key": "validation",
    },
    "arc-e": {
        "dataset_name": ("ai2_arc", "ARC-Easy"),
        "num_labels": 4,
        "metric": "accuracy",
        "is_regression": False,
        "is_multiple_choice": True,
        "validation_key": "validation",
    },
    "obqa": {
        "dataset_name": ("openbookqa", "main"),
        "num_labels": 4,
        "metric": "accuracy",
        "is_regression": False,
        "is_multiple_choice": True,
        "validation_key": "validation",
    },
    "hellaswag": {
        "dataset_name": ("hellaswag", None),
        "num_labels": 4,
        "metric": "accuracy",
        "is_regression": False,
        "is_multiple_choice": True,
        "validation_key": "validation",
    },
    "winog": {
        "dataset_name": ("winogrande", "winogrande_xl"),
        "num_labels": 2,
        "metric": "accuracy",
        "is_regression": False,
        "is_multiple_choice": True,
        "validation_key": "validation",
    },
}


@dataclass
class GLUEConfig:
    """Configuration for GLUE benchmark training."""
    # Model
    model_name: str = "microsoft/deberta-v3-base"
    
    # Task
    task: str = "sst2"
    
    # SOARA
    method: str = "v1"
    soara_rank: int = 8
    soara_alpha: float = 16.0
    orthogonality_weight: float = 1e-4
    low_rank_r: int = 4
    total_cycles: int = 3  # For V2: how many full cycles through all layers
    use_butterfly: bool = False  # For V2: use butterfly factorization instead of sequential Givens
    butterfly_sequential: bool = False  # For V2 (butterfly): train components sequentially
    
    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 32
    max_seq_length: int = 256
    warmup_ratio: float = 0.06
    
    # Seeds for averaging
    seed: int = 42
    
    # Logging
    use_wandb: bool = True
    project_name: str = "glue-deberta-soara"
    output_dir: str = "./glue_results"
    logging_steps: int = 100
    logging_steps: int = 100
    max_steps: int = -1
    track_grad_norm: bool = False  # If False, skip computing/logging grad norm for speed
    
    # Model architecture
    no_pooler: bool = False  # If True, use CLS token directly instead of pooler
    
    # Device
    device: str = "auto"


# ============================================================================
# DATA LOADING
# ============================================================================

@dataclass
class DataCollatorForMultipleChoice:
    """Collator for multiple choice tasks."""
    tokenizer: Any
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def load_glue_dataset(task: str, tokenizer, max_seq_length: int = 128):
    """Load and tokenize a GLUE dataset."""
    task_config = GLUE_TASKS[task]
    is_mc = task_config.get("is_multiple_choice", False)
    
    # Load dataset
    if "dataset_name" in task_config:
        ds_name, ds_subset = task_config["dataset_name"]
        if ds_subset:
            dataset = load_dataset(ds_name, ds_subset, trust_remote_code=True)
        else:
            dataset = load_dataset(ds_name, trust_remote_code=True)
    else:
        dataset = load_dataset("glue", task, trust_remote_code=True)
    
    if is_mc:
        def preprocess_mc(example):
            if task == "piqa":
                context = example["goal"]
                choices = [example["sol1"], example["sol2"]]
                label = example["label"]
            elif task == "siqa":
                context = example["context"] + " " + example["question"]
                choices = [example["answerA"], example["answerB"], example["answerC"]]
                label = int(example["label"]) - 1
            elif task.startswith("arc"):
                context = example["question"]
                choices_text = example["choices"]["text"]
                choices_labels = example["choices"]["label"]
                # Try to find the correct index for answerKey
                try:
                    label = choices_labels.index(example["answerKey"])
                except ValueError:
                    label = 0
                choices = choices_text
                # Pad/truncate to ensure constant number of choices matching num_labels
                while len(choices) < task_config["num_labels"]:
                    choices.append("")
                choices = choices[:task_config["num_labels"]]
            elif task == "obqa":
                context = example["question_stem"]
                choices = example["choices"]["text"]
                try:
                    label = example["choices"]["label"].index(example["answerKey"])
                except ValueError:
                    label = 0
            elif task == "hellaswag":
                context = example["ctx"]
                choices = example["endings"]
                try:
                   # hellaswag label can be a string sometimes depending on format
                   label = int(example["label"]) if example["label"] else 0
                except:
                   label = 0
            elif task == "winog":
                context = example["sentence"]
                choices = [example["option1"], example["option2"]]
                label = int(example["answer"]) - 1
            else:
                raise ValueError(f"Unknown MC task mapping: {task}")
                
            return {"context": context, "choices": choices, "label": label}
            
        dataset = dataset.map(preprocess_mc)

        def tokenize_mc(examples):
            # Tokenize multiple choice: flatten choices and repeat contexts
            first_sentences = [[context] * len(choices) for context, choices in zip(examples["context"], examples["choices"])]
            second_sentences = examples["choices"]
            
            first_sentences = sum(first_sentences, [])
            second_sentences = sum(second_sentences, [])
            
            tokenized = tokenizer(first_sentences, second_sentences, truncation=True, max_length=max_seq_length, padding=False)
            
            # Un-flatten
            num_choices = task_config["num_labels"]
            return {
                k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)] 
                for k, v in tokenized.items()
            }
            
        tokenized = dataset.map(tokenize_mc, batched=True, remove_columns=dataset["train"].column_names)
        
        # Rename label -> labels for HF Trainer consistency
        if "label" in tokenized["train"].column_names:
            tokenized = tokenized.rename_column("label", "labels")
            
        return tokenized

    else:
        key1, key2 = task_config["keys"]
        
        def tokenize_function(examples):
            if key2 is None:
                return tokenizer(
                    examples[key1],
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                )
            else:
                return tokenizer(
                    examples[key1],
                    examples[key2],
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                )
        
        # Get columns to remove (all text columns except label)
        columns_to_remove = [col for col in dataset["train"].column_names 
                             if col not in ["label", "labels"]]
        
        # Tokenize - preserve label column
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
        
        # Rename label -> labels for consistency with HuggingFace Trainer
        if "label" in tokenized["train"].column_names:
            tokenized = tokenized.rename_column("label", "labels")
        
        return tokenized


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(task: str, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute task-specific metrics."""
    task_config = GLUE_TASKS[task]
    metric_type = task_config["metric"]
    is_regression = task_config["is_regression"]
    
    if is_regression:
        # STS-B: Spearman and Pearson correlation
        spearman = spearmanr(predictions, labels)[0]
        pearson = pearsonr(predictions, labels)[0]
        return {"spearman": spearman, "pearson": pearson, "corr": (spearman + pearson) / 2}
    
    # Classification: convert logits to predictions
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    
    if metric_type == "accuracy":
        return {"accuracy": accuracy_score(labels, predictions)}
    
    elif metric_type == "matthews":
        return {"matthews": matthews_corrcoef(labels, predictions)}
    
    elif metric_type == "acc_f1":
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="binary")
        return {"accuracy": acc, "f1": f1, "acc_f1": (acc + f1) / 2}
    
    return {}


# ============================================================================
# TRAINER CLASS
# ============================================================================

class GLUETrainer:
    """Trainer for GLUE tasks with SOARA."""
    
    def __init__(self, config: GLUEConfig):
        self.config = config
        self.task_config = GLUE_TASKS[config.task]
        
        # Set seed
        self._set_seed(config.seed)
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"\n{'='*60}")
        print(f"Task: {config.task.upper()}")
        print(f"Method: {config.method}")
        print(f"Seed: {config.seed}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Load tokenizer and model
        self._load_model()
        
        # Load data
        self._load_data()
        
        # Apply SOARA
        self._apply_soara()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _load_model(self):
        """Load DeBERTaV3 model and tokenizer."""
        print(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=self.task_config["num_labels"],
        )
        
        if self.task_config.get("is_multiple_choice", False):
            self.model = AutoModelForMultipleChoice.from_pretrained(
                self.config.model_name,
                config=model_config,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                config=model_config,
            )
        
        # Remove pooler if requested - use CLS token directly
        if self.config.no_pooler:
            print("  Removing pooler - using [CLS] token directly for classification")
            # Replace pooler with identity (just passes through the hidden state)
            class IdentityPooler(torch.nn.Module):
                def __init__(self, config=None):
                    super().__init__()
                    self.dense = torch.nn.Identity()
                    self.dropout = torch.nn.Dropout(0)
                def forward(self, hidden_states, **kwargs):
                    # Just return the [CLS] token (first token)
                    return hidden_states[:, 0]
            
            if hasattr(self.model, "pooler"):
                self.model.pooler = IdentityPooler(model_config)
            
            # Reinitialize classifier to match hidden_size directly
            if self.task_config.get("is_multiple_choice", False):
                self.model.classifier = torch.nn.Linear(
                    model_config.hidden_size, 
                    1
                )
                print(f"  New classifier: Linear({model_config.hidden_size}, 1)")
            else:
                self.model.classifier = torch.nn.Linear(
                    model_config.hidden_size, 
                    self.task_config["num_labels"]
                )
                print(f"  New classifier: Linear({model_config.hidden_size}, {self.task_config['num_labels']})")
        
        self.model.to(self.device)
    
    def _load_data(self):
        """Load GLUE dataset."""
        print(f"Loading dataset: {self.config.task}")
        
        dataset = load_glue_dataset(
            self.config.task,
            self.tokenizer,
            self.config.max_seq_length,
        )
        
        # Data collator
        if self.task_config.get("is_multiple_choice", False):
            collator = DataCollatorForMultipleChoice(self.tokenizer, padding=True)
        else:
            collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        
        # Create dataloaders (with parallel loading for speed)
        self.train_loader = DataLoader(
            dataset["train"],
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )
        
        val_key = self.task_config["validation_key"]
        self.val_loader = DataLoader(
            dataset[val_key],
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )
        
        # For MNLI, also load mismatched validation
        if self.config.task == "mnli":
            self.val_loader_mm = DataLoader(
                dataset["validation_mismatched"],
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=4,
                pin_memory=True,
            )
        
        print(f"  Train samples: {len(dataset['train'])}")
        print(f"  Val samples: {len(dataset[val_key])}")
    
    def _apply_soara(self):
        """Apply SOARA to the model."""
        print(f"\nApplying SOARA (method={self.config.method}, rank={self.config.soara_rank})")
        
        # Calculate adaptive steps_per_phase
        import math
        total_steps = len(self.train_loader) * self.config.epochs
        
        # Determine number of phases per cycle
        if self.config.use_butterfly:
            # Butterfly sequential: log2(d_padded)
            # We need to approximate d_padded if we don't know it yet
            # But roughly it's just log2(r) or next power of 2
            r = self.config.soara_rank
            if hasattr(self.model.config, "hidden_size") and r == self.model.config.hidden_size:
                 # If using full rank, it likely matches d_model which is often power of 2 or close
                 pass
            
            d_padded = 2 ** math.ceil(math.log2(r)) if r > 0 else 1 
            # If r is not power of 2, ButterflyRotationLayer pads it.
            # Number of components m = log2(d_padded)
            phases_per_cycle = int(math.log2(d_padded))
        else:
            # Standard sequential Givens: r-1 phases
            phases_per_cycle = max(1, self.config.soara_rank - 1)
        
        # Calculate steps per phase to fit total_cycles exactly into total_steps
        # total_steps = steps_per_phase * phases_per_cycle * total_cycles
        total_phases = phases_per_cycle * self.config.total_cycles
        steps_per_phase = max(1, total_steps // total_phases)
        
        print(f"  Adaptive Scheduling:")
        print(f"    Total steps: {total_steps}")
        print(f"    Phases per cycle: {phases_per_cycle}")
        print(f"    Total cycles: {self.config.total_cycles}")
        print(f"    Total phases: {total_phases}")
        print(f"    => Steps per phase: {steps_per_phase} (was default 100)")

        # Configure SOARA
        soara_config = SOARAConfig(
            r=self.config.soara_rank,
            lora_alpha=self.config.soara_alpha,
            method=self.config.method,
            orthogonality_reg_weight=self.config.orthogonality_weight,
            low_rank_r=self.config.low_rank_r,
            total_cycles=self.config.total_cycles,
            steps_per_phase=steps_per_phase,
            use_butterfly=self.config.use_butterfly,
            butterfly_sequential=self.config.butterfly_sequential,
            init_identity=True,
            freeze_singular_values=False,
            s_dtype_fp32=True,
        )
        
        # Target modules for DeBERTa:
        # - query_proj, key_proj, value_proj: attention Q, K, V projections
        # - attention.output.dense: attention O projection  
        # - intermediate.dense: FFN up-projection (768 -> 3072)
        # - output.dense: FFN down-projection (3072 -> 768)
        # We use "dense" which matches all dense layers in encoder
        # BUT we exclude pooler.dense and classifier (these need full training)
        # target_modules = ["query_proj", "key_proj", "value_proj"]
        # All encoder dense layers (not pooler)

        target_modules = [
            "query_proj", "key_proj", "value_proj",  # Q, K, V
            "attention.output.dense",                 # O projection
            "intermediate.dense",                     # FFN up
            "output.dense",                           # FFN down (careful: not pooler)
        ]
        exclude_modules = ["pooler", "classifier"]  # Don't apply SOARA to these
        
        # Replace linear layers (excludes pooler and classifier)
        self.adapters = replace_linear_with_soara(
            self.model,
            soara_config,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            freeze_base_model=True,
        )
        
        # Create trainer for orthogonality loss
        self.rotational_trainer = SOARATrainer(self.model, soara_config)
        self.soara_config = soara_config
        
        # CRITICAL: Unfreeze classifier (and pooler if present) - they're randomly initialized!
        # They need FULL training, not SOARA adaptation
        print("  Making classifier (and pooler if present) fully trainable (no SOARA):")
        for name, param in self.model.named_parameters():
            if "classifier" in name or ("pooler" in name and not self.config.no_pooler):
                param.requires_grad = True
                print(f"    Unfreezing: {name} ({param.numel():,} params)")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Calculate total steps
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    
    def train(self) -> Dict[str, float]:
        """Train the model and return final metrics."""
        # Initialize wandb (skip if already running from sweep agent)
        if self.config.use_wandb and wandb.run is None:
            run_name = f"{self.config.task}_{self.config.method}_seed{self.config.seed}"
            wandb.init(
                project=self.config.project_name,
                name=run_name,
                config={
                    "task": self.config.task,
                    "method": self.config.method,
                    "seed": self.config.seed,
                    "rank": self.config.soara_rank,
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                },
            )
        # If wandb.run exists (from sweep), enable logging
        elif wandb.run is not None:
            self.config.use_wandb = True
        
        best_metric = -float("inf")
        best_results = {}
        
        global_step = 0
        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss, avg_grad_norm, global_step = self._train_epoch(epoch, global_step)
            
            # Evaluate
            results = self._evaluate()
            
            # Get primary metric
            if self.task_config["metric"] == "accuracy":
                primary_metric = results.get("accuracy", 0)
            elif self.task_config["metric"] == "matthews":
                primary_metric = results.get("matthews", 0)
            elif self.task_config["metric"] == "acc_f1":
                primary_metric = results.get("acc_f1", 0)
            elif self.task_config["metric"] == "spearman":
                primary_metric = results.get("corr", 0)
            else:
                primary_metric = results.get("accuracy", 0)
            
            if primary_metric > best_metric:
                best_metric = primary_metric
                best_results = results.copy()
            
            # Log to wandb
            if self.config.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "avg_grad_norm": avg_grad_norm,
                    **{f"val_{k}": v for k, v in results.items()},
                    "best_metric": best_metric,
                }
                wandb.log(log_dict)
            
            print(f"Epoch {epoch + 1}/{self.config.epochs} - Loss: {train_loss:.4f} - Metric: {primary_metric:.4f} - Grad Norm: {avg_grad_norm:.4f}")
        
        if self.config.use_wandb:
            wandb.finish()
        
        return best_results
    
    def _train_epoch(self, epoch: int, global_step: int = 0) -> Tuple[float, float, int]:
        """Train for one epoch. Returns (avg_loss, avg_grad_norm, final_global_step)."""
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_grad_norm = 0
        num_steps = 0  # Local steps for average calc
        num_logging_steps = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Add orthogonality regularization for v1
            if self.soara_config.method == "v1":
                ortho_loss = self.rotational_trainer.get_orthogonality_loss()
                loss = loss + ortho_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            # Always clip for stability, but optionally track norm
            # If tracking is disabled, we skip the expensive .item() synchronization
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimization: Only sync to CPU (item()) if we are going to log
            # Use global_step for logging frequency
            is_logging_step = (global_step % self.config.logging_steps == 0)
            
            if is_logging_step:
                # Efficiently handle grad norm tracking
                if self.config.track_grad_norm:
                    grad_norm_val = grad_norm.item()
                    loss_val = loss.item() # Sync loss too if tracking is robust
                    total_grad_norm += grad_norm_val
                else:
                    grad_norm_val = 0.0
                    # Note: We skip total_grad_norm accumulation if not tracking to avoid mix of real/zero values
                
                # loss.item() is also a sync, but usually we want loss logged.
                # Assuming user prioritized grad norm sync removal.
                # We will still sync loss for wandb.
                loss_val = loss.item() 
                num_logging_steps += 1

            num_steps += 1
            global_step += 1
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update SOARA phase (for V2 (SOARA-V2)) using GLOBAL step
            if self.soara_config.method.upper() == "V2" and self.rotational_trainer.should_step_phase(global_step):
                self.rotational_trainer.step_phase()
            
            # Accumulate loss on GPU to avoid sync
            total_loss += loss.detach()
            
            # Log per-step metrics to wandb
            if self.config.use_wandb and is_logging_step:
                wandb.log({
                    "train/loss": loss_val,
                    "train/grad_norm": grad_norm_val,
                    "train/lr": self.scheduler.get_last_lr()[0],
                })
            
            # Stop if max_steps reached
            if self.config.max_steps > 0 and num_steps >= self.config.max_steps:
                break
        
        avg_loss = total_loss.item() / len(self.train_loader)
        avg_grad_norm = total_grad_norm / max(1, num_logging_steps)
        
        return avg_loss, avg_grad_norm, global_step
    
    def _evaluate(self, loader=None) -> Dict[str, float]:
        """Evaluate on validation set."""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                
                outputs = self.model(**batch)
                
                if self.task_config["is_regression"]:
                    predictions = outputs.logits.squeeze(-1)
                else:
                    predictions = outputs.logits
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        results = compute_metrics(self.config.task, predictions, labels)
        
        # For MNLI, also evaluate on mismatched
        if self.config.task == "mnli" and loader == self.val_loader:
            mm_results = self._evaluate(self.val_loader_mm)
            results["accuracy_mm"] = mm_results["accuracy"]
            results["m_mm"] = (results["accuracy"] + results["accuracy_mm"]) / 2
        
        return results


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def run_single_experiment(config: GLUEConfig) -> Dict[str, float]:
    """Run a single experiment and return results."""
    print("Running experiment with config:", config)
    trainer = GLUETrainer(config)
    return trainer.train()


def main():
    parser = argparse.ArgumentParser(description="GLUE Benchmark with DeBERTaV3 + SOARA")
    
    # Task and method
    parser.add_argument("--task", type=str, default="cola",
                        choices=list(GLUE_TASKS.keys()),
                        help="GLUE task to run")
    parser.add_argument("--method", type=str, default="v1",
                        choices=["v1", "v2", "V2", "v3", "V3", "v4", "V4", "all"],
                        help="SOARA method")
    
    # Model and training
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X steps")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--orthogonality_weight", type=float, default=1e-4)
    parser.add_argument("--total_cycles", type=int, default=3,
                        help="For V2: how many full cycles through all layers")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=-1, help="Limit number of steps per epoch for debugging")
    parser.add_argument("--no-pooler", action="store_true",
                        help="Remove pooler, use [CLS] token directly for classification")
    parser.add_argument("--use-butterfly", action="store_true",
                        help="For V2: use butterfly parameterization (log(r) layers) instead of sequential Givens")
    parser.add_argument("--butterfly-sequential", action="store_true",
                        help="If True, train butterfly components one at a time (like Givens sequential mode)")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./glue_results")
    parser.add_argument("--no-wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Determine rank logic
    if args.rank is None:
        if args.use_butterfly:
            # For butterfly, default to full rank d_model
            try:
                temp_config = AutoConfig.from_pretrained(args.model)
                args.rank = temp_config.hidden_size
                print(f"🦋 Butterfly mode: defaulting rank to d_model={args.rank}")
            except Exception as e:
                print(f"⚠️ Could not determine d_model from config, defaulting to 128: {e}")
                args.rank = 128
        else:
            # Default for standard SOARA
            args.rank = 8

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Single task, single method
    config = GLUEConfig(
        task=args.task,
        method=args.method,
        model_name=args.model,
        soara_rank=args.rank,
        orthogonality_weight=args.orthogonality_weight,
        total_cycles=args.total_cycles,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs= args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        seed=args.seed,
        no_pooler=args.no_pooler,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        use_butterfly=args.use_butterfly,
        butterfly_sequential=args.butterfly_sequential,
    )
    
    results = run_single_experiment(config)
    print(f"\nFinal results: {results}")


if __name__ == "__main__":
    main()
