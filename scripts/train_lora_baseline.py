#!/usr/bin/env python3
"""
Simple LoRA Fine-tuning Pipeline for Llama-2-7B on MetaMathQA
==============================================================

This script provides a minimal, clean pipeline for fine-tuning Llama-2-7B
using LoRA (rank=2) on the MetaMathQA dataset.

Usage:
    python train_lora_baseline.py
    
    # With custom settings
    python train_lora_baseline.py --rank 4 --epochs 1 --batch-size 2
"""

import os
import sys
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Llama-2-7B")
    
    # Model
    parser.add_argument("--model", type=str, default="google/gemma-7b",
                        help="Model name or path")
    
    # LoRA
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--alpha", type=float, default=16.0,
                        help="LoRA alpha (default: 16.0)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="LoRA dropout (default: 0.0)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per device (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=32,
                        help="Gradient accumulation steps (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    
    # Data
    parser.add_argument("--max-train-samples", type=int, default=10000,
                        help="Maximum training samples (default: 10000)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./outputs/lora_baseline",
                        help="Output directory")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    return parser.parse_args()


# ============================================================================
# DATA PROCESSING
# ============================================================================

def format_prompt(question: str) -> str:
    """Format question as instruction prompt."""
    return (
        "Below is a math question. Solve it step by step.\n\n"
        f"### Question:\n{question}\n\n"
        "### Answer:\n"
    )


def preprocess_function(examples, tokenizer, max_length):
    """Tokenize and prepare dataset examples."""
    
    prompts = []
    responses = []
    
    for question, answer in zip(examples["query"], examples["response"]):
        prompt = format_prompt(question)
        prompts.append(prompt)
        responses.append(answer)
    
    # Combine prompt + response
    texts = [p + r + tokenizer.eos_token for p, r in zip(prompts, responses)]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # Create labels (mask prompt tokens with -100)
    prompt_tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    labels = []
    for i, input_ids in enumerate(tokenized["input_ids"]):
        prompt_len = len(prompt_tokenized["input_ids"][i])
        label = [-100] * prompt_len + input_ids[prompt_len:]
        labels.append(label)
    
    tokenized["labels"] = labels
    
    return tokenized


def load_metamath_dataset(tokenizer, max_length, max_samples):
    """Load and preprocess MetaMathQA dataset."""
    
    print("📚 Loading MetaMathQA dataset...")
    
    # Load dataset
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    
    # Filter to GSM8K-like samples (optional, for faster training)
    dataset = dataset.filter(lambda x: "GSM" in x.get("type", ""))
    
    # Limit samples
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    print(f"  ✓ Loaded {len(dataset)} samples")
    
    # Preprocess
    print("  Processing dataset...")
    dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    return dataset


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer(model_name, lora_rank, lora_alpha, lora_dropout):
    """Load model and apply LoRA."""
    
    print(f"\n🔧 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model in bf16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Loaded model: {total_params} parameters")
    
    # Configure LoRA
    print(f"\n🔗 Applying LoRA (rank={lora_rank}, alpha={lora_alpha})")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
    
    return model, tokenizer


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    """Main training function."""
    
    print("=" * 60)
    print("LoRA Fine-tuning: Llama-2-7B on MetaMathQA")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model,
        args.rank,
        args.alpha,
        args.dropout,
    )
    
    # Load dataset
    train_dataset = load_metamath_dataset(
        tokenizer,
        args.max_seq_length,
        args.max_train_samples,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",  # Disable wandb
        seed=args.seed,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Print training config
    print(f"\n📋 Training Configuration:")
    print(f"  • Epochs: {args.epochs}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Gradient accumulation: {args.grad_accum}")
    print(f"  • Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  • Learning rate: {args.lr}")
    print(f"  • Max sequence length: {args.max_seq_length}")
    print(f"  • LoRA rank: {args.rank}")
    
    # Train
    print("\n🚀 Starting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    trainer.train()
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"\n⚡ Peak VRAM usage: {peak_memory:.2f} GB")
    
    # Save model
    print(f"\n💾 Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n✅ Training complete!")
    
    return trainer


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
