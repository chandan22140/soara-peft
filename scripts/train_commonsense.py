"""
Training Script for LLaMA 3 on Commonsense QA Datasets using SOARA Adapter
==========================================================================
"""
import torch
import os
from fire import Fire
from datasets import load_dataset
from accelerate import Accelerator
import wandb

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

from soara import (
    SOARAConfig,
    replace_linear_with_soara,
)
from utils import (
    initialize_text_to_text_model,
    find_all_linear_modules,
    train_text_to_text_model,
)

# ============================================================================
# CausalLM FORMATTER
# ============================================================================

def format_commonsense(task: str, example: dict) -> dict:
    """
    Wraps the evaluation logic perfectly into X (Prompt) and Y (Target Value).
    The causal LM trainer calculates CrossEntropy on Y tokens.
    """
    if task == "boolq":
        prompt = f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer exactly 'True' or 'False'.\nAnswer: "
        gt = example['label']  
        y = "True" if gt == 1 else "False"

    elif task == "piqa":
        prompt = f"Goal: {example['goal']}\nA. {example['sol1']}\nB. {example['sol2']}\nWhich option makes more sense? Answer exactly 'A' or 'B'.\nAnswer: "
        gt = example['label']
        y = "B" if gt == 1 else "A"

    elif task == "siqa":
        prompt = f"Context: {example['context']}\nQuestion: {example['question']}\nA. {example['answerA']}\nB. {example['answerB']}\nC. {example['answerC']}\nAnswer exactly 'A', 'B', or 'C'.\nAnswer: "
        gt = int(example['label']) - 1
        y = chr(ord('A') + gt)

    elif task.startswith("arc") or task == "obqa":
        question = example["question_stem"] if task == "obqa" else example["question"]
        choices_text = example["choices"]["text"]
        choices_labels = example["choices"]["label"]
        
        try:
            gt = choices_labels.index(example["answerKey"])
        except ValueError:
            gt = 0
            
        prompt = f"Question: {question}\n"
        letters = []
        for i, text in enumerate(choices_text):
            letter = chr(ord('A') + i)
            letters.append(letter)
            prompt += f"{letter}. {text}\n"
        
        prompt += f"Answer exactly one of the options: {', '.join(letters)}.\nAnswer: "
        y = chr(ord('A') + gt)

    elif task == "hellaswag":
        prompt = f"Context: {example['ctx']}\nSelect the most logical continuation:\n"
        choices = example["endings"]
        for i, text in enumerate(choices):
            prompt += f"{chr(ord('A') + i)}. {text}\n"
        
        try:
            gt = int(example['label']) if example.get('label') else 0
        except (ValueError, TypeError):
            gt = 0
            
        prompt += "Answer exactly 'A', 'B', 'C', or 'D'.\nAnswer: "
        y = chr(ord('A') + gt)

    elif task == "winogrande":
        prompt = f"Sentence: {example['sentence']}\nA. {example['option1']}\nB. {example['option2']}\nWhich option best fills the blank? Answer exactly 'A' or 'B'.\nAnswer: "
        gt = int(example['answer']) - 1
        y = chr(ord('A') + gt)

    else:
        raise ValueError(f"Task {task} not supported.")
        
    return {"x": prompt, "y": y}

# ============================================================================
# DATASET WRAPPER
# ============================================================================
from pathlib import Path

def _safe_load_dataset(dataset_name: str, config: str = None):
    load_kwargs = {
        "trust_remote_code": True,
    }

    try:
        if config is None:
            return load_dataset(dataset_name, **load_kwargs)
        return load_dataset(dataset_name, config, **load_kwargs)
    except Exception as exc:
        print(f"[DatasetRetry] Initial load failed for {dataset_name} ({config}): {exc}")

        # Retry with a clean cache path and forced download to bypass stale metadata.
        retry_cache_dir = Path("./data_cache/hf_retry") / dataset_name.replace("/", "_")
        retry_cache_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs["cache_dir"] = str(retry_cache_dir)
        load_kwargs["download_mode"] = "force_redownload"

        if config is None:
            return load_dataset(dataset_name, **load_kwargs)
        return load_dataset(dataset_name, config, **load_kwargs)

def prep_dataset(task: str):
    if task == "boolq":
        ds = _safe_load_dataset("super_glue", "boolq")
    elif task == "piqa":
        ds = _safe_load_dataset("piqa")
    elif task == "siqa":
        ds = _safe_load_dataset("social_i_qa")
    elif task == "arc-c":
        ds = _safe_load_dataset("ai2_arc", "ARC-Challenge")
    elif task == "arc-e":
        ds = _safe_load_dataset("ai2_arc", "ARC-Easy")
    elif task == "obqa":
        ds = _safe_load_dataset("openbookqa", "main")
    elif task == "hellaswag":
        ds = _safe_load_dataset("hellaswag")
    elif task == "winogrande":
        ds = _safe_load_dataset("winogrande", "winogrande_xl")
    else:
        raise ValueError("Invalid task.")
        
    train_ds = ds["train"].map(lambda x: format_commonsense(task, x))
    val_ds = ds["validation"].map(lambda x: format_commonsense(task, x))
    
    # Strip everything down to `x` and `y` securely for causal modeling tokenization 
    train_ds = train_ds.select_columns(["x", "y"])
    val_ds = val_ds.select_columns(["x", "y"])
    
    return train_ds, val_ds, val_ds

# ============================================================================
# MAIN
# ============================================================================

def main(
    task="boolq", 
    lora_alpha=128, 
    lora_rank=None, 
    sample_size=128, 
    seed=42, 
    resume_from_checkpoint=None, 
    track_grad_norm=False, 
    method="v1", 
    total_cycles=4, 
    epochs=1.0, 
    use_butterfly=False, 
    butterfly_sequential=False,
    learning_rate=2e-4,
    real_batch_size=128,
    per_device_batch_size=1,
    max_length=512,
    s_lr_multiplier=10.0,
):
    accelerator = Accelerator()
    model_id = "meta-llama/Meta-Llama-3-8B" 
    model_type = "CausalLM"
    model_dtype = "bf16"

    if lora_rank is None:
        if use_butterfly:
            from transformers import AutoConfig
            config_obj = AutoConfig.from_pretrained(model_id)
            lora_rank = config_obj.hidden_size
            if accelerator.is_local_main_process:
                print(f"🦋 Butterfly mode: defaulting rank to d_model={lora_rank}")
        else:
            lora_rank = 128
            
    config = dict(
        model=model_id.replace("/", "_"),
        d=task,
        a=lora_alpha,
        r=lora_rank,
        s=sample_size,
        sd=seed,
        method=method,
        butterfly=use_butterfly,
        seq=butterfly_sequential,
    )
    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])
    if butterfly_sequential:
         wandb_name = "butterfly_seq_" + wandb_name
    
    if accelerator.is_local_main_process:
        wandb.init(
            name=wandb_name,
            mode="online",
            group="commonsense_tuning",
            project="LLaMA SOARA Commonsense",
        )
        
    model, tokenizer = initialize_text_to_text_model(
        model_id, model_type, model_dtype, flash_attention=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_local_main_process:
        print(model)

    soara_config = SOARAConfig(
        r=config["r"],
        lora_alpha=config["a"],
        method=method,
        total_cycles=total_cycles,
        use_butterfly=use_butterfly,
        butterfly_sequential=butterfly_sequential,
        orthogonality_reg_weight=0,   
        init_identity=True,
        freeze_singular_values=False,
        quantize_residual=False,
        quantize_base_components=False,
    )

    train_set, val_set, _ = prep_dataset(task)

    if accelerator.is_local_main_process:
        print(soara_config)

    print(f"[Rank {accelerator.process_index}] Starting SOARA initialization on {accelerator.device}...")
    adapters = replace_linear_with_soara(
        model=model,
        soara_config=soara_config,
        target_modules=find_all_linear_modules(model=model),
        adapter_name="default",
        freeze_base_model=True,
        device=accelerator.device,
    )
    print(f"[Rank {accelerator.process_index}] SOARA initialization complete.")

    save_dir = os.path.join("./snapshot/commonsense", wandb_name)
    if accelerator.is_local_main_process:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'soara_config': soara_config,
            'adapters': list(adapters.keys()),
        }, os.path.join(save_dir, "init_checkpoint.pt"))

    model = train_text_to_text_model(
        run_name=os.path.join("commonsense_soara", wandb_name),
        train_dataset=train_set,
        valid_dataset=val_set,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        num_train_epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        real_batch_size=real_batch_size,
        bf16=(model_dtype == "bf16"),
        eval_epochs=0.1,         # Validates mid-epoch
        early_stopping_patience=3,
        max_length=max_length,
        logging_steps=1,
        use_loraplus=False,
        loraplus_lr_ratio=None,
        learning_rate=learning_rate,
        num_process=accelerator.num_processes,
        gradient_checkpointing=False,
        seed=seed,
        soara_config=soara_config,
        s_lr_multiplier=s_lr_multiplier,
        resume_from_checkpoint=resume_from_checkpoint,
        save_total_limit=2,
        training_args=dict(
            lr_scheduler_type="cosine",
            adam_epsilon=1e-10,
            max_grad_norm=1.0 if track_grad_norm else 0.0,
            warmup_ratio=0.03,
            weight_decay=0.0,
            torch_compile=False,
        ),
    )
    
    if accelerator.is_local_main_process:
        torch.save({
            'model_state_dict': model.state_dict(),
            'soara_config': soara_config,
            'adapters': list(adapters.keys()),
        }, os.path.join(save_dir, "final_checkpoint.pt"))

if __name__ == "__main__":
    Fire(main)
