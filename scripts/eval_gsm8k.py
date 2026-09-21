"""
GSM8K Evaluation Script with Robust Answer Extraction
======================================================

This script evaluates LLM performance on the GSM8K test set using vLLM for fast inference.
It supports multiple answer extraction strategies:
1. GSM8K format: #### <number>
2. Boxed format: \\boxed{<answer>}
3. Numerical fallback: last number in text

Logs all samples with question, ground truth, generated text, and extracted answer.
"""

from data import load_gsm8k
from fire import Fire
import re
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
from typing import Optional, Union


# ============================================================================
# ANSWER EXTRACTION FUNCTIONS
# (Copied from train_llm_rotational.py for robustness)
# ============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{} notation.
    
    Args:
        text: Generated text containing boxed answer
        
    Returns:
        Extracted answer string or None if not found
        
    Examples:
        "\\boxed{42}" -> "42"
        "\\boxed{3.14}" -> "3.14"
        "\\boxed{\\frac{1}{2}}" -> "0.5"
    """
    # Try to find \boxed{...}
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)
    
    if matches:
        answer = matches[-1].strip()  # Take last boxed answer
        
        # Convert LaTeX fractions to decimals
        frac_pattern = r'\\frac\{(\d+)\}\{(\d+)\}'
        frac_match = re.search(frac_pattern, answer)
        if frac_match:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            return str(numerator / denominator)
        
        # Convert dfrac (display fraction) to decimals
        dfrac_pattern = r'\\dfrac\{(\d+)\}\{(\d+)\}'
        dfrac_match = re.search(dfrac_pattern, answer)
        if dfrac_match:
            numerator = float(dfrac_match.group(1))
            denominator = float(dfrac_match.group(2))
            return str(numerator / denominator)
        
        # Remove LaTeX formatting
        answer = answer.replace('$', '').replace(',', '').strip()
        
        return answer
    
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Extract answer from GSM8K format: #### {number}
    
    Args:
        text: Generated text containing #### answer
        
    Returns:
        Extracted numerical answer string or None if not found
        
    Examples:
        "Step 1...\\n#### 42" -> "42"
        "The answer is 100.\\n#### 100" -> "100"
    """
    # GSM8K format: #### followed by number
    pattern = r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern, text)
    
    if match:
        # Remove commas from number
        answer = match.group(1).replace(',', '').strip()
        return answer
    
    return None


def extract_numerical_answer(text: str) -> Optional[float]:
    """
    Extract final numerical answer from text using multiple strategies.
    
    Fallback extraction when \\boxed{} and #### are not present:
    1. Look for "Final Answer", "Therefore", "The answer is"
    2. Extract last numerical value from last sentence
    3. Handle scientific notation, percentages, currencies
    
    Args:
        text: Generated text
        
    Returns:
        Numerical answer as float or None if not found
    """
    # Strategy 1: Look for explicit answer markers
    answer_markers = [
        r'Final Answer[:\s]+([+-]?\d+\.?\d*)',
        r'Therefore[,\s]+(?:the answer is\s+)?([+-]?\d+\.?\d*)',
        r'The answer is[:\s]+([+-]?\d+\.?\d*)',
        r'Answer[:\s]+([+-]?\d+\.?\d*)',
        r'= ([+-]?\d+\.?\d*)\s*$',  # Equals at end of line
    ]
    
    for pattern in answer_markers:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # Strategy 2: Extract from boxed notation first
    boxed = extract_boxed_answer(text)
    if boxed:
        try:
            # Clean and convert
            cleaned = re.sub(r'[^\d\.\-\+]', '', boxed)
            return float(cleaned)
        except ValueError:
            pass
    
    # Strategy 3: Extract last number from last few sentences
    sentences = text.split('.')
    for sentence in reversed(sentences[-5:]):  # Check last 5 sentences
        # Find all numbers (including decimals, negatives)
        numbers = re.findall(r'[+-]?\d+\.?\d*', sentence)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                continue
    
    return None


def normalize_answer(answer: Union[str, float, int, None]) -> str:
    """
    Normalize answer to standard format for comparison.
    
    Args:
        answer: Raw answer (string, float, int, or None)
        
    Returns:
        Normalized string representation
    """
    if answer is None:
        return "INVALID"
    
    # Convert to string
    if isinstance(answer, (int, float)):
        answer_str = str(answer)
    else:
        answer_str = str(answer)
    
    # Remove whitespace and common formatting
    answer_str = answer_str.strip().lower()
    answer_str = answer_str.replace(',', '').replace('$', '').replace('%', '')
    
    # Try to convert to float for numerical comparison
    try:
        num = float(answer_str)
        # Round to 2 decimal places for comparison
        return f"{num:.2f}"
    except ValueError:
        return answer_str


def extract_answer(text: str) -> tuple:
    """
    Extract answer using multiple strategies.
    
    Returns:
        (extracted_value, method_used) tuple
        
    Priority:
    1. GSM8K format: #### <number>
    2. Boxed format: \\boxed{<answer>}
    3. Numerical fallback: last number in text
    """
    # Strategy 1: GSM8K format (highest priority for GSM8K eval)
    gsm8k_answer = extract_gsm8k_answer(text)
    if gsm8k_answer:
        try:
            return int(float(gsm8k_answer.replace(',', ''))), "gsm8k"
        except ValueError:
            pass
    
    # Strategy 2: Boxed format
    boxed_answer = extract_boxed_answer(text)
    if boxed_answer:
        try:
            cleaned = re.sub(r'[^\d\.\-\+]', '', boxed_answer)
            return int(float(cleaned)), "boxed"
        except ValueError:
            pass
    
    # Strategy 3: Numerical fallback
    numerical_answer = extract_numerical_answer(text)
    if numerical_answer is not None:
        try:
            return int(numerical_answer), "numerical"
        except (ValueError, OverflowError):
            pass
    
    return 0, "none"


def main(model_name, eval_seed=0, temperature=0.0, bsz=4, tokenizer_name=None, 
         gpu_memory_utilization=0.8, max_num_seqs=64, log_file=None, verbose=True):
    """
    Evaluate model on GSM8K test set.
    
    Args:
        model_name: Path to model or HuggingFace model name
        eval_seed: Random seed for sampling
        temperature: Sampling temperature
        bsz: Batch size
        tokenizer_name: Optional tokenizer name (defaults to model_name)
        gpu_memory_utilization: vLLM GPU memory utilization
        max_num_seqs: Maximum concurrent sequences for vLLM
        log_file: Optional file to log all samples (default: gsm8k_eval_log.txt)
        verbose: Print each sample's details
    """
    _, _, test_set = load_gsm8k()
    
    # Setup tokenizer
    tokenizer = tokenizer_name if tokenizer_name else model_name
    
    print(f"Loading model: {model_name}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Max num seqs: {max_num_seqs}")
    
    model = LLM(model_name, dtype="bfloat16", seed=eval_seed, tokenizer=tokenizer,
                gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs)
    
    sampling_params = SamplingParams(
        top_p=0.95, temperature=temperature, max_tokens=1024
    )
    
    # Setup logging
    if log_file is None:
        log_file = "gsm8k_eval_log.txt"
    
    total = 0
    correct = 0
    extraction_stats = {"gsm8k": 0, "boxed": 0, "numerical": 0, "none": 0}
    
    # Open log file
    with open(log_file, "w") as log:
        log.write(f"GSM8K Evaluation Log\n")
        log.write(f"Model: {model_name}\n")
        log.write(f"Temperature: {temperature}, Seed: {eval_seed}\n")
        log.write("=" * 80 + "\n\n")
        
        t = tqdm(range(0, len(test_set), bsz))
        for idx in t:
            examples = test_set[idx : idx + bsz]
            prompts = examples["x"]
            outputs = model.generate(
                prompts, sampling_params=sampling_params, use_tqdm=False
            )
            
            for i, (question, y, output) in enumerate(zip(prompts, examples["y"], outputs)):
                pred_text = output.outputs[0].text
                
                # Extract ground truth
                gt_answer = extract_gsm8k_answer(y)
                if gt_answer:
                    gt = int(float(gt_answer.replace(',', '')))
                else:
                    gt = 0
                
                # Extract prediction using robust extraction
                pred, extraction_method = extract_answer(pred_text)
                extraction_stats[extraction_method] += 1
                
                # Check correctness
                is_correct = (gt == pred)
                correct += int(is_correct)
                total += 1
                
                # Log sample details
                log.write(f"--- Sample {total} ---\n")
                log.write(f"QUESTION:\n{question}\n\n")
                log.write(f"GROUND TRUTH (full): {y}\n")
                log.write(f"GROUND TRUTH ANSWER: {gt}\n\n")
                log.write(f"GENERATED:\n{pred_text}\n\n")
                log.write(f"EXTRACTED ANSWER: {pred} (method: {extraction_method})\n")
                log.write(f"CORRECT: {'✓' if is_correct else '✗'}\n")
                log.write("=" * 80 + "\n\n")
                
                # Verbose console output
                if verbose:
                    status = "✓" if is_correct else "✗"
                    print(f"\n[{total}] {status} GT: {gt}, Pred: {pred} ({extraction_method})")
                    if not is_correct:
                        print(f"  Question: {question}")
                        print(f"  Prediction: {pred_text}")
                
                t.set_description(f"Accuracy: {correct/total*100:.2f}%")
        
        # Final summary
        final_acc = correct / total if total > 0 else 0
        log.write(f"\n{'=' * 80}\n")
        log.write(f"FINAL RESULTS\n")
        log.write(f"{'=' * 80}\n")
        log.write(f"Total samples: {total}\n")
        log.write(f"Correct: {correct}\n")
        log.write(f"Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)\n\n")
        log.write(f"Extraction method statistics:\n")
        for method, count in extraction_stats.items():
            pct = count / total * 100 if total > 0 else 0
            log.write(f"  {method}: {count} ({pct:.1f}%)\n")
    
    print(f"\n{'=' * 60}")
    print(f"FINAL ACCURACY: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"Correct: {correct}/{total}")
    print(f"\nExtraction statistics:")
    for method, count in extraction_stats.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {method}: {count} ({pct:.1f}%)")
    print(f"\nDetailed log saved to: {log_file}")
    
    # Append to results file
    if not os.path.exists("gsm8k_results.txt"):
        with open("gsm8k_results.txt", "w") as f:
            f.write("Model\teval_seed\ttemperature\taccuracy\n")
    
    with open("gsm8k_results.txt", "a") as f:
        f.write(f"{model_name}\t{eval_seed}\t{temperature}\t{final_acc}\n")
    
    return final_acc


if __name__ == "__main__":
    Fire(main)
