#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# echo "=== COMMAND 1: SOARA-V1 (r=128, lambda_ortho=0) ==="
# python train_llama_commonsense_temp.py \
#   --task siqa --method v1 --lora_rank 128 \
#   --epochs 3 --learning_rate 1e-4 --real_batch_size 16 \
#   --per_device_batch_size 16 --max_length 512 --s_lr_multiplier 10 > logs/siqa_cmd1.log 2>&1

# echo "=== COMMAND 2: SOARA-V2a (r=None, givens) ==="
# python train_llama_commonsense_temp.py \
#   --task siqa --method V2 --lora_rank 128 \
#   --use_butterfly False --butterfly_sequential False \
#   --epochs 3 --learning_rate 1e-4 --real_batch_size 16 \
#   --per_device_batch_size 16 --max_length 512 \
#   --total_cycles 3 --s_lr_multiplier 10 > logs/siqa_cmd2.log 2>&1

echo "=== COMMAND 3: SOARA-V2b (r=None, butterfly sequential) ==="
python train_llama_commonsense_temp.py \
  --task siqa --method V2 \
  --use_butterfly True --butterfly_sequential True \
  --epochs 3 --learning_rate 1e-4 --real_batch_size 16 \
  --per_device_batch_size 16 --max_length 512 \
  --total_cycles 3 --s_lr_multiplier 10 > logs/siqa_cmd3.log 2>&1

echo "All 3 siqa experiments completed successfully!"