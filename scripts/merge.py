from fire import Fire
# CHANGED: Import SOARA instead of PEFT
# from peft import PeftModel
from rotational_pissa_unified import replace_linear_with_soara, SOARAConfig
from utils import initialize_text_to_text_model
import os
import torch
import torch.nn as nn
import concurrent.futures
from functools import partial
# CHANGED: No need for bnb LoRA layers since we use SOARA
# from peft.tuners.lora.bnb import (
#     Linear8bitLt as LoraLinear8bitLt,
#     Linear4bit as LoraLinear4bit,
# )


def get_float_weight(model: torch.nn.Module):
    model: torch.nn.Linear

    device = model.weight.device
    in_features = model.in_features
    with torch.no_grad():
        I = torch.eye(in_features).to(device)
        w = model(I)
        if hasattr(model, "bias") and isinstance(model.bias, torch.Tensor):
            w -= model.bias
        w = torch.transpose(w, 0, 1)
    w.requires_grad = model.weight.requires_grad
    return w


def replace_A_with_Linear(model: torch.nn.Module, target):
    for name, module in model.named_children():
        if isinstance(module, target):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            new_module = torch.nn.Linear(in_features, out_features, bias)
            with torch.no_grad():
                new_module.weight.data = get_float_weight(module).data
                if bias:
                    new_module.bias.data = (
                        module.bias if module.bias is not None else None
                    )
            setattr(model, name, new_module)

        else:
            replace_A_with_Linear(module, target)


def dequantize(model, dtype):
    # CHANGED: This function is kept for compatibility but may not be needed
    # for SOARA if we don't use quantization
    try:
        from bitsandbytes.nn import Linear8bitLt as LoraLinear8bitLt
        from bitsandbytes.nn import Linear4bit as LoraLinear4bit
        
        if dtype == "int8":
            target = LoraLinear8bitLt
        elif dtype == "nf4":
            target = LoraLinear4bit
        replace_A_with_Linear(model=model, target=target)
    except ImportError:
        print("Warning: bitsandbytes not available, skipping dequantization")


def process_layer_merge_task(layer_prefix, checkpoint_state_dict, soara_config, device="cpu", merged_layer_count=0):
    """
    Worker function to merge a single layer's weights.
    Returns: (layer_prefix, merged_state_dict_subset, logs)
    """
    local_state = {}
    logs = []
    
    # Get all components for this layer
    S_key = f"{layer_prefix}.S"
    R_U_key = f"{layer_prefix}.R_U"
    R_V_key = f"{layer_prefix}.R_V"
    U_key = f"{layer_prefix}.U"
    V_key = f"{layer_prefix}.V"
    base_weight_key = f"{layer_prefix}.base_layer.weight"
    base_bias_key = f"{layer_prefix}.base_layer.bias"
    
    # Check if all required components exist
    if not all(k in checkpoint_state_dict for k in [S_key, U_key, V_key, base_weight_key]):
        logs.append(f"Warning: Missing components for {layer_prefix}, skipping")
        return layer_prefix, local_state, logs
    
    # Load components
    S = checkpoint_state_dict[S_key]  # [r]
    U = checkpoint_state_dict[U_key]  # [r, out_features]
    V = checkpoint_state_dict[V_key]  # [r, in_features]
    W_residual = checkpoint_state_dict[base_weight_key]  # [out_features, in_features]
    
    # Get rotation matrices - handle all 4 rotation methods
    r = S.shape[0]
    
    # Keys for v3/V4 parameters
    B_U_key = f"{layer_prefix}.B_U"
    C_U_key = f"{layer_prefix}.C_U"
    B_V_key = f"{layer_prefix}.B_V"
    C_V_key = f"{layer_prefix}.C_V"
    
    # Key for V2 Givens angles (after final step, these are merged into U/V)
    givens_u_key = f"{layer_prefix}.current_givens_u.thetas"
    givens_v_key = f"{layer_prefix}.current_givens_v.thetas"
    
    # Determine rotation method and compute R_U, R_V
    # V1 (SOARA-V1): R_U and R_V stored directly
    if R_U_key in checkpoint_state_dict and R_V_key in checkpoint_state_dict:
        R_U = checkpoint_state_dict[R_U_key]
        R_V = checkpoint_state_dict[R_V_key]
        if merged_layer_count == 0:
            logs.append(f"  Using rotation method: v1 (direct R_U, R_V)")
    
    # v3/3: Compute R from B, C matrices
    elif B_U_key in checkpoint_state_dict and C_U_key in checkpoint_state_dict:
        B_U = checkpoint_state_dict[B_U_key].to(S.dtype)
        C_U = checkpoint_state_dict[C_U_key].to(S.dtype)
        B_V = checkpoint_state_dict[B_V_key].to(S.dtype)
        C_V = checkpoint_state_dict[C_V_key].to(S.dtype)
        
        # Determine if v3 or V4 based on soara_config if available
        use_exp = False  # Default to v3
        if soara_config is not None:
            method = getattr(soara_config, 'method', 'v3')
            use_exp = (method == 'V4')
        
        if use_exp:
            # V4: R = exp(BC^T - CB^T)
            skew_U = B_U @ C_U.T - C_U @ B_U.T
            skew_V = B_V @ C_V.T - C_V @ B_V.T
            R_U = torch.matrix_exp(skew_U.float()).to(S.dtype)
            R_V = torch.matrix_exp(skew_V.float()).to(S.dtype)
            if merged_layer_count == 0:
                logs.append(f"  Using rotation method: V4 (exp of skew-symmetric)")
        else:
            # v3: R = I + BC^T - CB^T
            I = torch.eye(r, dtype=S.dtype, device=S.device)
            R_U = I + B_U @ C_U.T - C_U @ B_U.T
            R_V = I + B_V @ C_V.T - C_V @ B_V.T
            if merged_layer_count == 0:
                logs.append(f"  Using rotation method: v3 (I + BC^T - CB^T)")
    
    # V2 (SOARA-V2): Givens rotations - after training, rotations are merged into U/V
    # So we use identity here (the rotations are already in U/V)
    elif givens_u_key in checkpoint_state_dict:
        # Givens angles present - need to compute rotation matrix from angles
        # This is rare since step_phase() merges rotations into U/V
        # But we support it for completeness
        from rotational_pissa_unified import GivensRotationLayer, generate_givens_pairings
        
        thetas_u = checkpoint_state_dict[givens_u_key]
        thetas_v = checkpoint_state_dict[givens_v_key]
        
        # Get the pairings (same as what was used in training)
        n_layers = soara_config.n_givens_layers if soara_config else (r - 1)
        pairings = generate_givens_pairings(r, n_layers)[0]  # Use first layer pairings
        
        # Compute rotation matrices from Givens angles
        R_U = torch.eye(r, device=thetas_u.device, dtype=S.dtype)
        R_V = torch.eye(r, device=thetas_v.device, dtype=S.dtype)
        
        cos_u, sin_u = torch.cos(thetas_u), torch.sin(thetas_u)
        cos_v, sin_v = torch.cos(thetas_v), torch.sin(thetas_v)
        
        for i, (p, q) in enumerate(pairings):
            if p < r and q < r:
                R_U[p, p], R_U[q, q] = cos_u[i], cos_u[i]
                R_U[p, q], R_U[q, p] = -sin_u[i], sin_u[i]
                R_V[p, p], R_V[q, q] = cos_v[i], cos_v[i]
                R_V[p, q], R_V[q, p] = -sin_v[i], sin_v[i]
            
            if merged_layer_count == 0:
                logs.append(f"  Using rotation method: V2 (Givens angles)")
    
    else:
        # Default: identity (V2 after all rotations merged into U/V)
        R_U = torch.eye(r, dtype=S.dtype, device=S.device)
        R_V = torch.eye(r, dtype=S.dtype, device=S.device)
        if merged_layer_count == 0:
            logs.append(f"  Using rotation method: identity (V2 with rotations merged into U/V)")
    
    # Compute scaling
    # NOTE: In rotational_pissa_unified.py line 497, scaling = 1 (lora_alpha/r is commented out)
    # So we use 1.0 here, not lora_alpha / r
    scaling = 1.0
    
    with torch.no_grad():
        # Ensure everything is in the same dtype
        dtype = W_residual.dtype
        
        # Move to target device for computation
        S_dev = S.to(device).to(dtype)
        U_dev = U.to(device).to(dtype)  # [out_features, r]
        V_dev = V.to(device).to(dtype)  # [r, in_features]
        R_U_dev = R_U.to(device).to(dtype)  # [r, r]
        R_V_dev = R_V.to(device).to(dtype)  # [r, r]
        
        # Step 1: R_U @ diag(S) @ R_V = [r, r]
        middle = R_U_dev @ torch.diag(S_dev) @ R_V_dev  # [r, r] @ [r, r] @ [r, r] = [r, r]
        
        # Step 2: U @ middle @ V = [out_features, r] @ [r, r] @ [r, in_features] = [out_features, in_features]
        W_principal = U_dev @ middle @ V_dev  # [out_features, in_features]
        
        # Merge: W_merged = W_residual + scaling * W_principal
        # Note: In the training code, scaling = 1 (see line 497 in rotational_pissa_unified.py)
        # The lora_alpha/r scaling is commented out
        # Move result back to CPU (or same device as residual) for addition to avoid OOM on GPU if holding full model
        
        # VERY IMPORTANT: If we are multithreading on CPU, W_principal needs to be on CPU.
        # If device is cuda, we move it back.
        if device != "cpu":
             W_principal = W_principal.to("cpu")
             
        W_merged = W_residual + scaling * W_principal.to(W_residual.dtype)
    
    # Store merged weight with standard key (remove .base_layer)
    merged_weight_key = f"{layer_prefix}.weight"
    local_state[merged_weight_key] = W_merged
    
    # Handle bias if present
    if base_bias_key in checkpoint_state_dict:
        merged_bias_key = f"{layer_prefix}.bias"
        local_state[merged_bias_key] = checkpoint_state_dict[base_bias_key]
        
    return layer_prefix, local_state, logs


def merge_soara_weights(checkpoint_state_dict, base_model, soara_config=None, device="cpu", num_threads=1):
    """
    Merge SOARA adapter weights back into standard linear layer weights.
    
    The forward pass computes:
        Y = X @ W_residual + X @ V^T @ R_V^T @ S @ R_U^T @ U^T * scaling
    
    So the merged weight is:
        W_merged = W_residual + scaling * (U @ R_U @ diag(S) @ R_V^T @ V^T)
        
    Where:
        - W_residual is stored in base_layer.weight
        - U is [r, out_features], V is [r, in_features]
        - R_U, R_V are [r, r] rotation matrices
        - S is [r] singular values
        - scaling = lora_alpha / r
    """
    merged_state_dict = {}
    
    # Default config
    lora_alpha = 8.0 if soara_config is None else getattr(soara_config, 'lora_alpha', 8.0)
    
    # Find all SOARA layers by looking for .S parameters
    soara_layers = set()
    for key in checkpoint_state_dict.keys():
        if key.endswith('.S'):
            # print(f"Found SOARA layer: {key}")
            # Extract the layer prefix (e.g., "model.layers.0.self_attn.q_proj")
            layer_prefix = key[:-2]  # Remove ".S"
            # print(f"Layer prefix: {layer_prefix}")
            soara_layers.add(layer_prefix)
    
    if not soara_layers:
        print("No SOARA layers found. Returning checkpoint as-is.")
        return checkpoint_state_dict
    
    print(f"Found {len(soara_layers)} SOARA layers to merge")
    
    # Process each layer
    merged_layer_count = 0
    
    # Prepare tasks
    print(f"Merging {len(soara_layers)} layers with {num_threads} threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, layer_prefix in enumerate(soara_layers):
            # Pass i as merged_layer_count to allow logging only on first layer (approximately)
            futures.append(
                executor.submit(
                    process_layer_merge_task, 
                    layer_prefix, 
                    checkpoint_state_dict, 
                    soara_config, 
                    device, 
                    i
                )
            )
        
        for future in concurrent.futures.as_completed(futures):
            try:
                layer_prefix, local_state, logs = future.result()
                # Print any logs from the worker
                for log_msg in logs:
                    print(log_msg)
                
                # Update main dict
                merged_state_dict.update(local_state)
                merged_layer_count += 1
                
                if merged_layer_count % 10 == 0:
                    print(f"Merged {merged_layer_count}/{len(soara_layers)} layers", end="\r")
                    
            except Exception as e:
                print(f"Error merging layer: {e}")
                raise e
                
    print(f"\nMerged {merged_layer_count}/{len(soara_layers)} layers")
    
    # Copy over non-SOARA layers as-is
    for key, value in checkpoint_state_dict.items():
        # Skip SOARA-specific keys
        skip = False
        for layer_prefix in soara_layers:
            if key.startswith(layer_prefix + "."):
                skip = True
                break
        
        if not skip:
            merged_state_dict[key] = value
    
    print(f"Successfully merged {merged_layer_count} SOARA layers")
    return merged_state_dict


def merge(
    checkpoint: str,
    dtype: str = "bf16",
    model_name="google/gemma-3-1b-it",
    model_type="CausalLM",
    merge_suffix="merged_checkpoint",
    lora_alpha: float = 8.0,
    snapshot_path: str = None,
    device: str = "cpu",
    num_threads: int = 16, # Default to 16 threads for high CPU usage
):
    """
    Merge SOARA adapters back into base model weights.
    
    Args:
        checkpoint: Path to the checkpoint directory containing pytorch_model.bin
        dtype: Model dtype (bf16, fp32, etc.)
        model_name: Base model identifier
        model_type: Model type (CausalLM, ConditionalGeneration)
        merge_suffix: Suffix for merged checkpoint directory
        lora_alpha: LoRA alpha scaling factor used during training
        snapshot_path: Path to the snapshot folder containing final_checkpoint.pt (if different from checkpoint)
    """
    print(f"Loading checkpoint from: {checkpoint}")
    
    # Find the checkpoint file
    index_file = os.path.join(checkpoint, "pytorch_model.bin.index.json")
    pytorch_model_path = os.path.join(checkpoint, "pytorch_model.bin")
    
    # final_checkpoint.pt may be in a different folder (snapshot_path)
    if snapshot_path:
        final_checkpoint_path = os.path.join(snapshot_path, "final_checkpoint.pt")
        init_checkpoint_path = os.path.join(snapshot_path, "init_checkpoint.pt")
        print(f"Using snapshot path: {snapshot_path}")
    else:
        final_checkpoint_path = os.path.join(checkpoint, "final_checkpoint.pt")
        init_checkpoint_path = os.path.join(checkpoint, "init_checkpoint.pt")
    
    checkpoint_data = {}
    soara_config = None
    
    # Priority 1: Sharded Checkpoint
    if os.path.exists(index_file):
        print(f"Found sharded checkpoint index: {index_file}")
        import json
        with open(index_file, "r") as f:
            index = json.load(f)
        
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        
        print(f"Loading {len(shard_files)} shards with {num_threads} threads...")
        
        def load_shard(shard_file):
            shard_path = os.path.join(checkpoint, shard_file)
            print(f"Loading shard: {shard_path}")
            try:
                # Load to CPU to avoid OOM
                return torch.load(shard_path, map_location="cpu")
            except Exception as e:
                print(f"Error loading shard {shard_path}: {e}")
                raise e
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all load tasks
            future_to_shard = {
                executor.submit(load_shard, sf): sf for sf in shard_files
            }
            
            for future in concurrent.futures.as_completed(future_to_shard):
                shard_data = future.result()
                checkpoint_data.update(shard_data)
                del shard_data
                
        # Check if soara_config is in the directory (unlikely for standard sharded saves but check anyway)
        # Note: We rely on passed lora_alpha argument if config is missing
        
    # Priority 2: Custom final_checkpoint.pt (has soara_config metadata)
    elif os.path.exists(final_checkpoint_path):
        print(f"Loading from final_checkpoint.pt: {final_checkpoint_path}")
        loaded = torch.load(final_checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            print("loaded was a dict")
            checkpoint_data = loaded['model_state_dict']
            soara_config = loaded.get('soara_config')
        else:
            checkpoint_data = loaded

    # Priority 3: Monolithic pytorch_model.bin
    elif os.path.exists(pytorch_model_path):
        print(f"Loading from pytorch_model.bin")
        checkpoint_data = torch.load(pytorch_model_path, map_location='cpu')

    # Priority 4: Custom init_checkpoint.pt
    elif os.path.exists(init_checkpoint_path):
        print(f"Loading from init_checkpoint.pt: {init_checkpoint_path}")
        loaded = torch.load(init_checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            checkpoint_data = loaded['model_state_dict']
            soara_config = loaded.get('soara_config')
        else:
            checkpoint_data = loaded
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint} or {snapshot_path}")
    
    # Check if this is a SOARA checkpoint (has .S, .U, .V, .R_U, .R_V keys)
    has_soara_keys = any(k.endswith('.S') for k in checkpoint_data.keys())
    
    if has_soara_keys:
        print("Detected SOARA checkpoint. Merging adapters...")
        
        # Create a dummy config if not loaded
        class SimpleConfig:
            def __init__(self, alpha):
                self.lora_alpha = alpha
        
        if soara_config is None:
            soara_config = SimpleConfig(lora_alpha)
            print(f"Using lora_alpha={lora_alpha}")
        else:
            print(f"Using lora_alpha from config: {getattr(soara_config, 'lora_alpha', lora_alpha)}")
        
        # Merge the weights
        merged_state_dict = merge_soara_weights(
            checkpoint_data, 
            base_model=None,  # Not needed for pure merge
            soara_config=soara_config,
            device=device,
            num_threads=num_threads
        )
    else:
        print("No SOARA layers detected. Using checkpoint as-is.")
        merged_state_dict = checkpoint_data
    
    # Now load the base model and apply merged weights
    print(f"Loading base model: {model_name}")
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, dtype="bf16"
    )
    
    # Load the merged state dict
    print("Loading merged weights into model...")
    missing, unexpected = model.load_state_dict(merged_state_dict, strict=False)
    
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
    
    # Save merged model
    output_path = os.path.join(checkpoint, merge_suffix)
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done!")
    
    return output_path


if __name__ == "__main__":
    Fire(merge)
