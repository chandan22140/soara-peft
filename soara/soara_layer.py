"""
Unified SOARA Implementation
======================================

This module provides a complete implementation of SOARA (Subspace Orthogonal
Adaptation via Rotational Alignment).

Mathematical Framework:
1. Original weight W is decomposed: W = U @ S @ V^T
2. Split into principal and residual components based on rank r
3. W_residual = U[:,r:] @ S[r:,r:] @ V[:,r:]^T (NF4 quantized, frozen)
4. W_principal = U[:,:r] @ R_U @ S[:r,:r] @ R_V^T @ V[:,:r]^T (trainable via rotations)
5. Final computation: Y = X @ (W_residual + W_principal)

Trainable Components:
- R_U, R_V: Rotation matrices (4 different parameterization methods)
- S[:r,:r]: Principal singular values (optionally trainable)

Frozen Components:
- W_residual: NF4 quantized residual matrix
- U[:,:r], V[:,:r]: Principal singular vectors (optionally NF4 quantized)

Rotation Parameterization Methods:
- V1 (SOARA-V1): Direct optimization with orthogonality regularization
- V2 (SOARA-V2): Exact orthogonality via Givens rotations or Butterfly factorizations
- V3: Low-rank skew-symmetric perturbation I + BC^T - CB^T
- V4: Exponential map of skew-symmetric matrix exp(BC^T - CB^T)
"""

import os
import sys
import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Dict, Any, Literal, Union, List, Tuple
from dataclasses import dataclass, field

# DeepSpeed ZeRO-3 support for gathering sharded weights
try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    deepspeed = None


# ============================================================================
# MEMORY-EFFICIENT TENSOR UTILITIES
# ============================================================================

def to_free(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert tensor to different dtype/device AND free the original tensor's memory.
    
    Unlike tensor.to() which creates a new tensor but leaves the original in memory
    until garbage collection, this function explicitly frees the original tensor
    immediately after conversion.
    
    Args:
        tensor: The input tensor to convert (will be freed after conversion)
        dtype: Target dtype (e.g., torch.float16). If None, keeps original dtype.
        device: Target device (e.g., "cuda:0"). If None, keeps original device.
        
    Returns:
        New tensor with converted dtype/device. Original tensor memory is freed.
        
    Example:
        # Instead of:
        #   x = x.to(torch.float16)  # Original fp32 tensor still in memory!
        #   del old_x  # Would need to save reference first
        #   gc.collect()
        #   torch.cuda.empty_cache()
        
        # Use:
        x = to_free(x, dtype=torch.float16)  # Original freed automatically
        
    Warning:
        After calling to_free(tensor, ...), the original tensor reference is invalid.
        Only use the returned tensor.
    """
    # Build kwargs for .to() call
    to_kwargs = {}
    if dtype is not None:
        to_kwargs['dtype'] = dtype
    if device is not None:
        to_kwargs['device'] = device
    
    # If no conversion needed, return original tensor (no copy)
    if not to_kwargs:
        return tensor
    
    # Check if conversion is actually needed
    needs_dtype_change = dtype is not None and tensor.dtype != dtype
    needs_device_change = device is not None and tensor.device != device
    
    if not needs_dtype_change and not needs_device_change:
        return tensor  # No conversion needed, return original
    
    # Create new tensor with converted dtype/device
    new_tensor = tensor.to(**to_kwargs)
    
    # Explicitly free the original tensor's memory
    # We use tensor.data to access underlying storage and clear it
    # Then delete the tensor and force garbage collection
    del tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return new_tensor


def gather_zero3_weights(modules: List[nn.Module], device: torch.device) -> torch.Tensor:
    """
    Gather weights from modules that may be sharded with ZeRO-3.
    
    With ZeRO Stage 3, model weights are partitioned across GPUs. Accessing
    module.weight.data directly returns only the local shard (with shape [out, 0]
    on non-primary ranks). This function gathers the full weights.
    
    Args:
        modules: List of nn.Linear modules to gather weights from
        device: Target device for the stacked weights
        
    Returns:
        Stacked weight tensor of shape [num_modules, out_features, in_features]
    """
    weights = []
    
    if HAS_DEEPSPEED and deepspeed is not None:
        # Check if any module has ZeRO-3 partitioned params
        params_to_gather = [m.weight for m in modules]
        
        # Check if params are ZeRO-3 partitioned (have ds_status attribute)
        is_zero3 = any(
            hasattr(p, 'ds_status') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
            for p in params_to_gather
        )
        
        if is_zero3:
            # Gather all weights in a single context (more efficient than per-weight)
            with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=None):
                for m in modules:
                    # Clone the gathered weight to keep it after context exits
                    weights.append(m.weight.data.clone().to(device))
        else:
            # Params are not ZeRO-3 partitioned, access directly
            for m in modules:
                weights.append(m.weight.data.to(device))
    else:
        # No DeepSpeed, access weights directly
        for m in modules:
            weights.append(m.weight.data.to(device))
    
    return torch.stack(weights)

def to_free_inplace(
    obj: object,
    attr_name: str,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Convert an object's tensor attribute in-place and free the original memory.
    
    This is useful when you have a tensor stored as an attribute (e.g., self.weight)
    and want to convert it while freeing the original.
    
    Args:
        obj: Object containing the tensor attribute
        attr_name: Name of the attribute (e.g., "weight")
        dtype: Target dtype
        device: Target device
        
    Example:
        # Instead of:
        #   old = self.weight
        #   self.weight = self.weight.to(torch.float16)
        #   del old
        #   gc.collect()
        #   torch.cuda.empty_cache()
        
        # Use:
        to_free_inplace(self, "weight", dtype=torch.float16)
    """
    old_tensor = getattr(obj, attr_name)
    new_tensor = to_free(old_tensor, dtype=dtype, device=device)
    setattr(obj, attr_name, new_tensor)


# ============================================================================
# CORE CONFIGURATION
# ============================================================================
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    Linear4bit = None

@dataclass
class SOARAConfig:
    """Configuration for SOARA adapter."""
    
    # Core parameters
    r: Optional[int] = 16                         # Rank. If None, use per-layer max rank from SVD.
    lora_alpha: float = 16.0                     # Scaling factor
    lora_dropout: float = 0.0                    # Should be 0 for SOARA
    
    # Rotation parameterization method
    method: Literal["v1", "V2", "v3", "V4"] = "v1"
    
    # V1 (SOARA-V1) specific parameters (Direct orthogonality regularization)
    orthogonality_reg_weight: float = 1e-4      # Weight for orthogonality regularization loss (frobenius only)
    regularization_type: str = "frobenius"    # frobenius (recommended - fast), determinant, log_determinant
    
    # V2 specific parameters (Sequential Givens rotations OR Butterfly factorization)
    n_givens_layers: Optional[int] = None        # Number of Givens layers (default: r-1)
    steps_per_phase: int = 100                   # Steps to train each Givens layer
    total_cycles: int = 3                        # Total cycles through all layers
    use_butterfly: bool = False                  # If True, use log(r) butterfly layers instead of (r-1) Givens
    butterfly_sequential: bool = False           # If True, train butterfly components one at a time (like Givens)
    butterfly_block_size: int = 1                # Block size b for BOFT(m, b). b=2 gives O(r log r) params
    
    # V3/V4 specific parameters (Low-rank methods)
    low_rank_r: int = 4                          # Low rank for B,C matrices in ways V3&V4
    
    # General parameters
    init_identity: bool = True                # Initialize rotation matrices as identity
    freeze_singular_values: bool = False         # Whether to freeze S matrix
    s_dtype_fp32: bool = True                    # Store S in FP32 for precision (bfloat16 rounds small updates to 0)
    
    # Ablation: Rotation sidedness control (V1 only)
    # "both" = default (R_u and R_v both trainable)
    # "u_only" = only R_u trainable, R_v frozen as identity
    # "v_only" = only R_v trainable, R_u frozen as identity
    rotation_side: str = "both"
    
    # Quantization parameters
    quantize_residual: bool = False               # Whether to NF4 quantize W_residual
    quantize_base_components: bool = False       # Whether to NF4 quantize U and V^T

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

class ResidualLinear4bit(nn.Module):
    """
    4-bit quantized linear layer for W_residual following QLoRA pattern.
    
    This properly handles NF4 quantization by:
    1. Storing weight as Params4bit (NF4 quantized, ~4x memory reduction)
    2. Using dequantize_4bit for forward pass (reconstructs fp16 weight on-the-fly)
    3. Keeping bias in fp16/fp32 (not quantized, negligible memory)
    
    Memory efficiency:
    - Weight is stored in packed NF4 format (~4 bits per element)
    - quant_state stores quantization metadata (scales, zero points)
    - Forward pass dequantizes to fp16, computes matmul, then discards dequantized weight
    - No intermediate fp16 weight is stored, only the quantized representation
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        weight_data: The weight tensor to quantize (required, not optional)
        bias_data: Optional bias tensor (will be stored in fp16, not quantized)
        compute_dtype: Dtype for forward pass computation (default: float16)
        quant_type: Quantization type, "nf4" for normalized float 4-bit (default)
        compress_statistics: Whether to compress quantization statistics (default: True)
        device: Target device for the layer
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_data: torch.Tensor,
        bias_data: Optional[torch.Tensor] = None,
        compute_dtype: torch.dtype = torch.float16,
        quant_type: str = "nf4",
        compress_statistics: bool = True,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        
        # ====== WEIGHT QUANTIZATION ======
        # Directly create Params4bit from weight_data, no intermediate nn.Parameter allocation
        # This saves memory by avoiding: empty tensor -> copy weight -> quantize -> delete old
        if HAS_BITSANDBYTES:
            try:
                from bitsandbytes.nn import Params4bit
                
                # Params4bit expects fp16 input, will internally quantize to NF4
                # The resulting tensor is ~4x smaller than fp16
                # quant_state stores: absmax (scales), code (NF4 lookup table), blocksize, dtype
                self.weight = Params4bit(
                    weight_data.to(dtype=torch.float16, device=device),
                    requires_grad=False,  # Frozen, not trainable
                    quant_type=quant_type,  # "nf4" = normalized float 4-bit
                    compress_statistics=compress_statistics,  # Compress absmax for more savings
                )
                # Immediately free the original weight_data to reclaim memory
                del weight_data
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ Warning: NF4 quantization failed, falling back to fp16: {e}")
                # Fallback: store as regular fp16 parameter (no quantization)
                self.weight = nn.Parameter(
                    weight_data.to(dtype=torch.float16, device=device),
                    requires_grad=False
                )
        else:
            # bitsandbytes not available, use fp16
            self.weight = nn.Parameter(
                weight_data.to(dtype=torch.float16, device=device),
                requires_grad=False
            )
        
        # ====== BIAS (not quantized) ======
        # Bias is small (out_features,) so quantization overhead not worth it
        if bias_data is not None:
            self.bias = nn.Parameter(
                bias_data.to(dtype=torch.float16, device=device),
                requires_grad=False  # Frozen along with weight
            )
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ResidualLinear4bit.
        Handles both quantized (NF4/Params4bit) and non-quantized weights.

        Quantization logic:
        - If weight is quantized (Params4bit with quant_state), use bitsandbytes dequantization to recover the full-precision weight for matmul.
        - This avoids shape errors from using F.linear directly on packed/flattened quantized weights.
        - If dequantization fails (rare), fallback to creating a temporary Linear4bit layer and use its forward method.
        - If weight is not quantized, use standard F.linear.
        """
        inp_dtype = x.dtype

        # Ensure input is in compute_dtype (usually float16/bfloat16)
        if x.dtype != self.compute_dtype:
            x = x.to(self.compute_dtype)

        # --- Quantized weight path ---
        # Params4bit is a bitsandbytes class for 4-bit quantized weights
        # quant_state stores quantization metadata (scales, zero points, etc)
        if HAS_BITSANDBYTES and hasattr(self.weight, 'quant_state') and self.weight.quant_state is not None:
            try:
                # Dequantize the packed 4-bit weight to full precision
                # This reconstructs the original 2D weight matrix for matmul
                from bitsandbytes.functional import dequantize_4bit
                weight_deq = dequantize_4bit(
                    self.weight.data, 
                    self.weight.quant_state
                ).to(self.compute_dtype)

                # Bias must also match compute_dtype
                bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
                # Use standard F.linear with dequantized weight
                out = F.linear(x, weight_deq, bias)

            except Exception as e:
                print(f"⚠️ Warning: dequantize_4bit failed ({e}), trying Linear4bit wrapper...")
                # Fallback: Create a temporary Linear4bit layer
                # Linear4bit is a bitsandbytes nn.Module that handles quantized weights internally
                try:
                    linear_4bit = Linear4bit(
                        self.in_features, 
                        self.out_features,
                        bias=self.bias is not None,
                        compute_dtype=self.compute_dtype,
                        quant_type="nf4",
                    )
                    # Assign quantized weight and bias
                    linear_4bit.weight = self.weight
                    if self.bias is not None:
                        linear_4bit.bias = self.bias
                    linear_4bit = linear_4bit.to(x.device)
                    # Use Linear4bit's forward method
                    out = linear_4bit(x)
                except Exception as e2:
                    raise RuntimeError(f"All 4-bit forward methods failed: dequantize={e}, Linear4bit={e2}")
        else:
            # --- Non-quantized weight path ---
            # Use standard F.linear for fp16/fp32 weights
            out = F.linear(x, self.weight, self.bias)

        # Return output in original input dtype for consistency
        return out.to(inp_dtype)


def dequantize_params4bit(params: torch.Tensor) -> torch.Tensor:
    """
    Dequantize Params4bit tensor to fp16/fp32 for use in custom operations.
    
    Args:
        params: Params4bit tensor with quant_state
        
    Returns:
        Dequantized fp16 tensor
    """
    if not HAS_BITSANDBYTES or not hasattr(params, 'quant_state'):
        return params
    
    try:
        from bitsandbytes.functional import dequantize_4bit
        return dequantize_4bit(params, params.quant_state)
    except Exception as e:
        print(f"Warning: Dequantization failed: {e}")
        return params

def generate_givens_pairings(r: int, n_layers: int) -> List[List[Tuple[int, int]]]:
    """
    Generate disjoint pairings for Givens rotations using round-robin tournament.
    
    Args:
        r: Dimension of rotation space
        n_layers: Number of rotation layers to generate
        
    Returns:
        List of rotation layers, each containing disjoint pairs
    """
    if r <= 1:
        return []
        
    nodes = list(range(r))
    
    # Handle odd dimensions by adding dummy node
    if r % 2 == 1:
        nodes.append(-1)  # Dummy node
    
    fixed_node = nodes[0]
    rotating_nodes = nodes[1:]
    
    all_pairings = []
    for phase in range(min(n_layers, len(nodes) - 1)):
        current_pairs = []
        
        # Pair fixed node with first rotating node
        if rotating_nodes[0] != -1:  # Skip dummy
            current_pairs.append((fixed_node, rotating_nodes[0]))
        
        # Pair remaining nodes horizontally
        # After pairing position 0, we have positions 1 to n-1 left (even count)
        # We need to pair them: (1, n-1), (2, n-2), ..., up to the middle
        for i in range(1, (len(rotating_nodes) + 1) // 2):
            if rotating_nodes[i] != -1 and rotating_nodes[-i] != -1:
                current_pairs.append((rotating_nodes[i], rotating_nodes[-i]))
        
        all_pairings.append(current_pairs)
        
        # Rotate for next phase
        rotating_nodes = [rotating_nodes[-1]] + rotating_nodes[:-1]
    
    return all_pairings

# ============================================================================
# ROTATION LAYER IMPLEMENTATIONS
# ============================================================================

class GivensRotationLayer(nn.Module):
    """A layer representing parallel Givens rotations on disjoint axis pairs."""
    
    def __init__(self, dimension: int, pairs: List[Tuple[int, int]]):
        super().__init__()
        self.dimension = dimension
        self.pairs = pairs
        
        # Pre-compute indices for vectorized scatter/gather
        # Extract p and q indices from pairs
        p_indices = [p for p, q in pairs if p < dimension and q < dimension]
        q_indices = [q for p, q in pairs if p < dimension and q < dimension]
        
        # Register indices as buffers so they move with device but aren't params
        self.register_buffer('p_indices', torch.tensor(p_indices, dtype=torch.long))
        self.register_buffer('q_indices', torch.tensor(q_indices, dtype=torch.long))
        
        # Trainable rotation angles for each pair
        self.thetas = nn.Parameter(torch.zeros(len(self.p_indices)))
    
    def forward(self) -> torch.Tensor:
        """Construct the rotation matrix from Givens rotations (Vectorized)."""
        # Start with identity matrix
        R = torch.eye(self.dimension, device=self.thetas.device, dtype=self.thetas.dtype)
        
        # Compute sines and cosines
        cos_thetas = torch.cos(self.thetas)
        sin_thetas = torch.sin(self.thetas)
        
        # Vectorized update using advanced indexing
        # R[p, p] = cos(theta)
        # R[q, q] = cos(theta)
        # R[p, q] = -sin(theta)
        # R[q, p] = sin(theta)
        
        # Diagonal elements (updates are disjoint, so no conflict)
        R.diagonal().scatter_(0, self.p_indices, cos_thetas)
        R.diagonal().scatter_(0, self.q_indices, cos_thetas)
        
        # Off-diagonal elements
        R[self.p_indices, self.q_indices] = -sin_thetas
        R[self.q_indices, self.p_indices] = sin_thetas
        
        return R
    
    def reset(self):
        """Reset all rotation angles to zero (identity rotation)."""
        with torch.no_grad():
            self.thetas.zero_()


class ButterflyComponent(nn.Module):
    """
    Single butterfly component B_tilde_b(d, k) from BOFT paper.
    
    This implements a block-diagonal matrix where each block is a butterfly factor BF(k).
    For orthogonality, we parameterize each 2b×2b block as a product of Givens rotations.
    
    The butterfly factor structure (Equation 1 in paper):
        BF(k) = [[diag(d1), diag(d2)],
                 [diag(d3), diag(d4)]]
    
    For b=2, each 2×2 block is a Givens rotation parameterized by angle θ:
        [[cos(θ), -sin(θ)],
         [sin(θ),  cos(θ)]]
    
    Args:
        d: Full dimension of the orthogonal matrix R
        k: Level size (power of 2: 2, 4, 8, ..., d). Determines block arrangement.
        block_size: Size b of the base orthogonal blocks (default 2 for Givens rotations)
    """
    
    def __init__(self, d: int, k: int, block_size: int = 1):
        super().__init__()
        self.d = d
        self.k = k
        self.block_size = block_size
        
        # Number of butterfly factor blocks of size k along diagonal
        self.n_blocks = d // k
        
        # Each butterfly factor BF(k) has k/2 Givens rotations (for b=2)
        # arranged in a specific pattern. For b=2:
        # - We have k/2 rotation pairs in the top-left and bottom-right
        # - Plus k/2 rotation pairs connecting top-right and bottom-left
        # Total: k rotations per BF(k) block
        
        # For block_size=2, each BF(k) needs k/2 angle parameters
        # (each 2×2 Givens block has 1 angle)
        self.rotations_per_block = k // 2
        self.total_rotations = self.n_blocks * self.rotations_per_block
        
        # Trainable rotation angles - one per 2×2 Givens block
        self.thetas = nn.Parameter(torch.zeros(self.total_rotations))
        
        # Pre-compute indices for vectorized operations
        self._precompute_indices()
    
    def _precompute_indices(self):
        """Precompute index patterns for efficient forward pass."""
        # For each block, we need to apply Givens rotations in a butterfly pattern
        # The pattern depends on k and creates the cross-connections
        
        # For butterfly component at level k:
        # - Matrix is d×d with n_blocks = d/k butterfly factors on diagonal
        # - Each BF(k) is k×k and connects indices in a specific pattern
        
        # Build index arrays for source and destination of rotations
        p_indices = []  # First element of each rotation pair
        q_indices = []  # Second element of each rotation pair
        
        for block_idx in range(self.n_blocks):
            base_idx = block_idx * self.k
            half_k = self.k // 2
            
            # Butterfly pattern: pair i with i + k/2
            for i in range(half_k):
                p_indices.append(base_idx + i)
                q_indices.append(base_idx + i + half_k)
        
        self.register_buffer('p_indices', torch.tensor(p_indices, dtype=torch.long))
        self.register_buffer('q_indices', torch.tensor(q_indices, dtype=torch.long))
    
    def forward(self) -> torch.Tensor:
        """Construct the butterfly component matrix (for debugging/visualization).
        
        NOTE: For efficiency during training, use apply() instead which avoids
        building the full O(d²) matrix.
        """
        # Start with identity
        R = torch.eye(self.d, device=self.thetas.device, dtype=self.thetas.dtype)
        
        # Compute sin and cos for all rotations
        cos_thetas = torch.cos(self.thetas)
        sin_thetas = torch.sin(self.thetas)
        
        # Apply all Givens rotations in the butterfly pattern
        # R[p, p] = cos(θ), R[q, q] = cos(θ)
        # R[p, q] = -sin(θ), R[q, p] = sin(θ)
        R.diagonal().scatter_(0, self.p_indices, cos_thetas)
        R.diagonal().scatter_(0, self.q_indices, cos_thetas)
        R[self.p_indices, self.q_indices] = -sin_thetas
        R[self.q_indices, self.p_indices] = sin_thetas
        
        return R
    
    def apply_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply butterfly rotation directly to input (matrix-free, O(d) per sample).
        
        Computes x @ R where R is the butterfly component matrix.
        
        Instead of computing x @ R with a full d×d matrix, we apply the sparse
        Givens rotations directly to x. Each Givens rotation only affects 2 elements.
        
        Args:
            x: Input tensor of shape [..., d] where last dim is the rotation dimension
               
        Returns:
            Rotated tensor of same shape (x @ R)
            
        Complexity: O(d) instead of O(d²) for matrix multiplication
        """
        # Compute sin and cos for all rotations
        cos_thetas = torch.cos(self.thetas)
        sin_thetas = torch.sin(self.thetas)
        
        # Clone x to avoid in-place modification issues with autograd
        y = x.clone()
        
        # Extract values at p and q indices
        # For x of shape [..., d], we index the last dimension
        x_p = x[..., self.p_indices]  # [..., num_rotations]
        x_q = x[..., self.q_indices]  # [..., num_rotations]
        
        # Apply Givens rotations: x @ [cos -sin; sin cos]
        # For row vector x @ R:
        # y_p = cos * x_p + sin * x_q  (row p of R.T = [cos, sin])
        # y_q = -sin * x_p + cos * x_q (row q of R.T = [-sin, cos])
        y[..., self.p_indices] = cos_thetas * x_p + sin_thetas * x_q
        y[..., self.q_indices] = -sin_thetas * x_p + cos_thetas * x_q
        
        return y
    
    def apply_transpose(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transpose of butterfly rotation: x @ R.T (matrix-free, O(d) per sample).
        
        For Givens rotation R = [cos, -sin; sin, cos]:
        R.T = [cos, sin; -sin, cos]
        
        This is what the forward pass needs: x @ R_V.T and x @ R_U.T
        
        Args:
            x: Input tensor of shape [..., d] where last dim is the rotation dimension
               
        Returns:
            Rotated tensor of same shape (x @ R.T)
            
        Complexity: O(d) instead of O(d²) for matrix multiplication
        """
        # Compute sin and cos for all rotations
        cos_thetas = torch.cos(self.thetas)
        sin_thetas = torch.sin(self.thetas)
        
        # Clone x to avoid in-place modification issues with autograd
        y = x.clone()
        
        # Extract values at p and q indices
        x_p = x[..., self.p_indices]
        x_q = x[..., self.q_indices]
        
        # Apply transpose of Givens rotations: x @ R.T = x @ [cos, sin; -sin, cos]
        # For row vector x @ R.T:
        # y_p = cos * x_p - sin * x_q  (column p of R = [cos, -sin].T)
        # y_q = sin * x_p + cos * x_q  (column q of R = [sin, cos].T)
        y[..., self.p_indices] = cos_thetas * x_p - sin_thetas * x_q
        y[..., self.q_indices] = sin_thetas * x_p + cos_thetas * x_q
        
        return y
    
    def reset(self):
        """Reset all rotation angles to zero (identity)."""
        with torch.no_grad():
            self.thetas.zero_()


class ButterflyRotationLayer(nn.Module):
    """
    Full butterfly rotation matrix R(m, b) from BOFT paper.
    
    Composes m = log2(d) butterfly components to form a dense orthogonal matrix:
        R = B_tilde(d, d) @ B_tilde(d, d/2) @ ... @ B_tilde(d, 2)
    
    This achieves O(d log d) parameters while producing a dense orthogonal matrix,
    compared to O(d²) for a general orthogonal matrix.
    
    Key insight from paper: The butterfly structure is based on FFT's Cooley-Tukey
    algorithm, where local changes propagate globally through log(d) levels.
    
    Args:
        d: Dimension of the orthogonal matrix (should be power of 2)
        block_size: Size b of base orthogonal blocks (default 2)
    
    Example:
        For d=8, block_size=2:
        - m = log2(8) = 3 butterfly components
        - Levels k = 8, 4, 2
        - Total params = d * log2(d) / 2 = 8 * 3 / 2 = 12 angles per R_U/R_V
    """
    
    def __init__(self, d: int, block_size: int = 1):
        super().__init__()
        self.d = d
        self.block_size = block_size
        
        import math  # Import at top of method for use throughout
        
        # Validate d is a power of 2
        if d < 2 or (d & (d - 1)) != 0:
            # If not power of 2, round up to nearest power of 2
            d_padded = 2 ** math.ceil(math.log2(d))
            print(f"Warning: d={d} is not a power of 2. Using d_padded={d_padded}")
            self.d_padded = d_padded
            self.needs_padding = True
        else:
            self.d_padded = d
            self.needs_padding = False
        
        # Number of butterfly component levels
        self.m = int(math.log2(self.d_padded))
        
        # Create butterfly components for levels k = d, d/2, d/4, ..., 2
        # Following paper: R = B(d,d) @ B(d,d/2) @ ... @ B(d,2)
        self.components = nn.ModuleList()
        k = self.d_padded
        while k >= 2:
            self.components.append(ButterflyComponent(self.d_padded, k, block_size))
            k = k // 2
    
    def forward(self) -> torch.Tensor:
        """Compute the full butterfly rotation matrix R.
        
        Builds R = B(d,d) @ B(d,d/2) @ ... @ B(d,2) by composing all components.
        Uses optimized cuBLAS matrix multiplication which is faster than the
        theoretically-optimal O(d log d) apply() due to GPU parallelism.
        """
        # Start with identity
        R = torch.eye(self.d_padded, device=self.components[0].thetas.device, 
                      dtype=self.components[0].thetas.dtype)
        
        # Multiply all butterfly components: R = B(d,d) @ B(d,d/2) @ ... @ B(d,2)
        for component in self.components:
            R = R @ component()
        
        # If we padded, extract the d×d submatrix
        if self.needs_padding:
            R = R[:self.d, :self.d]
        
        return R
    
    def apply_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply butterfly rotation R to input x efficiently (matrix-free).
        
        Following BOFT paper insight: instead of computing x @ R where R is d×d,
        apply each butterfly component sequentially. This reduces complexity from
        O(d²) to O(d log d) per sample.
        
        Args:
            x: Input tensor of shape [..., d] where last dim is the rotation dimension
               
        Returns:
            Rotated tensor x @ R of same shape
            
        Complexity: O(d log d) instead of O(d²)
        """
        # If input dimension doesn't match d_padded, we need to handle padding
        if self.needs_padding:
            # Pad input to d_padded
            pad_size = self.d_padded - self.d
            x_padded = torch.nn.functional.pad(x, (0, pad_size), value=0)
        else:
            x_padded = x
        
        # Apply butterfly components sequentially: x @ B1 @ B2 @ ... @ Bm
        # Since apply does x @ B_i, we chain them
        result = x_padded
        for component in self.components:
            result = component.apply_rotation(result)
        
        # If we padded, extract the original d dimensions
        if self.needs_padding:
            result = result[..., :self.d]
        
        return result
    
    def apply_transpose(self, x: torch.Tensor) -> torch.Tensor:
        """Apply R.T to input x efficiently (matrix-free).
        
        Computes x @ R.T where R = B1 @ B2 @ ... @ Bm.
        Since (B1 @ B2 @ ... @ Bm).T = Bm.T @ ... @ B2.T @ B1.T,
        we apply component transposes in REVERSE order.
        
        This is what the forward pass needs: x @ R_V.T and x @ R_U.T
        
        Args:
            x: Input tensor of shape [..., d] where last dim is the rotation dimension
               
        Returns:
            Rotated tensor x @ R.T of same shape
            
        Complexity: O(d log d) instead of O(d²)
        """
        # If input dimension doesn't match d_padded, we need to handle padding
        if self.needs_padding:
            pad_size = self.d_padded - self.d
            x_padded = torch.nn.functional.pad(x, (0, pad_size), value=0)
        else:
            x_padded = x
        
        # Apply butterfly component transposes in REVERSE order
        # (B1 @ B2 @ ... @ Bm).T = Bm.T @ ... @ B2.T @ B1.T
        result = x_padded
        for component in reversed(self.components):
            result = component.apply_transpose(result)
        
        # If we padded, extract the original d dimensions
        if self.needs_padding:
            result = result[..., :self.d]
        
        return result
    
    def reset(self):
        """Reset all butterfly components to identity."""
        for component in self.components:
            component.reset()
    
    def get_num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# MAIN SOARA LAYER
# ============================================================================

class SOARALinearLayer(nn.Module):
    """
    A Linear layer with SOARA adaptation.
    
    Implementation follows SOARA paper methodology:
    1. SVD decomposition: W = U @ S @ V^T
    2. Split into principal and residual components
    3. W_residual = U[:,r:] @ S[r:,r:] @ V[:,r:]^T (NF4 quantized and frozen)
    4. W_principal = U[:,:r] @ R_U @ S[:r,:r] @ R_V^T @ V[:,:r]^T (trainable via rotations)
    5. Final output: Y = X * (W_residual + W_principal)
    
    Where:
    - U[:,:r], V[:,:r]^T are frozen (optionally NF4 quantized)
    - R_U @ S[:r,:r] @ R_V^T is the trainable rotation part
    - R_U, R_V are learnable rotation matrices
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        soara_config: SOARAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.soara_config = soara_config
        self.adapter_name = adapter_name
        
        # Store base layer properties
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Cap r to the layer's maximum possible rank
        # For a matrix of shape (out_features, in_features), max rank is min(out, in)
        max_rank = min(self.in_features, self.out_features)
        if soara_config.r is None:
            self.r = max_rank
        else:
            self.r = min(soara_config.r, max_rank)
        
        # Perform SVD decomposition on base weight
        self._initialize_svd_components()
        
        # Initialize rotation matrices based on selected method
        self._initialize_rotation_matrices()
        
        # Dropout layer
        if soara_config.lora_dropout > 0:
            self.dropout = nn.Dropout(soara_config.lora_dropout)
        else:
            self.dropout = nn.Identity()
            
        # Scaling factor
        self.scaling = 1
        # self.scaling = soara_config.lora_alpha / soara_config.r
        
        # Training state for V2 (sequential Givens) - not needed for butterfly mode
        if soara_config.method == "V2" and not soara_config.use_butterfly:
            self.current_layer_index = 0
            self.current_cycle = 0
            self.step_count = 0
    
    def _initialize_svd_components(self):
        """Perform SVD on base layer weight and implement SOARA methodology.
        
        Following SOARA paper:
        1. Decompose W = U @ S @ V^T
        2. Split into principal {U[:,:r], S[:r,:r], V[:,:r]} and residual {U[:,r:], S[r:,r:], V[:,r:]}
        3. Create W_residual = U[:,r:] @ S[r:,r:] @ V[:,r:]^T (NF4 quantized, frozen)
        4. Store principal components for rotational adapter: U[:,:r], S[:r,:r], V[:,:r]
        5. Adapter computes W_principal = U[:,:r] @ R_U @ S[:r,:r] @ R_V^T @ V[:,:r]^T
        """
        import gc
        
        with torch.no_grad():
            original_weight = self.base_layer.weight.data.clone()
            
            # SVD doesn't support BF16/FP16 on CUDA - convert to FP32 for SVD
            original_dtype = original_weight.dtype
            original_device = original_weight.device  # Preserve device (GPU)
            if original_weight.dtype in (torch.bfloat16, torch.float16):
                original_weight = original_weight.float()
            
            # Perform SVD: W = U @ S @ V^T
            U, S, V = torch.linalg.svd(original_weight, full_matrices=False)
            
            # Immediately delete original_weight copy to free VRAM
            del original_weight
            gc.collect()
            torch.cuda.empty_cache()
            
            # Split into principal (top-r) and residual components
            U_principal = U[:, :self.r].contiguous()  # [out_features, r]
            U_residual = U[:, self.r:].contiguous()   # [out_features, remaining]
            
            S_principal = S[:self.r].contiguous()     # [r]
            S_residual = S[self.r:].contiguous()      # [remaining]
            
            V_principal = V[:self.r, :].contiguous()  # [r, in_features]
            V_residual = V[self.r:, :].contiguous()   # [remaining, in_features]
            
            # Delete full U, S, V to free VRAM
            del U, S, V
            gc.collect()
            torch.cuda.empty_cache()
            
            # === RESIDUAL COMPONENTS (FROZEN, OPTIONALLY QUANTIZED) ===
            # Compute residual matrix: W_residual = U[:,r:] @ S[r:,r:] @ V[:,r:]^T
            if U_residual.shape[1] > 0:  # Check if there are residual components
                S_residual_diag = torch.diag(S_residual)
                W_residual = U_residual @ S_residual_diag @ V_residual
                
                # Delete temporary residual components immediately
                del S_residual_diag
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # If r equals the rank of the matrix, residual is zero
                W_residual = torch.zeros_like(self.base_layer.weight.data)
            
            # Delete U_residual, S_residual, V_residual to free VRAM
            del U_residual, S_residual, V_residual
            gc.collect()
            torch.cuda.empty_cache()
            
            # ========== Quantize W_residual (QLoRA pattern) ==========
            if self.soara_config.quantize_residual and HAS_BITSANDBYTES:
                print(f"      Quantizing W_residual to NF4 (QLoRA pattern)")
                
                # Get bias data if exists (before we replace base_layer)
                bias_data = self.base_layer.bias.data.clone() if self.base_layer.bias is not None else None
                
                # Create ResidualLinear4bit with weight_data directly
                # This quantizes W_residual immediately in __init__, no intermediate allocation
                quantized_layer = ResidualLinear4bit(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    weight_data=W_residual,  # Passed directly, quantized in __init__
                    bias_data=bias_data,      # Passed directly, stored as fp16
                    compute_dtype=torch.float16,
                    device=original_device,
                )
                
                # Replace base layer with quantized version
                self.base_layer = quantized_layer
            else:
                # Standard fp16/fp32 approach - use to_free to convert and free W_residual
                self.base_layer.weight.data = to_free(W_residual, dtype=original_dtype, device=original_device)
                self.base_layer.weight.requires_grad = False
                # Note: W_residual already freed by to_free() above, no need for explicit del
            
            # Note: In quantized path, W_residual is passed to ResidualLinear4bit which frees it internally
            
            # ========== Store principal components U, V, S ==========
            # These are kept in fp16 by default (like LoRA A/B in QLoRA)
            # Optional: quantize U, V if requested (not recommended, but supported)
            
            if self.soara_config.quantize_base_components and HAS_BITSANDBYTES:
                print(f"      Quantizing U and V to NF4 (not recommended, may impact accuracy)")
                from bitsandbytes.nn import Params4bit
                
                try:
                    # Convert to fp16 and free original fp32 tensors
                    U_fp16 = to_free(U_principal, dtype=torch.float16, device=original_device)
                    V_fp16 = to_free(V_principal, dtype=torch.float16, device=original_device)
                    
                    # Quantize U and V (Params4bit takes ownership of the tensor)
                    U_quantized = Params4bit(
                        U_fp16,
                        requires_grad=False,
                        quant_type="nf4",
                        compress_statistics=True,
                    )
                    V_quantized = Params4bit(
                        V_fp16,
                        requires_grad=False,
                        quant_type="nf4",
                        compress_statistics=True,
                    )
                    
                    # Free the fp16 intermediates (Params4bit made a copy)
                    del U_fp16, V_fp16
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Register as buffers (non-trainable)
                    self.register_buffer("U", U_quantized)
                    self.register_buffer("V", V_quantized)
                    
                    print(f"      ✓ U, V quantized to NF4")
                    
                except Exception as e:
                    print(f"      Warning: U/V quantization failed, using fp16: {e}")
                    # Fallback to fp16 - use to_free to convert and free originals
                    self.register_buffer("U", to_free(U_principal, dtype=torch.float16, device=original_device))
                    self.register_buffer("V", to_free(V_principal, dtype=torch.float16, device=original_device))
            else:
                # Standard approach: keep U, V in same dtype as original model
                # Use to_free to convert dtype and free the original fp32 tensors
                self.register_buffer("U", to_free(U_principal, dtype=original_dtype, device=original_device))
                self.register_buffer("V", to_free(V_principal, dtype=original_dtype, device=original_device))
            
            # Note: U_principal and V_principal are already freed by to_free() above
            # No need for explicit del + gc.collect + empty_cache here
            
            # S: Use FP32 if s_dtype_fp32 is enabled (fixes bfloat16 precision issues)
            # With bfloat16, small optimizer updates (lr*grad ~ 2e-6) round to 0 for values ~6.0
            s_dtype = torch.float32 if self.soara_config.s_dtype_fp32 else original_dtype
            S_converted = to_free(S_principal, dtype=s_dtype, device=original_device)
            if self.soara_config.freeze_singular_values:
                self.register_buffer("S", S_converted)
            else:
                self.S = nn.Parameter(S_converted)
            
            # Note: S_principal already freed by to_free() above
            
            # print(f"Layer {self.adapter_name}: S stats - Min: {self.S.min().item():.4f}, Max: {self.S.max().item():.4f}, Mean: {self.S.mean().item():.4f}, dtype: {self.S.dtype}")
    
    def _initialize_rotation_matrices(self):
        """Initialize rotation matrices R_U and R_V based on the selected method."""
        if self.soara_config.method == "v1":
            self._init_v1()
        elif self.soara_config.method == "V2":
            self._init_V2()
        elif self.soara_config.method == "v3":
            self._init_v3()
        elif self.soara_config.method == "V4":
            self._init_V4()
        else:
            raise ValueError(f"Unknown rotation method: {self.soara_config.method}")
    
    def _init_v1(self):
        """V1 (SOARA-V1): Direct parameterization with orthogonality regularization.
        
        Supports rotation_side ablation:
        - "both": R_U and R_V are both trainable nn.Parameters
        - "u_only": R_U is trainable, R_V is a frozen identity buffer
        - "v_only": R_V is trainable, R_U is a frozen identity buffer
        """
        # Get device and dtype from U buffer (which is already on correct device)
        device = self.U.device
        dtype = self.U.dtype
        
        rotation_side = self.soara_config.rotation_side
        
        if self.soara_config.init_identity:
            R_U_init = torch.eye(self.r, dtype=dtype, device=device)
            R_V_init = torch.eye(self.r, dtype=dtype, device=device)
        else:
            # Initialize as random orthogonal matrices via QR
            R_U_init = torch.randn(self.r, self.r, dtype=dtype, device=device)
            R_V_init = torch.randn(self.r, self.r, dtype=dtype, device=device)
            R_U_init, _ = torch.linalg.qr(R_U_init)
            R_V_init, _ = torch.linalg.qr(R_V_init)
        
        # Apply rotation_side control: trainable parameter vs frozen buffer
        if rotation_side in ("both", "u_only"):
            self.R_U = nn.Parameter(R_U_init)
        else:  # v_only
            self.register_buffer("R_U", torch.eye(self.r, dtype=dtype, device=device))
        
        if rotation_side in ("both", "v_only"):
            self.R_V = nn.Parameter(R_V_init)
        else:  # u_only
            self.register_buffer("R_V", torch.eye(self.r, dtype=dtype, device=device))
    
    def _init_V2(self):
        """V2 (SOARA-V2): Greedy sequential Givens rotations OR Butterfly factorization.
        
        If use_butterfly=True:
            Uses ButterflyRotationLayer with log(r) levels for O(r log r) parameters.
            All components trained simultaneously (no sequential phases).
        
        If use_butterfly=False (default):
            Uses sequential Givens rotations with dynamic layer instantiation.
            To save VRAM, we only instantiate the current active Givens layer.
            After training each layer, we:
            1. Merge its rotation into U and V (in-place)
            2. Delete the layer to free VRAM
            3. Instantiate the next layer
        """
        device = self.U.device
        
        if self.soara_config.use_butterfly:
            block_size = self.soara_config.butterfly_block_size
            
            if self.soara_config.butterfly_sequential:
                # Sequential butterfly mode: train one component at a time (like Givens)
                # Components are B(d,d), B(d,d/2), ..., B(d,2) - m = log2(d) total
                d_padded = 2 ** math.ceil(math.log2(self.r)) if (self.r & (self.r - 1)) != 0 else self.r
                
                # Generate k values for all components: [d, d/2, d/4, ..., 2]
                self.butterfly_k_values = []
                k = d_padded
                while k >= 2:
                    self.butterfly_k_values.append(k)
                    k = k // 2
                
                # Track padding for non-power-of-2 ranks
                self.butterfly_d_padded = d_padded
                self.butterfly_needs_padding = d_padded != self.r
                
                # Track current component and cycle
                self.current_butterfly_idx = 0
                self.current_butterfly_cycle = 0
                
                # Create first component
                first_k = self.butterfly_k_values[0]
                current_u = ButterflyComponent(d_padded, first_k, block_size).to(device)
                current_v = ButterflyComponent(d_padded, first_k, block_size).to(device)
                
                self.add_module('current_butterfly_u', current_u)
                self.add_module('current_butterfly_v', current_v)
                
                n_components = len(self.butterfly_k_values)
                n_params = current_u.thetas.numel()
                print(f"    🦋 Sequential Butterfly: {n_components} components, starting with B(d,{first_k}), {n_params} params/component")
            else:
                # Standard butterfly mode: all components trained simultaneously
                self.butterfly_u = ButterflyRotationLayer(self.r, block_size).to(device)
                self.butterfly_v = ButterflyRotationLayer(self.r, block_size).to(device)
                
                n_params_per_layer = self.butterfly_u.get_num_parameters()
                n_levels = self.butterfly_u.m
                print(f"    🦋 Butterfly mode: {n_levels} levels (log2({self.r})), {n_params_per_layer} params per rotation matrix")
        else:
            # Classic Givens mode: sequential layer training
            n_layers = self.soara_config.n_givens_layers or (self.r - 1)
            
            # Generate pairings for all phases (we'll instantiate layers on-demand)
            self.givens_pairings = generate_givens_pairings(self.r, n_layers)
            
            # Instead of creating all layers, we'll create them dynamically
            # Store only the current active layer - must use add_module for proper registration
            current_u = GivensRotationLayer(self.r, self.givens_pairings[0]).to(device)
            current_v = GivensRotationLayer(self.r, self.givens_pairings[0]).to(device)
            
            # Register as submodules so PyTorch tracks them properly
            self.add_module('current_givens_u', current_u)
            self.add_module('current_givens_v', current_v)
            
            # Note: U and V buffers will be updated in-place as we merge rotations
            # No need for separate U_base/V_base since we modify U/V directly
    
    def _init_v3(self):
        """V3: Low-rank skew-symmetric perturbation I + BC^T - CB^T."""
        lr_r = self.soara_config.low_rank_r
        
        # Get device and dtype from U buffer
        device = self.U.device
        dtype = self.U.dtype
        
        # Low-rank matrices B, C for U and V rotations
        self.B_U = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
        self.B_V = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
        
        if self.soara_config.init_identity:
            # Initialize C to zero so that BC^T - CB^T = 0, resulting in Identity rotation
            self.C_U = nn.Parameter(torch.zeros(self.r, lr_r, dtype=dtype, device=device))
            self.C_V = nn.Parameter(torch.zeros(self.r, lr_r, dtype=dtype, device=device))
        else:
            self.C_U = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
            self.C_V = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
    
    def _init_V4(self):
        """V4: Exponential map of skew-symmetric matrix."""
        lr_r = self.soara_config.low_rank_r
        
        # Get device and dtype from U buffer
        device = self.U.device
        dtype = self.U.dtype
        
        # Low-rank matrices B, C for generating skew-symmetric matrices
        self.B_U = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
        self.B_V = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
        
        if self.soara_config.init_identity:
            # Initialize C to zero so that BC^T - CB^T = 0, resulting in Identity rotation (exp(0)=I)
            self.C_U = nn.Parameter(torch.zeros(self.r, lr_r, dtype=dtype, device=device))
            self.C_V = nn.Parameter(torch.zeros(self.r, lr_r, dtype=dtype, device=device))
        else:
            self.C_U = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
            self.C_V = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
    
    def get_rotation_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the current rotation matrices R_U and R_V."""
        if self.soara_config.method == "v1":
            return self.R_U, self.R_V
        
        elif self.soara_config.method == "V2":
            # For V2 (SOARA-V2), check if using butterfly or classic Givens
            if self.soara_config.use_butterfly:
                if self.soara_config.butterfly_sequential:
                    # Sequential butterfly: return current component's matrix
                    R_U = self.current_butterfly_u()
                    R_V = self.current_butterfly_v()
                    # Handle padding (extract r×r submatrix if needed)
                    if self.butterfly_needs_padding:
                        R_U = R_U[:self.r, :self.r]
                        R_V = R_V[:self.r, :self.r]
                    return R_U, R_V
                else:
                    return self.butterfly_u(), self.butterfly_v()
            else:
                return self.current_givens_u(), self.current_givens_v()
        
        elif self.soara_config.method == "v3":
            # R = I + BC^T - CB^T (skew-symmetric perturbation)
            R_U = (torch.eye(self.r, device=self.B_U.device, dtype=self.B_U.dtype) + 
                   self.B_U @ self.C_U.T - self.C_U @ self.B_U.T)
            R_V = (torch.eye(self.r, device=self.B_V.device, dtype=self.B_V.dtype) + 
                   self.B_V @ self.C_V.T - self.C_V @ self.B_V.T)
            return R_U, R_V
        
        elif self.soara_config.method == "V4":
            # R = exp(BC^T - CB^T)
            skew_U = self.B_U @ self.C_U.T - self.C_U @ self.B_U.T
            skew_V = self.B_V @ self.C_V.T - self.C_V @ self.B_V.T
            
            # Cast to float32 for matrix_exp stability/compatibility
            orig_dtype = skew_U.dtype
            R_U = torch.matrix_exp(skew_U.float()).to(dtype=orig_dtype)
            R_V = torch.matrix_exp(skew_V.float()).to(dtype=orig_dtype)
            return R_U, R_V
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the rotational adapter.
        
        Following SOARA formulation:
        Y = X * W = X * (W_residual + W_principal)
        where:
        - W_residual is stored in base_layer (NF4 quantized if quantize_residual=True)
        - W_principal = U @ R_U @ S @ R_V^T @ V^T (trainable via rotations)
        
        Handles quantized U/V by dequantizing before matmul operations.
        
        For butterfly mode (V2 + use_butterfly=True), uses O(d log d) apply()
        instead of building full O(d²) rotation matrices.
        """
        # Base layer forward pass (contains W_residual - optionally quantized)
        result = self.base_layer(x)

        # Get U, V (dequantize if needed)
        # If U/V are Params4bit, dequantize them for use in matmul
        U_current = dequantize_params4bit(self.U) if hasattr(self.U, 'quant_state') else self.U
        V_current = dequantize_params4bit(self.V) if hasattr(self.V, 'quant_state') else self.V
        
        # Ensure all components are in same dtype as input for matmul compatibility
        # Only convert if dtype differs to avoid unnecessary overhead
        target_dtype = x.dtype
        
        if U_current.dtype != target_dtype:
            U_current = U_current.to(target_dtype)
        if V_current.dtype != target_dtype:
            V_current = V_current.to(target_dtype)
        # S is stored in FP32 for optimizer precision, convert to compute dtype
        # Note: .to() is differentiable, gradients flow back to self.S
        if self.S.dtype != target_dtype:
            S_current = self.S.to(target_dtype)
        else:
            S_current = self.S
    
        # Forward through the adaptation path: x @ V^T @ R_V @ S @ R_U^T @ U^T
        x_adapted = self.dropout(x)
        
        # Apply the rotational transformation chain
        # Note: V is [r, in_features], so V^T is [in_features, r]
        x_adapted = x_adapted @ V_current.T  # [batch, in_features] @ [in_features, r] -> [batch, r]
        
        # Get rotation matrices (butterfly uses forward() which builds full matrix - 
        # this is actually faster than apply() due to optimized cuBLAS)
        R_U, R_V = self.get_rotation_matrices()
        
        if R_U.dtype != target_dtype:
            R_U = R_U.to(target_dtype)
        if R_V.dtype != target_dtype:
            R_V = R_V.to(target_dtype)
        
        x_adapted = x_adapted @ R_V.T        # [batch, r] @ [r, r] -> [batch, r]
        x_adapted = x_adapted * S_current    # [batch, r] * [r] -> [batch, r] (broadcasting)
        x_adapted = x_adapted @ R_U.T        # [batch, r] @ [r, r] -> [batch, r]
        
        x_adapted = x_adapted @ U_current.T  # [batch, r] @ [r, out_features] -> [batch, out_features]
        
        # Scale and add W_principal to W_residual result
        result = result + x_adapted * self.scaling
        
        return result

    def get_orthogonality_loss(self) -> torch.Tensor:
        """Compute orthogonality regularization loss for V1 (SOARA-V1).
        
        Supports three regularization types:
        1. frobenius: ||R^T @ R - I||_F^2 (encourages orthogonality)
        2. determinant: (det(R) - 1)^2 (encourages det = +1)
        3. log_determinant: (log(det(R)) - 0)^2 (numerically stable det constraint)
        
        Respects rotation_side: only computes loss for trainable rotation matrices.
        Frozen identity buffers always satisfy orthogonality, so their loss = 0.
        """
        # Early return if regularization is disabled to save compute
        if self.soara_config.orthogonality_reg_weight == 0:
            return torch.tensor(0.0, device=self.R_U.device if hasattr(self, 'R_U') else next(self.parameters()).device)
        
        if self.soara_config.method != "v1":
            # For ways 1, 2, 3: orthogonality is guaranteed by construction
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        
        reg_type = self.soara_config.regularization_type
        rotation_side = self.soara_config.rotation_side
        
        # Determine which rotation matrices need regularization
        # Frozen identity buffers are already orthogonal, skip them
        compute_u = rotation_side in ("both", "u_only") and isinstance(self.R_U, nn.Parameter)
        compute_v = rotation_side in ("both", "v_only") and isinstance(self.R_V, nn.Parameter)
        
        loss_u = torch.tensor(0.0, device=self.R_U.device)
        loss_v = torch.tensor(0.0, device=self.R_V.device)
        
        if reg_type == "frobenius":
            I = torch.eye(self.r, device=self.R_U.device, dtype=self.R_U.dtype)
            if compute_u:
                loss_u = torch.norm(self.R_U.T @ self.R_U - I, p='fro') ** 2
            if compute_v:
                loss_v = torch.norm(self.R_V.T @ self.R_V - I, p='fro') ** 2
            
        elif reg_type == "determinant":
            if compute_u:
                det_u = torch.det(self.R_U.float())
                loss_u = (det_u - 1.0) ** 2
            if compute_v:
                det_v = torch.det(self.R_V.float())
                loss_v = (det_v - 1.0) ** 2
            
        elif reg_type == "log_determinant":
            if compute_u:
                log_det_u = torch.logdet(self.R_U.float())
                loss_u = log_det_u ** 2
            if compute_v:
                log_det_v = torch.logdet(self.R_V.float())
                loss_v = log_det_v ** 2
        
        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")
        
        return (loss_u + loss_v) * self.soara_config.orthogonality_reg_weight
    
    def step_phase(self):
        """Advance to next phase for V2 (sequential Givens training).
        
        This method:
        1. Merges current Givens rotation into U and V (in-place)
        2. Deletes current Givens layers to free VRAM
        3. ALWAYS instantiates next Givens layer (even after all cycles complete)
           - This ensures forward pass never breaks
           - Layers after training completes are just frozen (won't be trained)
           
        Returns:
            tuple: (params_before, params_after) for tracking parameter count changes
        """
        if self.soara_config.method != "V2":
            return (0, 0)
        
        # Butterfly mode doesn't use sequential phases - all layers trained simultaneously
        if self.soara_config.use_butterfly and not self.soara_config.butterfly_sequential:
            return (0, 0)
        
        # Handle sequential butterfly transitions
        if self.soara_config.use_butterfly and self.soara_config.butterfly_sequential:
             # Count trainable parameters before merging
            params_before = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            with torch.no_grad():
                # Get current component matrices
                R_U = self.current_butterfly_u()
                R_V = self.current_butterfly_v()
                
                # Handle padding if needed (extract r×r submatrix)
                if self.butterfly_needs_padding:
                    R_U = R_U[:self.r, :self.r]
                    R_V = R_V[:self.r, :self.r]
                
                # Merge into U/V in-place: U_new = U @ R_U, V_new = R_V @ V
                self.U.copy_(self.U @ R_U)
                self.V.copy_(R_V @ self.V)
                
                # Delete current components to free VRAM
                delattr(self, 'current_butterfly_u')
                delattr(self, 'current_butterfly_v')

                # Force cleanup to ensure VRAM is available for next component
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Advance to next component
                self.current_butterfly_idx = (self.current_butterfly_idx + 1) % len(self.butterfly_k_values)
                
                # Check for cycle completion
                if self.current_butterfly_idx == 0:
                    self.current_butterfly_cycle += 1
                    # print(f"  ✓ Completed butterfly cycle {self.current_butterfly_cycle}/{self.soara_config.total_cycles}")
                
                # Create next component
                next_k = self.butterfly_k_values[self.current_butterfly_idx]
                block_size = self.soara_config.butterfly_block_size
                device = self.U.device
                
                next_u = ButterflyComponent(self.butterfly_d_padded, next_k, block_size).to(device)
                next_v = ButterflyComponent(self.butterfly_d_padded, next_k, block_size).to(device)
                
                # If cycles complete, just log it but don't freeze
                if self.current_butterfly_cycle >= self.soara_config.total_cycles:
                     # Start of extra cycles - keep training!
                     pass
                
                self.add_module('current_butterfly_u', next_u)
                self.add_module('current_butterfly_v', next_v)
            
            params_after = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return (params_before, params_after)

        # Standard Givens sequential training
        # Count trainable parameters before merging
        params_before = sum(p.numel() for p in self.parameters() if p.requires_grad)

        with torch.no_grad():
            # Get current rotation matrices before deletion
            R_U = self.current_givens_u()
            R_V = self.current_givens_v()
            
            # Merge rotations into U and V IN-PLACE
            # U_new = U_old @ R_U
            # V_new = R_V @ V_old
            self.U.copy_(self.U @ R_U)
            self.V.copy_(R_V @ self.V)
            
            # Delete current Givens layers to free VRAM
            delattr(self, 'current_givens_u')
            delattr(self, 'current_givens_v')

            # Force cleanup to ensure VRAM is available for next layer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Advance to next layer
            self.current_layer_index = (self.current_layer_index + 1) % len(self.givens_pairings)
            
            # Check if we completed a full cycle
            if self.current_layer_index == 0:
                self.current_cycle += 1
                # print(f"  ✓ Completed cycle {self.current_cycle}/{self.soara_config.total_cycles}")
            
            # ALWAYS instantiate next Givens layer (prevents AttributeError in forward/validation)
            # Even after all cycles complete, we need layers for forward pass
            next_pairs = self.givens_pairings[self.current_layer_index]
            next_u = GivensRotationLayer(self.r, next_pairs)
            next_v = GivensRotationLayer(self.r, next_pairs)
            
            # Move to same device as U
            device = self.U.device
            next_u = next_u.to(device)
            next_v = next_v.to(device)
            
            # If we've completed all cycles, just log it but don't freeze
            if self.current_cycle >= self.soara_config.total_cycles:
                 # Start of extra cycles - keep training!
                 pass
            
            # Register as submodules
            self.add_module('current_givens_u', next_u)
            self.add_module('current_givens_v', next_v)
            
        # Count trainable parameters after instantiating next layer
        params_after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return (params_before, params_after)
        
    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all trainable parameters in this adapter.
        
        Returns:
            Dictionary of parameter names to tensors
            
        Trainable components:
            - R_U, R_V: Rotation matrices (all 4 ways)
            - S: Singular values (if freeze_singular_values=False)
            - Givens angles: For V2 (SOARA-V2)
            - B_U, C_U, B_V, C_V: For Ways V3 & V4
            
        Frozen components:
            - base_layer.weight: W_residual (NF4 quantized)
            - U, V: Principal singular vectors (optionally quantized)
        """
        trainable = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable[name] = param
        
        return trainable
    
    def get_frozen_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all frozen parameters and buffers.
        
        Returns:
            Dictionary of frozen parameter/buffer names to tensors
        """
        frozen = {}
        
        # Add frozen parameters
        for name, param in self.named_parameters():
            if not param.requires_grad:
                frozen[name] = param
        
        # Add buffers (always frozen)
        for name, buffer in self.named_buffers():
            frozen[name] = buffer
            
        return frozen
    
    def freeze_base_and_unfreeze_adapters(self):
        """
        Freeze base layer (W_residual, bias) and buffers (U, V), 
        unfreeze rotation parameters and optionally S.
        
        This method ensures correct freezing regardless of the rotation method:
        - V1: R_U, R_V trainable
        - V2: givens_u_layers.*.thetas, givens_v_layers.*.thetas trainable
        - V3/V4: B_U, C_U, B_V, C_V trainable
        - S: trainable if freeze_singular_values=False
        
        Base layer weight/bias and U/V buffers are always frozen.
        """
        for name, param in self.named_parameters():
            # Freeze base_layer weight and bias (W_residual)
            if name.startswith('base_layer.'):
                param.requires_grad = False
            else:
                # Everything else (rotation params, S if not frozen) should be trainable
                param.requires_grad = True
        
        # Note: U and V are buffers (not parameters), so they're automatically frozen
        # S is either a buffer (frozen) or parameter (trainable) based on config
    
    def print_parameter_summary(self):
        """Print a summary of trainable vs frozen parameters."""
        trainable = self.get_trainable_parameters()
        frozen = self.get_frozen_parameters()
        
        print(f"\n📊 Parameter Summary for {self.__class__.__name__}")
        print("=" * 60)
        
        print(f"🔥 Trainable Parameters ({len(trainable)} items):")
        total_trainable = 0
        for name, param in trainable.items():
            size = param.numel()
            total_trainable += size
            print(f"  • {name}: {tuple(param.shape)} ({size:,} params)")
        
        print(f"\n🧊 Frozen Parameters/Buffers ({len(frozen)} items):")
        total_frozen = 0
        for name, param in frozen.items():
            size = param.numel() if hasattr(param, 'numel') else len(param)
            total_frozen += size
            dtype = getattr(param, 'dtype', 'unknown')
            print(f"  • {name}: {tuple(param.shape)} ({size:,} params, {dtype})")
        
        print(f"\n📈 Summary:")
        print(f"  • Total trainable: {total_trainable:,}")
        print(f"  • Total frozen: {total_frozen:,}")
        print(f"  • Efficiency ratio: {total_trainable/(total_trainable+total_frozen):.4f}")
        print(f"  • Quantization: W_residual={'✅' if self.soara_config.quantize_residual else '❌'}, U/V={'✅' if self.soara_config.quantize_base_components else '❌'}")

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class SOARATrainer:
    """Helper class for training SOARA models with V2 (sequential Givens)."""
    
    def __init__(self, model: nn.Module, config: SOARAConfig):
        self.model = model
        self.config = config
        
        # Find all rotational adapters
        self.adapters = []
        for module in model.modules():
            if isinstance(module, SOARALinearLayer):
                self.adapters.append(module)
    
    def should_step_phase(self, global_step: int) -> bool:
        """Check if we should advance to the next phase."""
        if self.config.method != "V2":
            return False
        
        return global_step % self.config.steps_per_phase == 0 and global_step > 0
    
    def step_phase(self):
        """Advance all adapters to next phase.
        
        Returns:
            tuple: (total_params_before, total_params_after) aggregated across all adapters
        """
        total_before = 0
        total_after = 0
        
        for adapter in self.adapters:
            before, after = adapter.step_phase()
            total_before += before
            total_after += after
        
        return (total_before, total_after)
    
    def get_orthogonality_loss(self) -> torch.Tensor:
        """Get total orthogonality regularization loss."""
        total_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for adapter in self.adapters:
            total_loss = total_loss + adapter.get_orthogonality_loss()
        
        return total_loss
    
    # def is_training_complete(self) -> bool:
    #     """Check if training is complete for V2 (SOARA-V2)."""
    #     if self.config.method != "V2" or not self.adapters:
    #         return False
        
    #     return self.adapters[0].current_cycle >= self.config.total_cycles

# ============================================================================
# BATCHED SVD HELPER FOR FAST INITIALIZATION
# ============================================================================

def _create_adapter_from_svd(
    base_layer: nn.Linear,
    soara_config: 'SOARAConfig',
    adapter_name: str,
    U: torch.Tensor,  # Full U from SVD [out_features, k]
    S: torch.Tensor,  # Full S from SVD [k]
    V: torch.Tensor,  # Full V from SVD [k, in_features]
    original_dtype: torch.dtype,
    original_device: torch.device
) -> 'SOARALinearLayer':
    """
    Create a SOARALinearLayer from pre-computed SVD results.
    
    This is used by batched SVD processing to avoid redundant per-layer SVD computation.
    The SVD is computed once in a batch for all layers with the same shape, then this
    function creates adapters from those pre-computed results.
    """
    # Effective rank per layer:
    # - If config.r is None: use layer max rank from SVD
    # - Else: cap requested rank to layer max to keep shapes valid
    max_rank = min(base_layer.in_features, base_layer.out_features)
    if soara_config.r is None:
        r = max_rank
    else:
        r = min(soara_config.r, max_rank)
    
    # Create adapter instance WITHOUT calling _initialize_svd_components
    adapter = object.__new__(SOARALinearLayer)
    nn.Module.__init__(adapter)
    
    adapter.base_layer = base_layer
    adapter.soara_config = soara_config
    adapter.adapter_name = adapter_name
    adapter.r = r
    adapter.in_features = base_layer.in_features
    adapter.out_features = base_layer.out_features
    
    # Split into principal and residual components
    U_principal = U[:, :r].contiguous()
    U_residual = U[:, r:].contiguous()
    S_principal = S[:r].contiguous()
    S_residual = S[r:].contiguous()
    V_principal = V[:r, :].contiguous()
    V_residual = V[r:, :].contiguous()
    
    # Compute W_residual = U_residual @ diag(S_residual) @ V_residual
    if U_residual.shape[1] > 0:
        W_residual = U_residual @ torch.diag(S_residual) @ V_residual
    else:
        W_residual = torch.zeros(adapter.out_features, adapter.in_features, 
                                  dtype=torch.float32, device=original_device)
    
    # Set base layer weight to W_residual (frozen)
    with torch.no_grad():
        adapter.base_layer.weight.data = W_residual.to(dtype=original_dtype, device=original_device)
        adapter.base_layer.weight.requires_grad = False
    
    # Store principal components as buffers
    adapter.register_buffer("U", U_principal.to(dtype=original_dtype, device=original_device))
    adapter.register_buffer("V", V_principal.to(dtype=original_dtype, device=original_device))
    
    # S: stored as FP32 if s_dtype_fp32 is enabled
    s_dtype = torch.float32 if soara_config.s_dtype_fp32 else original_dtype
    S_converted = S_principal.to(dtype=s_dtype, device=original_device)
    if soara_config.freeze_singular_values:
        adapter.register_buffer("S", S_converted)
    else:
        adapter.S = nn.Parameter(S_converted)
    
    # Initialize rotation matrices
    adapter._initialize_rotation_matrices()
    
    # Dropout
    if soara_config.lora_dropout > 0:
        adapter.dropout = nn.Dropout(soara_config.lora_dropout)
    else:
        adapter.dropout = nn.Identity()
    
    adapter.scaling = 1
    
    # Training state for V2 (SOARA-V2)
    if soara_config.method == "V2":
        adapter.current_layer_index = 0
        adapter.current_cycle = 0
        adapter.step_count = 0
    
    return adapter

# ============================================================================
# MODEL REPLACEMENT UTILITIES
# ============================================================================

def replace_linear_with_soara(
    model: nn.Module,
    soara_config: SOARAConfig,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    adapter_name: str = "default",
    freeze_base_model: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, nn.Module]:
    """
    Replace Linear layers in a model with SOARALinearLayer.
    
    This function implements the SOARA methodology:
    1. For each target linear layer with weight W
    2. Computes SVD: W = U @ S @ V^T  
    3. Splits into principal (top-r) and residual components
    4. Updates original layer weight to W_res = U[:,r:] @ S[r:,r:] @ V[:,r:]^T
    5. Creates adapter with W_pri = U[:,:r] @ R_U @ S[:r,:r] @ R_V^T @ V[:,:r]^T
    6. Final output: Y = X * (W_res + W_pri)
    
    Args:
        model: The model to modify
        soara_config: Configuration for the rotational adapters
        target_modules: List of module names to target (e.g., ["q_proj", "v_proj"])
        exclude_modules: List of module names to exclude (e.g., ["pooler", "classifier"])
        adapter_name: Name of the adapter
        freeze_base_model: If True, freeze all model params except adapter trainable params
        device: Optional[torch.device] = None. Explicitly specify target device for SVD ops.
               If None, defaults to current CUDA device or weight device.
    """
    import gc
    
    adapters = {}
    layers_to_replace = []
    
    # First pass: identify all layers to replace (avoid modifying dict during iteration)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should be excluded
            if exclude_modules is not None:
                should_exclude = any(excl in name for excl in exclude_modules)
                if should_exclude:
                    continue  # Skip this layer
            
            # Check if this module should be replaced
            if target_modules is None:
                should_replace = True
            else:
                should_replace = any(target in name for target in target_modules)
            
            if should_replace:
                layers_to_replace.append((name, module))
    
    print(f"  Found {len(layers_to_replace)} layers to replace")
    
    # ========== OPTIMIZED: Group layers by shape for parallel processing ==========
    # Group layers by (out_features, in_features) shape for efficient batching
    shape_groups = {}
    for name, module in layers_to_replace:
        shape = (module.out_features, module.in_features)
        shape_groups.setdefault(shape, []).append((name, module))
    
    print(f"  Grouped into {len(shape_groups)} shape groups: {[f'{s}: {len(g)} layers' for s, g in shape_groups.items()]}")
    
    # Create CUDA streams for parallel processing
    # NOTE: Disable streams on multi-GPU setups to avoid NCCL conflicts with DataParallel
    # However, batched SVD (torch.linalg.svd on stacked tensors) works fine on any GPU count
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_cuda = torch.cuda.is_available()
    use_streams = use_cuda and num_gpus == 1  # Only use streams for overlapping on single GPU
    
    if use_streams:
        num_streams = 4
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        print(f"  Using {num_streams} CUDA streams for parallel SVD (single GPU mode)")
    elif num_gpus > 1:
        streams = None
        print(f"  Multi-GPU detected ({num_gpus} GPUs) - using batched SVD without streams (NCCL-safe)")
    else:
        streams = None
    
    total_processed = 0
    
    # Process each shape group - use BATCHED SVD for maximum GPU utilization
    for shape, group_layers in shape_groups.items():
        batch_size = len(group_layers)
        print(f"  Processing shape {shape}: {batch_size} layers (batched SVD)...")
        
        # Use batched SVD when on GPU with multiple layers (works on single or multi-GPU)
        if use_cuda and batch_size > 1:
            # ========== ADAPTIVE BATCHED SVD with memory management ==========
            import time
            
            # Get dtype/device from first layer
            first_module = group_layers[0][1]
            original_dtype = first_module.weight.dtype
            original_device = first_module.weight.device
            
            # DIAGNOSTIC: Verify GPU is available and being used
            print(f"    Device: {original_device}, dtype: {original_dtype}")
            if not original_device.type == 'cuda':
                if device is not None:
                     print(f"    Moving weights from {original_device} to explicit device {device}...")
                     original_device = device
                else:
                    print(f"    WARNING: Weights are on {original_device}, not CUDA! Moving to GPU...")
                    # Use current CUDA device (respects CUDA_VISIBLE_DEVICES)
                    original_device = torch.device(f'cuda:{torch.cuda.current_device()}')
            
            # Estimate memory required for this batch
            out_feat, in_feat = shape
            # Memory for stacked weights in FP32: batch_size * out_feat * in_feat * 4 bytes
            # SVD output (U, S, V) roughly doubles this
            bytes_per_matrix = out_feat * in_feat * 4  # FP32
            estimated_gb = (batch_size * bytes_per_matrix * 2) / (1024**3)  # Input + Output
            
            # Target 4GB per batch to be safe (leaves room for other ops)
            MAX_BATCH_GB = 10.0
            
            if estimated_gb > MAX_BATCH_GB:
                # Split into smaller sub-batches
                matrices_per_batch = max(1, int(batch_size * (MAX_BATCH_GB / estimated_gb)))
                num_sub_batches = (batch_size + matrices_per_batch - 1) // matrices_per_batch
                print(f"    Memory estimate: {estimated_gb:.2f} GB > {MAX_BATCH_GB} GB limit")
                print(f"    Splitting into {num_sub_batches} sub-batches of ~{matrices_per_batch} layers each")
            else:
                matrices_per_batch = batch_size
                num_sub_batches = 1
                print(f"    Memory estimate: {estimated_gb:.2f} GB (fits in single batch)")
            
            # Process in sub-batches - create adapters IMMEDIATELY after each sub-batch
            # to avoid accumulating all SVD results in memory (which caused OOM)
            adapter_idx = 0  # Track which layer we're on in group_layers
            
            for sub_batch_idx in range(num_sub_batches):
                start_idx = sub_batch_idx * matrices_per_batch
                end_idx = min(start_idx + matrices_per_batch, batch_size)
                sub_batch = group_layers[start_idx:end_idx]
                sub_batch_size = len(sub_batch)
                
                if num_sub_batches > 1:
                    print(f"    Sub-batch {sub_batch_idx+1}/{num_sub_batches}: processing {sub_batch_size} layers...")
                
                # Stack weights for this sub-batch (supports ZeRO-3 sharded weights)
                start_time = time.time()
                sub_batch_modules = [m for _, m in sub_batch]
                weight_batch = gather_zero3_weights(sub_batch_modules, original_device)
                stack_time = time.time() - start_time
                
                if sub_batch_idx == 0:  # Only print details for first sub-batch
                    print(f"    Stacked {sub_batch_size} weights: shape={weight_batch.shape}, device={weight_batch.device}, dtype={weight_batch.dtype}")
                
                # SVD requires FP32 - convert but STAY ON GPU
                # VRAM optimization: explicitly free old tensor during conversion
                if weight_batch.dtype in (torch.bfloat16, torch.float16):
                    old_dtype = weight_batch.dtype
                    weight_batch_fp32 = weight_batch.float()  # Create FP32 copy
                    del weight_batch  # Free original BF16/FP16 tensor (saves ~50% memory)
                    weight_batch = weight_batch_fp32
                    if sub_batch_idx == 0:
                        print(f"    Converted {old_dtype} -> FP32: device={weight_batch.device}")
                
                # Ensure we're on CUDA before SVD
                assert weight_batch.device.type == 'cuda', f"weight_batch moved to {weight_batch.device}!"
                
                # Batched SVD for this sub-batch
                if sub_batch_idx == 0:
                    print(f"    Running batched SVD on {weight_batch.device}...")
                
                # NOTE: torch.cuda.synchronize() is for accurate timing only
                # PyTorch GPU ops are async - without sync, time.time() only measures CPU overhead
                # Set ENABLE_TIMING=False to skip sync and run ~5% faster (but no timing info)
                ENABLE_TIMING = False
                
                if ENABLE_TIMING:
                    torch.cuda.synchronize()  # Wait for previous ops before starting timer
                svd_start = time.time()
                
                U_batch, S_batch, V_batch = torch.linalg.svd(weight_batch, full_matrices=False)
                
                if ENABLE_TIMING:
                    torch.cuda.synchronize()  # Wait for SVD to complete before stopping timer
                svd_time = time.time() - svd_start
                
                if sub_batch_idx == 0 or num_sub_batches > 1:
                    print(f"    ✓ Sub-batch SVD complete in {svd_time:.2f}s (stack: {stack_time:.2f}s)")
                
                # Free weight batch immediately
                del weight_batch
                
                # Create adapters IMMEDIATELY from this sub-batch's SVD results
                # This prevents OOM from accumulating all sub-batch results
                for i, (name, module) in enumerate(sub_batch):
                    # Extract this layer's SVD components
                    U_i = U_batch[i]  # [out_features, k]
                    S_i = S_batch[i]  # [k]
                    V_i = V_batch[i]  # [k, in_features]
                    
                    # Create adapter with pre-computed SVD (bypass _initialize_svd_components)
                    adapted_layer = _create_adapter_from_svd(
                        module, soara_config, adapter_name,
                        U_i, S_i, V_i, original_dtype, original_device
                    )
                    
                    # Replace the module in model
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    
                    if parent_name:
                        parent_module = model.get_submodule(parent_name)
                        setattr(parent_module, child_name, adapted_layer)
                    else:
                        setattr(model, child_name, adapted_layer)
                    
                    adapters[name] = adapted_layer
                    del module
                    
                    total_processed += 1
                    adapter_idx += 1
                    if total_processed % 20 == 0:
                        print(f"    [{total_processed}/{len(layers_to_replace)}] layers processed...")
                
                # Clean up this sub-batch's SVD tensors BEFORE next sub-batch
                del U_batch, S_batch, V_batch
                torch.cuda.empty_cache()  # Free memory between sub-batches
            
        else:
            # Fallback: sequential processing (single layer or CPU)
            for name, module in group_layers:
                adapted_layer = SOARALinearLayer(module, soara_config, adapter_name)
                
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, child_name, adapted_layer)
                else:
                    setattr(model, child_name, adapted_layer)
                
                adapters[name] = adapted_layer
                del module
                total_processed += 1
        
        # Cleanup once per shape group (not per layer!) - much faster
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"  ✓ All {total_processed} layers processed")
    
    # Final cleanup
    del layers_to_replace, shape_groups
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Apply freezing strategy if requested
    if freeze_base_model:
        # 1) Freeze ALL parameters in the model first
        for param in model.parameters():
            param.requires_grad = False
        
        # 2) Unfreeze adapter trainable parameters (method-agnostic)
        for adapter in adapters.values():
            adapter.freeze_base_and_unfreeze_adapters()
    
    return adapters

