"""SOARA V2B persistent butterfly backend + V2A persistent Givens backend.

This module wraps the existing SOARA implementation and adds memory-stable
persistent operator modes for both V2 variants:

- V2B (butterfly-sequential): persistent PhaseIndexedButterflyOperator
- V2A (Givens-sequential):    persistent PhaseIndexedGivensOperator

In both modes, we keep a persistent operator for R_U and R_V, reset angles
between phases, and switch phase index maps instead of deleting and
reallocating modules. Both reuse the same Triton fused kernels.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

from rotational_pissa_unified import (
    SOARAConfig as _BaseSOARAConfig,
    SOARALinearLayer as _BaseSOARALinearLayer,
    SOARATrainer as _BaseSOARATrainer,
    dequantize_params4bit,
    generate_givens_pairings,
    replace_linear_with_soara as _base_replace_linear_with_soara,
)


@dataclass
class SOARAConfig(_BaseSOARAConfig):
    # V2B toggle. Defaults to True in this temp backend for V2+butterfly+sequential.
    butterfly_persistent: bool = True
    # V2A toggle. Defaults to True for V2+Givens sequential persistent mode.
    givens_persistent: bool = True
    # Runtime backend for phase-rotation application: auto | triton | torch.
    butterfly_backend: str = "auto"
    # Auto mode threshold for using Triton path.
    butterfly_triton_min_dim: int = 1024
    butterfly_triton_min_batch_rows: int = 8


if _TRITON_AVAILABLE:

    @triton.jit
    def _butterfly_forward_kernel(
        x_ptr,
        out_ptr,
        p_ptr,
        q_ptr,
        cos_ptr,
        sin_ptr,
        stride_xn,
        stride_on,
        n_rows,
        n_pairs,
        logical_dim,
        transpose: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_r = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)

        mask_n = offs_n < n_rows
        mask_r = offs_r < n_pairs

        p = tl.load(p_ptr + offs_r, mask=mask_r, other=0).to(tl.int32)
        q = tl.load(q_ptr + offs_r, mask=mask_r, other=0).to(tl.int32)

        p_valid = p < logical_dim
        q_valid = q < logical_dim

        mask_p = mask_n[:, None] & mask_r[None, :] & p_valid[None, :]
        mask_q = mask_n[:, None] & mask_r[None, :] & q_valid[None, :]

        c = tl.load(cos_ptr + offs_r, mask=mask_r, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)

        x_p_ptrs = x_ptr + offs_n[:, None] * stride_xn + p[None, :]
        x_q_ptrs = x_ptr + offs_n[:, None] * stride_xn + q[None, :]
        out_p_ptrs = out_ptr + offs_n[:, None] * stride_on + p[None, :]
        out_q_ptrs = out_ptr + offs_n[:, None] * stride_on + q[None, :]

        x_p = tl.load(x_p_ptrs, mask=mask_p, other=0).to(tl.float32)
        x_q = tl.load(x_q_ptrs, mask=mask_q, other=0).to(tl.float32)

        if transpose:
            y_p = c[None, :] * x_p - s[None, :] * x_q
            y_q = s[None, :] * x_p + c[None, :] * x_q
        else:
            y_p = c[None, :] * x_p + s[None, :] * x_q
            y_q = -s[None, :] * x_p + c[None, :] * x_q

        tl.store(out_p_ptrs, y_p, mask=mask_p)
        tl.store(out_q_ptrs, y_q, mask=mask_q)


    @triton.jit
    def _butterfly_backward_kernel(
        x_ptr,
        grad_out_ptr,
        grad_x_ptr,
        grad_theta_ptr,
        p_ptr,
        q_ptr,
        cos_ptr,
        sin_ptr,
        stride_xn,
        stride_gon,
        stride_gxn,
        n_rows,
        n_pairs,
        logical_dim,
        transpose: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_r = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)

        mask_n = offs_n < n_rows
        mask_r = offs_r < n_pairs

        p = tl.load(p_ptr + offs_r, mask=mask_r, other=0).to(tl.int32)
        q = tl.load(q_ptr + offs_r, mask=mask_r, other=0).to(tl.int32)

        p_valid = p < logical_dim
        q_valid = q < logical_dim

        mask_p = mask_n[:, None] & mask_r[None, :] & p_valid[None, :]
        mask_q = mask_n[:, None] & mask_r[None, :] & q_valid[None, :]

        c = tl.load(cos_ptr + offs_r, mask=mask_r, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)

        x_p_ptrs = x_ptr + offs_n[:, None] * stride_xn + p[None, :]
        x_q_ptrs = x_ptr + offs_n[:, None] * stride_xn + q[None, :]
        g_p_ptrs = grad_out_ptr + offs_n[:, None] * stride_gon + p[None, :]
        g_q_ptrs = grad_out_ptr + offs_n[:, None] * stride_gon + q[None, :]
        gx_p_ptrs = grad_x_ptr + offs_n[:, None] * stride_gxn + p[None, :]
        gx_q_ptrs = grad_x_ptr + offs_n[:, None] * stride_gxn + q[None, :]

        x_p = tl.load(x_p_ptrs, mask=mask_p, other=0).to(tl.float32)
        x_q = tl.load(x_q_ptrs, mask=mask_q, other=0).to(tl.float32)
        g_p = tl.load(g_p_ptrs, mask=mask_p, other=0).to(tl.float32)
        g_q = tl.load(g_q_ptrs, mask=mask_q, other=0).to(tl.float32)

        if transpose:
            grad_x_p = c[None, :] * g_p + s[None, :] * g_q
            grad_x_q = -s[None, :] * g_p + c[None, :] * g_q
            grad_theta = g_p * (-s[None, :] * x_p - c[None, :] * x_q)
            grad_theta += g_q * (c[None, :] * x_p - s[None, :] * x_q)
        else:
            grad_x_p = c[None, :] * g_p - s[None, :] * g_q
            grad_x_q = s[None, :] * g_p + c[None, :] * g_q
            grad_theta = g_p * (-s[None, :] * x_p + c[None, :] * x_q)
            grad_theta += g_q * (-c[None, :] * x_p - s[None, :] * x_q)

        tl.store(gx_p_ptrs, grad_x_p, mask=mask_p)
        tl.store(gx_q_ptrs, grad_x_q, mask=mask_q)

        grad_theta_sum = tl.sum(grad_theta, axis=0)
        tl.atomic_add(grad_theta_ptr + offs_r, grad_theta_sum, mask=mask_r)


class _PhaseIndexedButterflyTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        thetas: torch.Tensor,
        p_idx: torch.Tensor,
        q_idx: torch.Tensor,
        transpose: bool,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton backend was requested but Triton is not available")
        if not x.is_cuda:
            raise RuntimeError("Triton backend requires CUDA tensors")

        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        n_rows = x_2d.shape[0]
        logical_dim = x_2d.shape[1]
        n_pairs = p_idx.numel()

        p_i32 = p_idx.to(device=x_2d.device, dtype=torch.int32).contiguous()
        q_i32 = q_idx.to(device=x_2d.device, dtype=torch.int32).contiguous()
        cos_t = torch.cos(thetas).to(device=x_2d.device, dtype=x_2d.dtype)
        sin_t = torch.sin(thetas).to(device=x_2d.device, dtype=x_2d.dtype)

        out_2d = torch.empty_like(x_2d)

        block_n = 32
        block_r = 64
        grid = (triton.cdiv(n_rows, block_n), triton.cdiv(n_pairs, block_r))
        _butterfly_forward_kernel[grid](
            x_2d,
            out_2d,
            p_i32,
            q_i32,
            cos_t,
            sin_t,
            x_2d.stride(0),
            out_2d.stride(0),
            n_rows,
            n_pairs,
            logical_dim,
            transpose=bool(transpose),
            BLOCK_N=block_n,
            BLOCK_R=block_r,
            num_warps=4,
        )

        ctx.save_for_backward(x_2d, thetas, p_i32, q_i32)
        ctx.transpose = bool(transpose)
        ctx.original_shape = tuple(original_shape)
        return out_2d.view(*original_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_2d, thetas, p_i32, q_i32 = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()

        n_rows = x_2d.shape[0]
        logical_dim = x_2d.shape[1]
        n_pairs = p_i32.numel()

        cos_t = torch.cos(thetas).to(device=x_2d.device, dtype=grad_out_2d.dtype)
        sin_t = torch.sin(thetas).to(device=x_2d.device, dtype=grad_out_2d.dtype)

        grad_x_2d = torch.empty_like(grad_out_2d)
        grad_theta_accum = torch.zeros(n_pairs, device=x_2d.device, dtype=torch.float32)

        block_n = 32
        block_r = 64
        grid = (triton.cdiv(n_rows, block_n), triton.cdiv(n_pairs, block_r))
        _butterfly_backward_kernel[grid](
            x_2d,
            grad_out_2d,
            grad_x_2d,
            grad_theta_accum,
            p_i32,
            q_i32,
            cos_t,
            sin_t,
            x_2d.stride(0),
            grad_out_2d.stride(0),
            grad_x_2d.stride(0),
            n_rows,
            n_pairs,
            logical_dim,
            transpose=ctx.transpose,
            BLOCK_N=block_n,
            BLOCK_R=block_r,
            num_warps=4,
        )

        grad_theta = grad_theta_accum.to(dtype=thetas.dtype)
        grad_x = grad_x_2d.view(*ctx.original_shape)
        return grad_x, grad_theta, None, None, None


def _apply_phase_indexed_rotation_triton(
    x: torch.Tensor,
    thetas: torch.Tensor,
    p_idx: torch.Tensor,
    q_idx: torch.Tensor,
    transpose: bool,
) -> torch.Tensor:
    return _PhaseIndexedButterflyTritonFn.apply(x, thetas, p_idx, q_idx, bool(transpose))


class PhaseIndexedButterflyOperator(nn.Module):
    """Persistent butterfly operator with phase-indexed sparse pair maps.

    Stores only trainable angles (dense length d/2) and sparse index maps for
    each butterfly phase k in [d, d/2, ..., 2].
    """

    def __init__(
        self,
        d: int,
        k_values: List[int],
        block_size: int = 1,
        backend: str = "auto",
        triton_min_dim: int = 1024,
        triton_min_batch_rows: int = 8,
    ):
        super().__init__()
        self.d = d
        self.k_values = list(k_values)
        self.block_size = block_size
        self.backend = backend.lower()
        self.triton_min_dim = int(triton_min_dim)
        self.triton_min_batch_rows = int(triton_min_batch_rows)

        if self.backend not in {"auto", "triton", "torch"}:
            raise ValueError(
                f"Unsupported butterfly backend '{backend}'. Expected one of auto|triton|torch"
            )

        if not self.k_values:
            raise ValueError("k_values cannot be empty")

        self.total_rotations = self.d // 2
        self.thetas = nn.Parameter(torch.zeros(self.total_rotations))

        p_by_phase = []
        q_by_phase = []
        for k in self.k_values:
            p_idx, q_idx = self._build_phase_indices(self.d, k)
            p_by_phase.append(p_idx)
            q_by_phase.append(q_idx)

        self.register_buffer("p_indices_by_phase", torch.stack(p_by_phase, dim=0))
        self.register_buffer("q_indices_by_phase", torch.stack(q_by_phase, dim=0))
        self.register_buffer("p_indices_by_phase_i32", self.p_indices_by_phase.to(torch.int32))
        self.register_buffer("q_indices_by_phase_i32", self.q_indices_by_phase.to(torch.int32))
        self.register_buffer("active_phase", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _build_phase_indices(d: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if d % k != 0:
            raise ValueError(f"d={d} must be divisible by k={k}")

        n_blocks = d // k
        half_k = k // 2
        p_indices = []
        q_indices = []

        for block_idx in range(n_blocks):
            base_idx = block_idx * k
            for i in range(half_k):
                p_indices.append(base_idx + i)
                q_indices.append(base_idx + i + half_k)

        p = torch.tensor(p_indices, dtype=torch.long)
        q = torch.tensor(q_indices, dtype=torch.long)

        if p.numel() != d // 2 or q.numel() != d // 2:
            raise ValueError(
                f"Unexpected number of rotations for k={k}: {p.numel()} vs expected {d // 2}"
            )

        return p, q

    def set_phase(self, phase_idx: int) -> None:
        phase_idx = int(phase_idx)
        if phase_idx < 0 or phase_idx >= len(self.k_values):
            raise IndexError(f"phase_idx={phase_idx} out of range [0, {len(self.k_values) - 1}]")
        self.active_phase.fill_(phase_idx)

    def reset(self) -> None:
        with torch.no_grad():
            self.thetas.zero_()

    def _active_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.active_phase.item())
        return self.p_indices_by_phase[idx], self.q_indices_by_phase[idx]

    def _active_indices_triton(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.active_phase.item())
        return self.p_indices_by_phase_i32[idx], self.q_indices_by_phase_i32[idx]

    def forward(self) -> torch.Tensor:
        """Build dense active-phase rotation matrix for compatibility/debug."""
        p_idx, q_idx = self._active_indices()

        R = torch.eye(self.d, device=self.thetas.device, dtype=self.thetas.dtype)
        cos_t = torch.cos(self.thetas)
        sin_t = torch.sin(self.thetas)

        R.diagonal().scatter_(0, p_idx, cos_t)
        R.diagonal().scatter_(0, q_idx, cos_t)
        R[p_idx, q_idx] = -sin_t
        R[q_idx, p_idx] = sin_t
        return R

    def apply_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply x @ R(active_phase) using sparse pair maps."""
        if self._should_use_triton(x):
            p_idx, q_idx = self._active_indices_triton()
            try:
                return _apply_phase_indexed_rotation_triton(
                    x=x,
                    thetas=self.thetas,
                    p_idx=p_idx,
                    q_idx=q_idx,
                    transpose=False,
                )
            except Exception:
                if self.backend == "triton":
                    raise

        p_idx, q_idx = self._active_indices()

        cos_t = torch.cos(self.thetas)
        sin_t = torch.sin(self.thetas)
        if cos_t.dtype != x.dtype:
            cos_t = cos_t.to(x.dtype)
            sin_t = sin_t.to(x.dtype)

        y = x.clone()
        x_p = x[..., p_idx]
        x_q = x[..., q_idx]

        y[..., p_idx] = cos_t * x_p + sin_t * x_q
        y[..., q_idx] = -sin_t * x_p + cos_t * x_q
        return y

    def apply_transpose(self, x: torch.Tensor) -> torch.Tensor:
        """Apply x @ R(active_phase).T using sparse pair maps."""
        if self._should_use_triton(x):
            p_idx, q_idx = self._active_indices_triton()
            try:
                return _apply_phase_indexed_rotation_triton(
                    x=x,
                    thetas=self.thetas,
                    p_idx=p_idx,
                    q_idx=q_idx,
                    transpose=True,
                )
            except Exception:
                if self.backend == "triton":
                    raise

        p_idx, q_idx = self._active_indices()

        cos_t = torch.cos(self.thetas)
        sin_t = torch.sin(self.thetas)
        if cos_t.dtype != x.dtype:
            cos_t = cos_t.to(x.dtype)
            sin_t = sin_t.to(x.dtype)

        y = x.clone()
        x_p = x[..., p_idx]
        x_q = x[..., q_idx]

        y[..., p_idx] = cos_t * x_p - sin_t * x_q
        y[..., q_idx] = sin_t * x_p + cos_t * x_q
        return y

    def _should_use_triton(self, x: torch.Tensor) -> bool:
        if self.backend == "torch":
            return False
        if not _TRITON_AVAILABLE:
            return False
        if not x.is_cuda:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if x.shape[-1] > self.d:
            return False

        n_rows = x.numel() // x.shape[-1]
        if self.backend == "triton":
            return True

        # Auto mode: only use Triton once dimensions are large enough.
        effective_dim = max(x.shape[-1], self.d)
        return effective_dim >= self.triton_min_dim and n_rows >= self.triton_min_batch_rows


class PhaseIndexedGivensOperator(nn.Module):
    """Persistent Givens operator with phase-indexed sparse pair maps.

    Analogous to PhaseIndexedButterflyOperator but uses round-robin
    tournament pairings from generate_givens_pairings(). Stores only
    trainable angles (max r//2) and sparse index maps for each Givens
    phase. Reuses the same Triton kernels as the butterfly operator.
    """

    def __init__(
        self,
        r: int,
        n_layers: Optional[int] = None,
        backend: str = "auto",
        triton_min_dim: int = 1024,
        triton_min_batch_rows: int = 8,
    ):
        super().__init__()
        self.r = r
        self.n_layers = n_layers if n_layers is not None else max(r - 1, 1)
        self.backend = backend.lower()
        self.triton_min_dim = int(triton_min_dim)
        self.triton_min_batch_rows = int(triton_min_batch_rows)

        if self.backend not in {"auto", "triton", "torch"}:
            raise ValueError(
                f"Unsupported backend '{backend}'. Expected one of auto|triton|torch"
            )

        # Generate all Givens phase pairings via round-robin tournament.
        all_pairings = generate_givens_pairings(self.r, self.n_layers)
        if not all_pairings:
            raise ValueError(f"generate_givens_pairings returned empty for r={r}, n_layers={self.n_layers}")

        self.num_phases = len(all_pairings)

        # Find the maximum number of pairs across all phases (may vary for odd r).
        max_pairs = max(len(pairs) for pairs in all_pairings)
        self.max_pairs = max_pairs

        # Trainable angles — one per pair in the densest phase.
        self.thetas = nn.Parameter(torch.zeros(max_pairs))

        # Build phase index buffers. Phases with fewer pairs are padded with
        # sentinel indices (r) which the Triton kernel masks out via
        # p_valid = p < logical_dim.
        p_by_phase = []
        q_by_phase = []
        pairs_count_list = []
        for pairs in all_pairings:
            p_list = [p for p, q in pairs]
            q_list = [q for p, q in pairs]
            n_actual = len(pairs)
            pairs_count_list.append(n_actual)
            # Pad to max_pairs with sentinel index (r) — will be masked by kernel.
            while len(p_list) < max_pairs:
                p_list.append(r)  # sentinel: >= logical_dim, masked out
                q_list.append(r)
            p_by_phase.append(torch.tensor(p_list, dtype=torch.long))
            q_by_phase.append(torch.tensor(q_list, dtype=torch.long))

        self.register_buffer("p_indices_by_phase", torch.stack(p_by_phase, dim=0))
        self.register_buffer("q_indices_by_phase", torch.stack(q_by_phase, dim=0))
        self.register_buffer("p_indices_by_phase_i32", self.p_indices_by_phase.to(torch.int32))
        self.register_buffer("q_indices_by_phase_i32", self.q_indices_by_phase.to(torch.int32))
        self.register_buffer("active_phase", torch.tensor(0, dtype=torch.long))
        self.register_buffer("pairs_per_phase", torch.tensor(pairs_count_list, dtype=torch.long))

    def set_phase(self, phase_idx: int) -> None:
        phase_idx = int(phase_idx)
        if phase_idx < 0 or phase_idx >= self.num_phases:
            raise IndexError(f"phase_idx={phase_idx} out of range [0, {self.num_phases - 1}]")
        self.active_phase.fill_(phase_idx)

    def reset(self) -> None:
        with torch.no_grad():
            self.thetas.zero_()

    @property
    def active_num_pairs(self) -> int:
        """Number of actual (non-padded) pairs in the current phase."""
        return int(self.pairs_per_phase[int(self.active_phase.item())].item())

    def _active_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.active_phase.item())
        return self.p_indices_by_phase[idx], self.q_indices_by_phase[idx]

    def _active_indices_triton(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.active_phase.item())
        return self.p_indices_by_phase_i32[idx], self.q_indices_by_phase_i32[idx]

    def forward(self) -> torch.Tensor:
        """Build dense active-phase rotation matrix for compatibility/debug."""
        p_idx, q_idx = self._active_indices()
        n = self.active_num_pairs

        R = torch.eye(self.r, device=self.thetas.device, dtype=self.thetas.dtype)
        # Only use the first n angles and indices (rest are padding sentinels).
        cos_t = torch.cos(self.thetas[:n])
        sin_t = torch.sin(self.thetas[:n])
        p_valid = p_idx[:n]
        q_valid = q_idx[:n]

        R.diagonal().scatter_(0, p_valid, cos_t)
        R.diagonal().scatter_(0, q_valid, cos_t)
        R[p_valid, q_valid] = -sin_t
        R[q_valid, p_valid] = sin_t
        return R

    def apply_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply x @ R(active_phase) using sparse pair maps."""
        if self._should_use_triton(x):
            p_idx, q_idx = self._active_indices_triton()
            try:
                return _apply_phase_indexed_rotation_triton(
                    x=x,
                    thetas=self.thetas,
                    p_idx=p_idx,
                    q_idx=q_idx,
                    transpose=False,
                )
            except Exception:
                if self.backend == "triton":
                    raise

        p_idx, q_idx = self._active_indices()
        n = self.active_num_pairs

        cos_t = torch.cos(self.thetas[:n])
        sin_t = torch.sin(self.thetas[:n])
        if cos_t.dtype != x.dtype:
            cos_t = cos_t.to(x.dtype)
            sin_t = sin_t.to(x.dtype)

        p_valid = p_idx[:n]
        q_valid = q_idx[:n]

        y = x.clone()
        x_p = x[..., p_valid]
        x_q = x[..., q_valid]

        y[..., p_valid] = cos_t * x_p + sin_t * x_q
        y[..., q_valid] = -sin_t * x_p + cos_t * x_q
        return y

    def apply_transpose(self, x: torch.Tensor) -> torch.Tensor:
        """Apply x @ R(active_phase).T using sparse pair maps."""
        if self._should_use_triton(x):
            p_idx, q_idx = self._active_indices_triton()
            try:
                return _apply_phase_indexed_rotation_triton(
                    x=x,
                    thetas=self.thetas,
                    p_idx=p_idx,
                    q_idx=q_idx,
                    transpose=True,
                )
            except Exception:
                if self.backend == "triton":
                    raise

        p_idx, q_idx = self._active_indices()
        n = self.active_num_pairs

        cos_t = torch.cos(self.thetas[:n])
        sin_t = torch.sin(self.thetas[:n])
        if cos_t.dtype != x.dtype:
            cos_t = cos_t.to(x.dtype)
            sin_t = sin_t.to(x.dtype)

        p_valid = p_idx[:n]
        q_valid = q_idx[:n]

        y = x.clone()
        x_p = x[..., p_valid]
        x_q = x[..., q_valid]

        y[..., p_valid] = cos_t * x_p - sin_t * x_q
        y[..., q_valid] = sin_t * x_p + cos_t * x_q
        return y

    def _should_use_triton(self, x: torch.Tensor) -> bool:
        if self.backend == "torch":
            return False
        if not _TRITON_AVAILABLE:
            return False
        if not x.is_cuda:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if x.shape[-1] > self.r:
            return False

        n_rows = x.numel() // x.shape[-1]
        if self.backend == "triton":
            return True

        # Auto mode: only use Triton once dimensions are large enough.
        effective_dim = max(x.shape[-1], self.r)
        return effective_dim >= self.triton_min_dim and n_rows >= self.triton_min_batch_rows


class SOARALinearLayer(_BaseSOARALinearLayer):
    """SOARA layer with optional persistent V2 butterfly-sequential or Givens-sequential behavior."""

    def _use_persistent_v2b(self) -> bool:
        cfg = self.soara_config
        return (
            cfg.method == "V2"
            and cfg.use_butterfly
            and cfg.butterfly_sequential
            and getattr(cfg, "butterfly_persistent", True)
        )

    def _use_persistent_v2a(self) -> bool:
        cfg = self.soara_config
        return (
            cfg.method == "V2"
            and not cfg.use_butterfly
            and getattr(cfg, "givens_persistent", True)
        )

    def _init_V2(self):
        if self._use_persistent_v2a():
            return self._init_V2_persistent_givens()
        if not self._use_persistent_v2b():
            return super()._init_V2()

        device = self.U.device
        dtype = self.U.dtype

        d_padded = 2 ** math.ceil(math.log2(float(self.r)))
        if d_padded < self.r:
            d_padded = self.r

        self.butterfly_k_values = []
        k = d_padded
        while k >= 2:
            self.butterfly_k_values.append(k)
            k = k // 2

        self.butterfly_d_padded = d_padded
        self.butterfly_needs_padding = d_padded != self.r
        self.current_butterfly_idx = 0
        self.current_butterfly_cycle = 0

        block_size = self.soara_config.butterfly_block_size
        backend = getattr(self.soara_config, "butterfly_backend", "auto")
        triton_min_dim = getattr(self.soara_config, "butterfly_triton_min_dim", 1024)
        triton_min_batch_rows = getattr(self.soara_config, "butterfly_triton_min_batch_rows", 8)
        self.current_butterfly_u = PhaseIndexedButterflyOperator(
            d_padded,
            self.butterfly_k_values,
            block_size,
            backend=backend,
            triton_min_dim=triton_min_dim,
            triton_min_batch_rows=triton_min_batch_rows,
        ).to(device=device, dtype=dtype)
        self.current_butterfly_v = PhaseIndexedButterflyOperator(
            d_padded,
            self.butterfly_k_values,
            block_size,
            backend=backend,
            triton_min_dim=triton_min_dim,
            triton_min_batch_rows=triton_min_batch_rows,
        ).to(device=device, dtype=dtype)
        self.current_butterfly_u.set_phase(0)
        self.current_butterfly_v.set_phase(0)

        print(
            f"    🦋 Persistent Sequential Butterfly (V2B): {len(self.butterfly_k_values)} phases, "
            f"{self.current_butterfly_u.thetas.numel()} trainable angles/side, "
            f"backend={backend} (triton_available={_TRITON_AVAILABLE})"
        )

    def _init_V2_persistent_givens(self):
        """Initialize persistent Givens operators for V2A sequential mode."""
        device = self.U.device
        dtype = self.U.dtype

        n_layers = self.soara_config.n_givens_layers or max(self.r - 1, 1)
        backend = getattr(self.soara_config, "butterfly_backend", "auto")
        triton_min_dim = getattr(self.soara_config, "butterfly_triton_min_dim", 1024)
        triton_min_batch_rows = getattr(self.soara_config, "butterfly_triton_min_batch_rows", 8)

        self.current_givens_u = PhaseIndexedGivensOperator(
            r=self.r,
            n_layers=n_layers,
            backend=backend,
            triton_min_dim=triton_min_dim,
            triton_min_batch_rows=triton_min_batch_rows,
        ).to(device=device, dtype=dtype)
        self.current_givens_v = PhaseIndexedGivensOperator(
            r=self.r,
            n_layers=n_layers,
            backend=backend,
            triton_min_dim=triton_min_dim,
            triton_min_batch_rows=triton_min_batch_rows,
        ).to(device=device, dtype=dtype)
        self.current_givens_u.set_phase(0)
        self.current_givens_v.set_phase(0)

        # V2A tracking state (mirrors V2B's butterfly_idx / butterfly_cycle).
        self.current_layer_index = 0
        self.current_cycle = 0

        print(
            f"    🔄 Persistent Sequential Givens (V2A): {self.current_givens_u.num_phases} phases, "
            f"{self.current_givens_u.max_pairs} max angles/side, "
            f"backend={backend} (triton_available={_TRITON_AVAILABLE})"
        )

    def _upgrade_to_persistent_butterfly(self) -> None:
        if not self._use_persistent_v2b():
            return
        # Re-initialize the V2 branch with persistent operators.
        self._init_V2()

    def _upgrade_to_persistent_givens(self) -> None:
        if not self._use_persistent_v2a():
            return
        # Re-initialize the V2 branch with persistent Givens operators.
        self._init_V2()

    def _apply_operator_rotation(self, op, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation via a persistent operator (butterfly or Givens)."""
        if isinstance(op, PhaseIndexedButterflyOperator) and self.butterfly_needs_padding:
            # Triton path can operate directly on logical (unpadded) rank via index masking.
            if op._should_use_triton(x):
                return op.apply_rotation(x)
            pad_size = self.butterfly_d_padded - self.r
            x = F.pad(x, (0, pad_size), value=0)
            x = op.apply_rotation(x)
            return x[..., : self.r]
        return op.apply_rotation(x)

    def _apply_operator_transpose(self, op, x: torch.Tensor) -> torch.Tensor:
        """Apply transpose via a persistent operator (butterfly or Givens)."""
        if isinstance(op, PhaseIndexedButterflyOperator) and self.butterfly_needs_padding:
            # Triton path can operate directly on logical (unpadded) rank via index masking.
            if op._should_use_triton(x):
                return op.apply_transpose(x)
            pad_size = self.butterfly_d_padded - self.r
            x = F.pad(x, (0, pad_size), value=0)
            x = op.apply_transpose(x)
            return x[..., : self.r]
        return op.apply_transpose(x)

    def get_rotation_matrices(self):
        if self._use_persistent_v2a():
            R_U = self.current_givens_u()
            R_V = self.current_givens_v()
            return R_U, R_V

        if not self._use_persistent_v2b():
            return super().get_rotation_matrices()

        R_U = self.current_butterfly_u()
        R_V = self.current_butterfly_v()
        if self.butterfly_needs_padding:
            R_U = R_U[: self.r, : self.r]
            R_V = R_V[: self.r, : self.r]
        return R_U, R_V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_persistent_v2a():
            return self._forward_persistent_givens(x)
        if not self._use_persistent_v2b():
            return super().forward(x)
        return self._forward_persistent_butterfly(x)

    def _forward_persistent_butterfly(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)

        U_current = dequantize_params4bit(self.U) if hasattr(self.U, "quant_state") else self.U
        V_current = dequantize_params4bit(self.V) if hasattr(self.V, "quant_state") else self.V

        target_dtype = x.dtype
        if U_current.dtype != target_dtype:
            U_current = U_current.to(target_dtype)
        if V_current.dtype != target_dtype:
            V_current = V_current.to(target_dtype)

        S_current = self.S.to(target_dtype) if self.S.dtype != target_dtype else self.S

        x_adapted = self.dropout(x)
        x_adapted = x_adapted @ V_current.T
        x_adapted = self._apply_operator_transpose(self.current_butterfly_v, x_adapted)
        x_adapted = x_adapted * S_current
        x_adapted = self._apply_operator_transpose(self.current_butterfly_u, x_adapted)
        x_adapted = x_adapted @ U_current.T

        result = result + x_adapted * self.scaling
        return result

    def _forward_persistent_givens(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)

        U_current = dequantize_params4bit(self.U) if hasattr(self.U, "quant_state") else self.U
        V_current = dequantize_params4bit(self.V) if hasattr(self.V, "quant_state") else self.V

        target_dtype = x.dtype
        if U_current.dtype != target_dtype:
            U_current = U_current.to(target_dtype)
        if V_current.dtype != target_dtype:
            V_current = V_current.to(target_dtype)

        S_current = self.S.to(target_dtype) if self.S.dtype != target_dtype else self.S

        x_adapted = self.dropout(x)
        x_adapted = x_adapted @ V_current.T
        x_adapted = self._apply_operator_transpose(self.current_givens_v, x_adapted)
        x_adapted = x_adapted * S_current
        x_adapted = self._apply_operator_transpose(self.current_givens_u, x_adapted)
        x_adapted = x_adapted @ U_current.T

        result = result + x_adapted * self.scaling
        return result

    def step_phase(self):
        if self._use_persistent_v2a():
            return self._step_phase_persistent_givens()
        if not self._use_persistent_v2b():
            return super().step_phase()
        return self._step_phase_persistent_butterfly()

    def _step_phase_persistent_butterfly(self):
        params_before = sum(p.numel() for p in self.parameters() if p.requires_grad)

        with torch.no_grad():
            # Merge active phase into U and V.
            self.U.copy_(self._apply_operator_rotation(self.current_butterfly_u, self.U))
            v_t = self._apply_operator_transpose(self.current_butterfly_v, self.V.T)
            self.V.copy_(v_t.T)

            # Advance phase index and cycle.
            self.current_butterfly_idx = (self.current_butterfly_idx + 1) % len(self.butterfly_k_values)
            if self.current_butterfly_idx == 0:
                self.current_butterfly_cycle += 1

            self.current_butterfly_u.set_phase(self.current_butterfly_idx)
            self.current_butterfly_v.set_phase(self.current_butterfly_idx)

            # Reinitialize active angles to identity.
            self.current_butterfly_u.reset()
            self.current_butterfly_v.reset()

        params_after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (params_before, params_after)

    def _step_phase_persistent_givens(self):
        params_before = sum(p.numel() for p in self.parameters() if p.requires_grad)

        with torch.no_grad():
            # Merge active Givens phase into U and V.
            # U_new = U_old @ R_U  (apply rotation to rows of U)
            self.U.copy_(self._apply_operator_rotation(self.current_givens_u, self.U))
            # V_new = R_V @ V_old  (apply transpose to columns, i.e., rows of V.T)
            v_t = self._apply_operator_transpose(self.current_givens_v, self.V.T)
            self.V.copy_(v_t.T)

            # Advance phase index and cycle.
            self.current_layer_index = (self.current_layer_index + 1) % self.current_givens_u.num_phases
            if self.current_layer_index == 0:
                self.current_cycle += 1

            self.current_givens_u.set_phase(self.current_layer_index)
            self.current_givens_v.set_phase(self.current_layer_index)

            # Reinitialize active angles to identity.
            self.current_givens_u.reset()
            self.current_givens_v.reset()

        params_after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (params_before, params_after)


SOARATrainer = _BaseSOARATrainer


def _should_use_persistent_v2b(soara_config: _BaseSOARAConfig) -> bool:
    return (
        soara_config.method == "V2"
        and soara_config.use_butterfly
        and soara_config.butterfly_sequential
        and getattr(soara_config, "butterfly_persistent", True)
    )


def _should_use_persistent_v2a(soara_config: _BaseSOARAConfig) -> bool:
    return (
        soara_config.method == "V2"
        and not soara_config.use_butterfly
        and getattr(soara_config, "givens_persistent", True)
    )


def replace_linear_with_soara(
    model: nn.Module,
    soara_config: _BaseSOARAConfig,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    adapter_name: str = "default",
    freeze_base_model: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, nn.Module]:
    """Wrap base replace function and upgrade V2 sequential adapters (butterfly or Givens)."""
    adapters = _base_replace_linear_with_soara(
        model=model,
        soara_config=soara_config,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        adapter_name=adapter_name,
        freeze_base_model=freeze_base_model,
        device=device,
    )

    if _should_use_persistent_v2b(soara_config):
        for adapter in adapters.values():
            if not isinstance(adapter, SOARALinearLayer):
                adapter.__class__ = SOARALinearLayer
            adapter._upgrade_to_persistent_butterfly()

    if _should_use_persistent_v2a(soara_config):
        for adapter in adapters.values():
            if not isinstance(adapter, SOARALinearLayer):
                adapter.__class__ = SOARALinearLayer
            adapter._upgrade_to_persistent_givens()

    return adapters
