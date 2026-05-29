"""Hadamard low-pass activation compression for `nn.Linear` modules.

Lifted from the `instant_lowpass` worktree (the GraLoRA-adapter version of
this technique) and adapted for plain `nn.Linear`. The compression scheme:

- Forward is exactly `F.linear(x, weight, bias)`.
- On the backward path:
    - The input gradient `grad_x = grad_output @ weight` is exact (no
      compression touches `grad_output` or `weight`).
    - The parameter gradient `grad_w` is computed from low-pass projections
      of both `x` and `grad_output`. Each contiguous chunk of `chunk_size`
      tokens is Hadamard- (or DCT- or Haar-) transformed along the sequence
      axis and only the lowest `keep` coefficients are stored. Memory drops
      by `chunk_size / keep` for the saved activation. The resulting
      param-grad is an approximation but Adam's EMA smooths the noise.

This is the same trick `_LowpassGraloraAdapterFunction` used in the
instant-lowpass worktree, just applied at the `nn.Linear` level so it
works with the full fine-tune path (no LoRA needed).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


_TRITON_PIECEWISE_PROJECT: Callable[[Tensor, Tensor], Tensor] | None = None
_TRITON_PIECEWISE_PROJECT_IMPORT_ERROR: Exception | None = None
_PROJECTOR_CACHE: dict[tuple[str, int, int, str, int, str], Tensor] = {}


@dataclass(frozen=True)
class LowpassConfig:
    projector_kind: str = "hadamard"
    chunk_size: int = 64
    keep: int = 32
    min_hidden_dim: int = 64
    hadamard_backend: str = "auto"
    enabled: bool = True

    def __post_init__(self) -> None:
        projector_kind = str(self.projector_kind).lower()
        hadamard_backend = str(self.hadamard_backend).lower().replace("_", "-")
        if hadamard_backend in {"fast", "triton"}:
            hadamard_backend = "piecewise"
        object.__setattr__(self, "projector_kind", projector_kind)
        object.__setattr__(self, "hadamard_backend", hadamard_backend)
        if projector_kind not in {"hadamard", "dct", "haar"}:
            raise ValueError(f"unknown projector_kind {self.projector_kind!r}")
        if hadamard_backend not in {"auto", "piecewise", "dense"}:
            raise ValueError(f"unknown hadamard_backend {self.hadamard_backend!r}")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.keep < 1 or self.keep > self.chunk_size:
            raise ValueError("keep must be in [1, chunk_size]")
        if self.min_hidden_dim < 0:
            raise ValueError("min_hidden_dim must be non-negative")
        if projector_kind in {"hadamard", "haar"} and not _is_power_of_two(self.chunk_size):
            raise ValueError(f"{projector_kind} requires power-of-two chunk_size")


def _torch_is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _bit_reverse(value: int, width: int) -> int:
    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _hadamard_index_for_sequency(sequency: int, width: int) -> int:
    gray = sequency ^ (sequency >> 1)
    return _bit_reverse(gray, width)


def _projector_cache_key(
    kind: str,
    seq_len: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[str, int, int, str, int, str]:
    device_index = -1 if device.index is None else int(device.index)
    return kind, int(seq_len), int(rank), device.type, device_index, str(dtype)


def _dct_projector(seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).add_(0.5)
    freqs = torch.arange(rank, device=device, dtype=torch.float32).unsqueeze(1)
    projector = torch.cos((math.pi / float(seq_len)) * freqs * positions.unsqueeze(0))
    if rank > 0:
        projector[0].mul_(math.sqrt(1.0 / float(seq_len)))
    if rank > 1:
        projector[1:].mul_(math.sqrt(2.0 / float(seq_len)))
    return projector.to(dtype=dtype).contiguous()


def _hadamard_projector(seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if not _is_power_of_two(seq_len):
        raise ValueError(f"Hadamard token projection requires power-of-two seq_len, got {seq_len}")
    width = int(math.log2(seq_len))
    row_indices = torch.tensor(
        [_hadamard_index_for_sequency(index, width) for index in range(rank)],
        device=device,
        dtype=torch.long,
    )
    columns = torch.arange(seq_len, device=device, dtype=torch.long)
    parity = torch.zeros((rank, seq_len), device=device, dtype=torch.bool)
    for bit in range(width):
        row_bit = torch.bitwise_and(torch.bitwise_right_shift(row_indices[:, None], bit), 1).bool()
        col_bit = torch.bitwise_and(torch.bitwise_right_shift(columns[None, :], bit), 1).bool()
        parity.logical_xor_(row_bit & col_bit)
    projector = torch.where(
        parity,
        torch.tensor(-1.0, device=device, dtype=torch.float32),
        torch.tensor(1.0, device=device, dtype=torch.float32),
    )
    projector.mul_(1.0 / math.sqrt(float(seq_len)))
    return projector.to(dtype=dtype).contiguous()


def _haar_projector(seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if not _is_power_of_two(seq_len):
        raise ValueError(f"Haar token projection requires power-of-two seq_len, got {seq_len}")
    projector = torch.zeros((rank, seq_len), device=device, dtype=torch.float32)
    projector[0].fill_(1.0 / math.sqrt(float(seq_len)))
    row = 1
    block = seq_len
    while row < rank and block > 1:
        half = block // 2
        value = 1.0 / math.sqrt(float(block))
        for start in range(0, seq_len, block):
            if row >= rank:
                break
            projector[row, start : start + half].fill_(value)
            projector[row, start + half : start + block].fill_(-value)
            row += 1
        block //= 2
    return projector.to(dtype=dtype).contiguous()


def _fixed_projector(kind: str, seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    rank = min(max(int(rank), 0), int(seq_len))
    if rank <= 0:
        return torch.empty((0, seq_len), device=device, dtype=dtype)
    key = _projector_cache_key(kind, seq_len, rank, device, dtype)
    cached = _PROJECTOR_CACHE.get(key)
    if cached is not None:
        return cached
    if kind == "dct":
        projector = _dct_projector(seq_len, rank, device, dtype)
    elif kind == "hadamard":
        projector = _hadamard_projector(seq_len, rank, device, dtype)
    elif kind == "haar":
        projector = _haar_projector(seq_len, rank, device, dtype)
    else:
        raise ValueError(f"unknown fixed projector kind {kind!r}")
    _PROJECTOR_CACHE[key] = projector
    return projector


def _piecewise_segment_count(kind: str, seq_len: int, rank: int) -> int | None:
    rank = min(max(int(rank), 0), int(seq_len))
    if kind not in {"hadamard", "haar"} or rank <= 0:
        return None
    if not _is_power_of_two(seq_len):
        return None
    segment_count = min(_next_power_of_two(rank), seq_len)
    if seq_len % segment_count != 0:
        return None
    return segment_count


def _piecewise_projector_coefficients(
    kind: str,
    seq_len: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, int] | None:
    segment_count = _piecewise_segment_count(kind, seq_len, rank)
    if segment_count is None:
        return None
    segment_len = seq_len // segment_count
    projector = _fixed_projector(kind, seq_len, rank, device, dtype)
    coefficients = projector[:, ::segment_len].contiguous()
    return coefficients, segment_len


def _load_triton_piecewise_project() -> Callable[[Tensor, Tensor], Tensor]:
    global _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR, _TRITON_PIECEWISE_PROJECT
    if _TRITON_PIECEWISE_PROJECT is not None:
        return _TRITON_PIECEWISE_PROJECT
    if _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR is not None:
        raise RuntimeError("lowpass token projection Triton kernel is unavailable") from (
            _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR
        )
    try:
        from lowpass_triton import piecewise_project
    except Exception as exc:  # pragma: no cover - depends on optional CUDA/Triton runtime.
        _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR = exc
        raise RuntimeError("lowpass token projection Triton kernel is unavailable") from exc
    _TRITON_PIECEWISE_PROJECT = piecewise_project
    return piecewise_project


def _project_fixed_token_basis(
    kind: str,
    x: Tensor,
    rank: int,
    *,
    hadamard_backend: str,
) -> Tensor:
    if (
        x.is_cuda
        and x.ndim == 3
        and kind in {"hadamard", "haar"}
        and hadamard_backend != "dense"
        and os.environ.get("LOWPASS_DISABLE_TRITON", "0") != "1"
    ):
        try:
            coefficients_and_segment_len = _piecewise_projector_coefficients(
                kind,
                int(x.shape[-2]),
                int(rank),
                x.device,
                x.dtype,
            )
            if coefficients_and_segment_len is not None:
                coefficients, segment_len = coefficients_and_segment_len
                return _load_triton_piecewise_project()(x.contiguous(), coefficients, segment_len=segment_len)
        except Exception:
            if os.environ.get("LOWPASS_REQUIRE_TRITON", "0") == "1":
                raise
    projector = _fixed_projector(kind, int(x.shape[-2]), int(rank), x.device, x.dtype)
    return torch.einsum("rl,nlc->nrc", projector, x)


def _project_token_chunks(
    x: Tensor,
    *,
    projector_kind: str,
    chunk_size: int,
    keep: int,
    hadamard_backend: str,
) -> Tensor:
    if x.ndim != 3:
        raise ValueError(f"expected [batch, seq, hidden], got shape {tuple(x.shape)}")
    batch, seq_len, hidden_dim = x.shape
    chunk_size = int(chunk_size)
    keep = int(keep)
    if seq_len % chunk_size:
        raise ValueError(f"sequence length {seq_len} is not divisible by chunk size {chunk_size}")
    chunk_count = seq_len // chunk_size
    chunks = x.reshape(batch, chunk_count, chunk_size, hidden_dim)
    chunks = chunks.reshape(batch * chunk_count, chunk_size, hidden_dim).contiguous()
    return _project_fixed_token_basis(
        projector_kind,
        chunks,
        keep,
        hadamard_backend=hadamard_backend,
    ).contiguous()


class _LowpassLinearFunction(torch.autograd.Function):
    """Custom-autograd `F.linear` that stores low-pass-projected activations.

    - Forward: exact `F.linear`, but `ctx` only saves `x_hat` (the
      `[batch*chunk_count, keep, hidden]` projection) and `weight`.
    - Backward:
        - `grad_x = grad_output @ weight` is exact.
        - `grad_w = go_hat^T @ x_hat` is approximate (both inputs
          low-pass-projected to `keep/chunk_size` of full length).
        - `grad_bias = grad_output.sum(reduce_dims)` is exact.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        projector_kind: str,
        chunk_size: int,
        keep: int,
        hadamard_backend: str,
    ) -> Tensor:
        y = F.linear(x, weight, bias)
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.has_bias = bias is not None
        ctx.projector_kind = str(projector_kind)
        ctx.chunk_size = int(chunk_size)
        ctx.keep = int(keep)
        ctx.hadamard_backend = str(hadamard_backend)
        x_hat = _project_token_chunks(
            x,
            projector_kind=ctx.projector_kind,
            chunk_size=ctx.chunk_size,
            keep=ctx.keep,
            hadamard_backend=ctx.hadamard_backend,
        )
        ctx.save_for_backward(x_hat.contiguous(), weight)
        return y

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        x_hat, weight = ctx.saved_tensors
        work_dtype = weight.dtype if weight.is_floating_point() else grad_output.dtype
        go = grad_output.to(work_dtype)
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = go.matmul(weight.to(work_dtype)).to(ctx.input_dtype)

        if ctx.needs_input_grad[1]:
            go_hat = _project_token_chunks(
                grad_output,
                projector_kind=ctx.projector_kind,
                chunk_size=ctx.chunk_size,
                keep=ctx.keep,
                hadamard_backend=ctx.hadamard_backend,
            ).to(work_dtype)
            x_work = x_hat.to(work_dtype)
            grad_weight = go_hat.reshape(-1, go_hat.shape[-1]).T.matmul(
                x_work.reshape(-1, x_work.shape[-1])
            ).to(ctx.weight_dtype)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(grad_output.ndim - 1))
            grad_bias = grad_output.sum(dim=reduce_dims)

        return grad_x, grad_weight, grad_bias, None, None, None, None


class LowpassLinear(torch.nn.Module):
    """Drop-in replacement for `nn.Linear` that low-pass-compresses saved activations."""

    def __init__(self, linear: torch.nn.Linear, config: LowpassConfig) -> None:
        super().__init__()
        self.config = config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.register_buffer("lowpass_calls", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("fallback_calls", torch.zeros((), dtype=torch.long), persistent=False)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, config: LowpassConfig) -> "LowpassLinear":
        return cls(linear, config)

    def _can_use_lowpass(self, x: Tensor) -> bool:
        min_hidden_dim = int(self.config.min_hidden_dim)
        return (
            self.config.enabled
            and torch.is_grad_enabled()
            and x.ndim >= 3
            and x.shape[-2] >= self.config.chunk_size
            and x.shape[-2] % self.config.chunk_size == 0
            and int(x.shape[-1]) >= min_hidden_dim
            and int(self.weight.shape[0]) >= min_hidden_dim
            and int(self.weight.shape[1]) >= min_hidden_dim
            and self.weight.requires_grad
        )

    @torch.compiler.disable
    def _apply_lowpass(self, x: Tensor) -> Tensor:
        # Custom autograd Functions break dynamo's graph at every call site.
        # Wrapping with @torch.compiler.disable makes dynamo treat this whole
        # forward as a single opaque op: the surrounding transformer block
        # stays fully compiled and only this one boundary is eager. The
        # Hadamard projection itself is cheap (~1 ms per step total); the
        # 38 % slowdown observed without this wrapper came entirely from
        # fragmenting the compiled graph into ~144 sub-segments.
        return _LowpassLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.config.projector_kind,
            self.config.chunk_size,
            self.config.keep,
            self.config.hadamard_backend,
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self._can_use_lowpass(x):
            if not _torch_is_compiling():
                self.fallback_calls += 1
            return F.linear(x, self.weight, self.bias)
        if not _torch_is_compiling():
            self.lowpass_calls += 1
        return self._apply_lowpass(x)


def replace_linear_with_lowpass(
    model: torch.nn.Module,
    config: LowpassConfig,
    module_filter: Callable[[str, torch.nn.Linear], bool] | None = None,
) -> list[str]:
    replaced: list[str] = []

    def visit(parent: torch.nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LowpassLinear):
                continue
            if isinstance(child, torch.nn.Linear):
                if module_filter is None or module_filter(full_name, child):
                    setattr(parent, child_name, LowpassLinear.from_linear(child, config))
                    replaced.append(full_name)
                continue
            visit(child, full_name)

    visit(model, "")
    return replaced


_MLP_NAME_PARTS = ("mlp", "gate_proj", "up_proj", "down_proj", "feed_forward", "ffn")


def mlp_module_filter(name: str, _module: torch.nn.Linear) -> bool:
    """Module-name filter for `replace_linear_with_lowpass(target='mlp')`.

    Matches any module whose dotted name contains an MLP-related token —
    `mlp`, `gate_proj`, `up_proj`, `down_proj`, `feed_forward`, or `ffn`.
    Covers the Qwen, Llama, and Gemma MLP naming conventions.
    """
    lowered = name.lower()
    return any(part in lowered for part in _MLP_NAME_PARTS)


def make_module_filter(target: str) -> Callable[[str, torch.nn.Linear], bool] | None:
    """Build a module filter for the `--lowpass-target-filter` CLI choice."""
    normalized = str(target).lower().replace("-", "_")
    if normalized in {"mlp"}:
        return mlp_module_filter
    if normalized in {"all", "every", "any"}:
        return None  # `None` means "replace every nn.Linear".
    if normalized in {"none", "off"}:
        return lambda _name, _module: False
    raise ValueError(f"unknown lowpass target filter {target!r}")
