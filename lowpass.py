"""Whole-sequence low-pass activation compression for `nn.Linear`.

Ported from `/home/lev/hermes-home/Research/AdaptiveRoundingSimp-qad-clean`'s
`instant.py` reference design. Each wrapped `nn.Linear` saves a
sequence-axis-projected `x_hat = projector @ x` for backward (shape
`[..., keep, hidden]` instead of `[..., seq, hidden]`), giving
seq_len/keep× memory savings on the saved activation.

Backward:
- `grad_x = grad_output @ weight` is **exact** (projecting it hurt
  convergence in the AdaptiveRoundingSimp experiments).
- `grad_w = einsum("nro,nri->oi", go_hat, x_hat)` is computed in the
  projected space using `go_hat = projector @ grad_output`. This is a
  low-rank approximation, accurate when the activation and grad signals
  along the sequence axis lie close to the projector's row space.

Supported projectors: `dct`, `hadamard`, `haar` (the latter two require
a power-of-two sequence length).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


_PROJECTOR_CACHE: dict[tuple[str, int, int, str, int, str], Tensor] = {}


@dataclass(frozen=True)
class LowpassConfig:
    projector_kind: str = "dct"
    keep: int = 8
    min_hidden_dim: int = 8000
    max_hidden_dim: int = 16000
    enabled: bool = True

    def __post_init__(self) -> None:
        projector_kind = str(self.projector_kind).lower()
        object.__setattr__(self, "projector_kind", projector_kind)
        if projector_kind not in {"dct", "hadamard", "haar"}:
            raise ValueError(f"unknown projector_kind {self.projector_kind!r}")
        if self.keep < 1:
            raise ValueError("keep must be >= 1")
        if self.min_hidden_dim < 0:
            raise ValueError("min_hidden_dim must be non-negative")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


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


class _LowpassLinearFunction(torch.autograd.Function):
    """Whole-sequence low-rank projection (matches AdaptiveRoundingSimp `instant.py`).

    Forward: exact `y = F.linear(x, w, b)`; saves
    `x_hat = projector @ x` of shape `[..., keep, hidden]` instead of
    the full `[..., seq, hidden]` input.

    Backward:
      grad_x = grad_output @ weight                  (exact)
      grad_w = einsum("nro,nri->oi", go_hat, x_hat)  (low-rank approx)
      grad_b = grad_output.sum(reduce_dims)          (exact)
    """

    @staticmethod
    def forward(ctx, x, weight, bias, projector_kind, keep):
        y = F.linear(x, weight, bias)
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.has_bias = bias is not None
        ctx.projector_kind = str(projector_kind)
        ctx.keep = int(keep)
        x_work = x if x.ndim >= 3 else x.unsqueeze(0)
        seq_len_actual = int(x_work.shape[-2])
        projector = _fixed_projector(
            ctx.projector_kind, seq_len_actual, ctx.keep, x_work.device, x_work.dtype
        )
        x_hat = torch.einsum("rl,...lc->...rc", projector, x_work)
        ctx.save_for_backward(x_hat.contiguous(), weight)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, weight = ctx.saved_tensors
        work_dtype = x_hat.dtype
        go = grad_output.to(work_dtype)
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = go.matmul(weight.to(work_dtype)).to(ctx.input_dtype)

        if ctx.needs_input_grad[1]:
            go_work = go if go.ndim >= 3 else go.unsqueeze(0)
            seq_len_actual = int(go_work.shape[-2])
            projector = _fixed_projector(
                ctx.projector_kind, seq_len_actual, ctx.keep, go.device, work_dtype
            )
            go_hat = torch.einsum("rl,...lo->...ro", projector, go_work)
            grad_weight = torch.einsum(
                "nro,nri->oi",
                go_hat.reshape(-1, go_hat.shape[-2], go_hat.shape[-1]),
                x_hat.reshape(-1, x_hat.shape[-2], x_hat.shape[-1]),
            ).to(ctx.weight_dtype)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(grad_output.ndim - 1))
            grad_bias = grad_output.sum(dim=reduce_dims)

        return grad_x, grad_weight, grad_bias, None, None


class LowpassLinear(torch.nn.Module):
    """Drop-in `nn.Linear` replacement that compresses saved activations."""

    def __init__(self, linear: torch.nn.Linear, config: LowpassConfig) -> None:
        super().__init__()
        self.config = config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, config: LowpassConfig) -> "LowpassLinear":
        return cls(linear, config)

    def _can_use_lowpass(self, x: Tensor) -> bool:
        if not self.config.enabled or not torch.is_grad_enabled():
            return False
        keep = int(self.config.keep)
        min_hidden_dim = int(self.config.min_hidden_dim)
        max_hidden_dim = int(self.config.max_hidden_dim)
        if x.ndim < 2:
            return False
        token_count = int(x.shape[-2])
        hidden = int(x.shape[-1])
        if token_count < keep:
            return False
        if self.config.projector_kind in {"hadamard", "haar"} and not _is_power_of_two(token_count):
            return False
        if hidden < min_hidden_dim:
            return False
        if max_hidden_dim > 0 and hidden > max_hidden_dim:
            return False
        return self.weight.requires_grad

    def forward(self, x: Tensor) -> Tensor:
        if not self._can_use_lowpass(x):
            return F.linear(x, self.weight, self.bias)
        return _LowpassLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.config.projector_kind,
            self.config.keep,
        )


_MLP_NAME_PARTS = ("mlp", "gate_proj", "up_proj", "down_proj", "feed_forward", "ffn")


def mlp_module_filter(name: str, _module: torch.nn.Linear) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in _MLP_NAME_PARTS)


def make_module_filter(target: str) -> Callable[[str, torch.nn.Linear], bool] | None:
    normalized = str(target).lower().replace("-", "_")
    if normalized == "mlp":
        return mlp_module_filter
    if normalized in {"all", "every", "any"}:
        return None  # None means replace every nn.Linear
    if normalized in {"none", "off"}:
        return lambda _name, _module: False
    raise ValueError(f"unknown lowpass target filter {target!r}")


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
