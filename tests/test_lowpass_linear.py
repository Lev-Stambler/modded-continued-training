"""Unit tests for the Hadamard low-pass `nn.Linear` replacement.

Run locally (`uv run python -m unittest tests.test_lowpass_linear -v`) if
torch is installed. Otherwise the tests skip; the Modal smoke run is the
real verification.
"""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn as nn

    from lowpass import (
        LowpassConfig,
        LowpassLinear,
        make_module_filter,
        mlp_module_filter,
        replace_linear_with_lowpass,
    )
except ImportError as exc:  # pragma: no cover - exercised when torch is absent.
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None


def _skip_if_no_torch(test_fn):
    def wrapper(self, *args, **kwargs):
        if _IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"torch unavailable in this env: {_IMPORT_ERROR}")
        return test_fn(self, *args, **kwargs)

    wrapper.__name__ = test_fn.__name__
    return wrapper


def _make_linear_and_lowpass(in_features: int, out_features: int, config: LowpassConfig):
    torch.manual_seed(1337)
    baseline = nn.Linear(in_features, out_features, bias=True)
    lp = LowpassLinear.from_linear(baseline, config)
    return baseline, lp


class ForwardParityTest(unittest.TestCase):
    @_skip_if_no_torch
    def test_forward_matches_linear(self):
        config = LowpassConfig(projector_kind="hadamard", keep=8, min_hidden_dim=0)
        baseline, lp = _make_linear_and_lowpass(256, 384, config)
        x = torch.randn(2, 128, 256, requires_grad=False)
        with torch.no_grad():
            y_ref = baseline(x)
            y_lp = lp(x)
        self.assertTrue(torch.allclose(y_ref, y_lp, atol=1e-6, rtol=1e-6))


class InputGradExactTest(unittest.TestCase):
    @_skip_if_no_torch
    def test_input_grad_exact(self):
        config = LowpassConfig(projector_kind="hadamard", keep=8, min_hidden_dim=0)
        baseline, lp = _make_linear_and_lowpass(256, 384, config)

        x = torch.randn(2, 128, 256)
        x_ref = x.clone().detach().requires_grad_(True)
        x_lp = x.clone().detach().requires_grad_(True)
        y_ref = baseline(x_ref)
        y_lp = lp(x_lp)
        grad_out = torch.randn_like(y_ref)
        y_ref.backward(grad_out)
        y_lp.backward(grad_out)
        self.assertIsNotNone(x_ref.grad)
        self.assertIsNotNone(x_lp.grad)
        # grad_x = grad_output @ weight is computed exactly in both paths.
        self.assertTrue(torch.allclose(x_ref.grad, x_lp.grad, atol=1e-5, rtol=1e-5))


class ParamGradToleranceTest(unittest.TestCase):
    @_skip_if_no_torch
    def test_param_grad_within_tolerance(self):
        # 32/64 keep ratio: param grad is approximate but should stay close to
        # the exact gradient. We assert relative RMS error is bounded.
        config = LowpassConfig(projector_kind="hadamard", keep=8, min_hidden_dim=0)
        baseline, lp = _make_linear_and_lowpass(256, 384, config)

        x = torch.randn(2, 128, 256, requires_grad=True)
        y_ref = baseline(x)
        grad_out = torch.randn_like(y_ref)
        baseline.zero_grad()
        y_ref.backward(grad_out)
        ref_grad_w = baseline.weight.grad.clone()

        x2 = x.detach().clone().requires_grad_(True)
        y_lp = lp(x2)
        lp.weight.grad = None
        y_lp.backward(grad_out)
        lp_grad_w = lp.weight.grad.clone()

        # Param grads from low-pass projection should be approximate but
        # well-correlated. With keep=chunk_size/2 we expect noticeable but
        # bounded error — Adam's EMA smooths these out in training.
        rms_ref = ref_grad_w.pow(2).mean().sqrt()
        rms_err = (lp_grad_w - ref_grad_w).pow(2).mean().sqrt()
        rel_err = (rms_err / rms_ref).item()
        # Generous bound: 50% RMS-relative — proves the gradient direction
        # carries signal without asserting it's near-exact.
        self.assertLess(rel_err, 0.50, msg=f"param grad RMS-rel error {rel_err:.3f} > 0.50")


class ReplaceWalkerTest(unittest.TestCase):
    @_skip_if_no_torch
    def test_replace_walks_model_mlp_filter(self):
        # Tiny mock of a transformer-ish layer with named MLP sub-linears.
        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(128, 256, bias=False)
                self.up_proj = nn.Linear(128, 256, bias=False)
                self.down_proj = nn.Linear(256, 128, bias=False)

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        class TinyAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(128, 128, bias=False)
                self.k_proj = nn.Linear(128, 128, bias=False)
                self.v_proj = nn.Linear(128, 128, bias=False)

            def forward(self, x):
                return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)

        class TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = TinyMLP()
                self.attn = TinyAttn()

            def forward(self, x):
                return self.mlp(x) + self.attn(x)

        config = LowpassConfig(projector_kind="dct", keep=8, min_hidden_dim=0)
        model = TinyBlock()
        replaced = replace_linear_with_lowpass(model, config, mlp_module_filter)
        # Should hit gate_proj/up_proj/down_proj (3 MLP linears), not Q/K/V.
        self.assertEqual(len(replaced), 3)
        for name in replaced:
            self.assertTrue("mlp" in name)
        # Verify attn Linears are untouched.
        self.assertIsInstance(model.attn.q_proj, nn.Linear)
        self.assertNotIsInstance(model.attn.q_proj, LowpassLinear)
        self.assertIsInstance(model.mlp.gate_proj, LowpassLinear)

    @_skip_if_no_torch
    def test_make_module_filter_choices(self):
        self.assertIs(make_module_filter("mlp"), mlp_module_filter)
        self.assertIsNone(make_module_filter("all"))
        none_filter = make_module_filter("none")
        self.assertFalse(none_filter("mlp.gate_proj", nn.Linear(8, 8)))
        with self.assertRaises(ValueError):
            make_module_filter("nonsense")


if __name__ == "__main__":
    unittest.main()
