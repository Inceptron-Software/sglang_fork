from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers import radix_linear_attention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def attention():
    layer = radix_linear_attention.RadixLinearAttention.__new__(
        radix_linear_attention.RadixLinearAttention
    )
    torch.nn.Module.__init__(layer)
    layer.num_v_heads = 2
    layer.head_v_dim = 4
    layer.layer_id = 7
    return layer


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.EXTEND])
@pytest.mark.parametrize("breakable", [False, True])
@pytest.mark.parametrize("num_tokens", [8, 4, 2])
def test_piecewise_attention_uses_split_op(
    monkeypatch, attention, mode, breakable, num_tokens
):
    batch = SimpleNamespace(forward_mode=mode)
    mixed_qkv = torch.ones(num_tokens, 8)
    a = torch.ones(num_tokens, 2)
    b = torch.ones(num_tokens, 2)
    split_op = Mock()
    other_split_op = Mock()
    monkeypatch.setattr(
        radix_linear_attention, "get_tc_piecewise_forward_context", lambda: object()
    )
    monkeypatch.setattr(
        radix_linear_attention, "is_in_breakable_cuda_graph", lambda: breakable
    )
    monkeypatch.setattr(
        radix_linear_attention,
        "unified_linear_attention_with_output",
        other_split_op if breakable else split_op,
    )
    monkeypatch.setattr(
        radix_linear_attention,
        "bcg_unified_linear_attention_with_output",
        split_op if breakable else other_split_op,
    )

    output = attention(batch, mixed_qkv, a, b)

    assert output.shape == (1, num_tokens, 2, 4)
    assert output.dtype == mixed_qkv.dtype
    assert output.device == mixed_qkv.device
    split_op.assert_called_once_with(mixed_qkv, a, b, output, 7)
    other_split_op.assert_not_called()


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.EXTEND])
def test_without_piecewise_context_uses_attention_backend(monkeypatch, attention, mode):
    batch = SimpleNamespace(forward_mode=mode)
    mixed_qkv = torch.ones(3, 8)
    a = b = torch.ones(3, 2)
    backend = SimpleNamespace(forward=Mock(return_value="backend-output"))
    monkeypatch.setattr(
        radix_linear_attention, "get_tc_piecewise_forward_context", lambda: None
    )
    monkeypatch.setattr(radix_linear_attention, "get_attn_backend", lambda: backend)

    assert attention(batch, mixed_qkv, a, b) == "backend-output"
    backend.forward.assert_called_once_with(
        layer=attention, forward_batch=batch, mixed_qkv=mixed_qkv, a=a, b=b
    )
