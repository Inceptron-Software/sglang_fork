from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.models import llama4


@pytest.mark.parametrize("cuda_path", [False, True])
@pytest.mark.parametrize("inplace_rope", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_attention_consumes_rope_results_without_cuda_repacking(
    cuda_path, inplace_rope, normalize
):
    def norm(x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype)

    def rope(positions, q, k):
        if inplace_rope:
            return q.add_(1), k.sub_(2)
        return q + 1, k - 2

    rope.is_torch_compile = not inplace_rope

    def qk_norm(q, k, q_norm, k_norm, head_dim):
        return (
            q_norm(q.reshape(-1, head_dim)).reshape(q.shape),
            k_norm(k.reshape(-1, head_dim)).reshape(k.shape),
        )

    def attention(q, k, v, batch):
        return q + k.repeat(1, 2) + v.repeat(1, 2)

    block = SimpleNamespace(
        q_size=8,
        kv_size=4,
        head_dim=4,
        qkv_proj=lambda x: (x, None),
        rotary_emb=rope,
        qk_norm=norm if normalize else None,
        attn_temperature_tuning=False,
        attn=attention,
        o_proj=lambda x: (x, None),
    )
    values = torch.arange(48, dtype=torch.bfloat16).reshape(3, 16)
    expected_q = values[:, :8] + 1
    expected_k = values[:, 8:12] - 2
    if normalize:
        expected_q, expected_k = qk_norm(expected_q, expected_k, norm, norm, 4)
    expected = attention(expected_q, expected_k, values[:, 12:], None)

    with (
        patch.object(llama4, "_is_cuda", cuda_path),
        patch.object(llama4, "_is_npu", not cuda_path),
        patch.object(llama4, "apply_qk_norm", side_effect=qk_norm),
        patch.object(torch, "cat", wraps=torch.cat) as cat,
    ):
        actual = llama4.Llama4Attention.forward(block, torch.arange(3), values, None)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if cuda_path:
        assert cat.call_count == 0
