from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention import flashattention_backend as fa


@pytest.mark.parametrize("provided_output", [False, True])
def test_local_decode_uses_provided_output(provided_output):
    cache = torch.zeros((8, 1, 4), dtype=torch.bfloat16)
    local_metadata = SimpleNamespace(
        local_block_table=torch.zeros((2, 4), dtype=torch.int32),
        local_seqused_k=torch.ones(2, dtype=torch.int32),
        local_query_start_loc=torch.arange(3, dtype=torch.int32),
        local_max_query_len=1,
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(local_attn_metadata=local_metadata),
        has_local_attention=True,
        attention_chunk_size=8192,
        topk=1,
        use_mla=False,
        page_size=1,
        kv_cache_dtype_str="auto",
        kv_cache_is_mxfp8=False,
        fa_impl_ver=3,
        num_splits=1,
        _mxfp8_sf_kwargs=lambda *args: {},
        token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda _: (cache, cache)),
    )
    layer = SimpleNamespace(
        use_irope=True,
        sliding_window_size=8192,
        is_cross_attention=False,
        attn_type=fa.AttentionType.DECODER,
        layer_id=0,
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=4,
        v_head_dim=4,
        scaling=0.5,
        logit_cap=0,
    )
    query = torch.zeros((2, 8), dtype=torch.bfloat16)
    output = torch.empty_like(query) if provided_output else None
    batch = SimpleNamespace(spec_info=None, _attn_output=output)

    def attention(**kwargs):
        destination = kwargs.get("out")
        if destination is None:
            destination = torch.empty((2, 2, 4), dtype=query.dtype)
        return destination.fill_(5)

    with patch.object(
        fa, "flash_attn_with_kvcache", side_effect=attention, create=True
    ):
        actual = fa.FlashAttentionBackend.forward_decode(
            backend, query, None, None, layer, batch, save_kv_cache=False
        )

    torch.testing.assert_close(actual, torch.full_like(query, 5))
    if provided_output:
        assert actual.data_ptr() == output.data_ptr()
        torch.testing.assert_close(output, actual)
