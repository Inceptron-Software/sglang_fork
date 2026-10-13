import pytest
import torch

from sglang.srt.models import mllama4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class ExpertLoader:
    moe_tp_size = 1
    moe_ep_size = 1
    num_local_experts = 2
    _num_global_routed = 2
    _has_fused_shared = False
    use_padded_loading = False
    use_presharded_weights = False
    use_triton_kernels = False
    use_flashinfer_trtllm_moe = False

    def __init__(self):
        self.quant_method = mllama4.ModelOptFp8MoEMethod.__new__(
            mllama4.ModelOptFp8MoEMethod
        )
        self.calls = []

    def weight_loader(self, param, weight, name, *, shard_id, expert_id):
        self.calls.append((shard_id, expert_id))
        destination = param.data[expert_id]
        if shard_id != "w2":
            half = destination.shape[0] // 2
            destination = destination[:half] if shard_id == "w1" else destination[half:]
        destination.copy_(weight)


def make_weight_case(projection, device):
    name = f"language_model.layers.0.feed_forward.experts.{projection}"
    model = mllama4.Llama4ForConditionalGeneration.__new__(
        mllama4.Llama4ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    parameter_name, _, _ = model._transform_expert_name(name, is_weight=True)
    output_dim = 8 if projection == "gate_up_proj" else 4
    weight = (torch.arange(2 * 3 * output_dim).reshape(2, 3, output_dim) / 16).to(
        torch.float8_e4m3fn
    )
    param = torch.nn.Parameter(
        torch.empty(2, output_dim, 3, device=device, dtype=weight.dtype),
        requires_grad=False,
    )
    loader = ExpertLoader()
    param.weight_loader = loader.weight_loader
    return model, name, parameter_name, weight, param, loader


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_cpu_destination_preserves_per_expert_loading(monkeypatch, projection):
    model, name, parameter_name, weight, param, loader = make_weight_case(
        projection, "cpu"
    )
    monkeypatch.setattr(mllama4, "get_global_expert_location_metadata", lambda: None)
    loaded_params = set()

    assert model._handle_expert_weight_params(
        name, weight, {parameter_name: param}, 2, loaded_params
    )

    assert len(loader.calls) == (4 if projection == "gate_up_proj" else 2)
    assert loaded_params == {parameter_name}
    torch.testing.assert_close(
        param.float(), weight.transpose(1, 2).float(), rtol=0, atol=0
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a reserved CUDA GPU"
)
@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_fp8_experts_load_in_bulk_on_cuda(monkeypatch, projection):
    model, name, parameter_name, weight, param, loader = make_weight_case(
        projection, "cuda"
    )
    monkeypatch.setattr(mllama4, "get_global_expert_location_metadata", lambda: None)
    loaded_params = set()

    assert model._handle_expert_weight_params(
        name, weight, {parameter_name: param}, 2, loaded_params
    )

    assert loader.calls == []
    assert loaded_params == {parameter_name}
    torch.testing.assert_close(
        param.float().cpu(), weight.transpose(1, 2).float(), rtol=0, atol=0
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a reserved CUDA GPU"
)
@pytest.mark.parametrize(
    "attribute,value",
    [
        ("moe_tp_size", 2),
        ("moe_ep_size", 2),
        ("_has_fused_shared", True),
        ("use_padded_loading", True),
        ("use_presharded_weights", True),
        ("use_triton_kernels", True),
        ("use_flashinfer_trtllm_moe", True),
        ("quant_method", None),
        ("expert_location", object()),
    ],
)
def test_unsupported_bulk_configuration_uses_weight_loader(
    monkeypatch, attribute, value
):
    model, name, parameter_name, weight, param, loader = make_weight_case(
        "gate_up_proj", "cuda"
    )
    if attribute == "expert_location":
        monkeypatch.setattr(
            mllama4, "get_global_expert_location_metadata", lambda: value
        )
    else:
        monkeypatch.setattr(
            mllama4, "get_global_expert_location_metadata", lambda: None
        )
        setattr(loader, attribute, value)
    loaded_params = set()

    assert model._handle_expert_weight_params(
        name, weight, {parameter_name: param}, 2, loaded_params
    )

    assert len(loader.calls) == 4
    assert loaded_params == {parameter_name}
    torch.testing.assert_close(
        param.float().cpu(), weight.transpose(1, 2).float(), rtol=0, atol=0
    )
