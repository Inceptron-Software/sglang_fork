from types import SimpleNamespace

import pytest

from sglang.srt.layers.utils import multi_platform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeTopK(multi_platform.MultiPlatformOp):
    def forward_native(self):
        return "native"


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 4])
def test_topk_compile_preserves_registered_platform_dispatch(
    monkeypatch, registered, num_tokens
):
    key = "test-oot"
    platform = SimpleNamespace(
        is_out_of_tree=lambda: True, get_dispatch_key_name=lambda: key
    )
    monkeypatch.setattr(multi_platform, "current_platform", platform)
    monkeypatch.setattr(multi_platform.MultiPlatformOp, "_oot_forward_registry", {})

    def platform_forward(self):
        return "platform"

    if registered:
        multi_platform.MultiPlatformOp.register_oot_forward(
            FakeTopK, platform_forward, key
        )
    op = FakeTopK()
    original = op._forward_method

    op.enter_torch_compile(num_tokens)
    op.enter_torch_compile(num_tokens)

    assert op() == ("platform" if registered else "native")
    assert op._original_forward_method == original
    op.leave_torch_compile()
    assert op._forward_method == original
    assert not op.is_torch_compile
