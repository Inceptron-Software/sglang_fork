import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as runner_module
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend import utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _backend_runner():
    server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL),
        ),
        enable_memory_saver=False,
        debug_cuda_graph=False,
    )
    model_runner = SimpleNamespace(server_args=server_args, device="cuda")
    return SimpleNamespace(model_runner=model_runner)


class TestDecodeBackendExtension(CustomTestCase):
    def test_out_of_tree_backend_precedes_builtin_resolution(self):
        class PlatformBackend:
            def __init__(self, runner):
                self.runner = runner

        runner = _backend_runner()
        platform = SimpleNamespace(
            get_cuda_graph_backend_cls=Mock(return_value=PlatformBackend)
        )

        with patch.object(utils, "current_platform", platform):
            backend = utils.resolve_decode_backend(runner)

        self.assertIsInstance(backend, PlatformBackend)
        self.assertIs(backend.runner, runner)
        platform.get_cuda_graph_backend_cls.assert_called_once_with(
            Phase.DECODE, Backend.FULL
        )

    def test_none_override_preserves_builtin_full_backend(self):
        class BuiltinBackend:
            def __init__(self, runner, *, enable_memory_saver):
                self.runner = runner
                self.enable_memory_saver = enable_memory_saver

        runner = _backend_runner()
        platform = SimpleNamespace(get_cuda_graph_backend_cls=Mock(return_value=None))
        with (
            patch.object(utils, "current_platform", platform),
            patch.object(utils, "FullCudaGraphBackend", BuiltinBackend),
        ):
            backend = utils.resolve_decode_backend(runner)

        self.assertIsInstance(backend, BuiltinBackend)
        self.assertIs(backend.runner, runner)
        self.assertFalse(backend.enable_memory_saver)

    def test_capture_hands_forward_batch_to_backend(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        forward_batch = SimpleNamespace(
            input_ids=torch.zeros(2, dtype=torch.int64),
            positions=torch.zeros(2, dtype=torch.int64),
            lora_ids=None,
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
            global_dp_buffer_len=None,
            global_num_tokens_cpu=None,
        )
        attn_backend = SimpleNamespace(
            init_forward_metadata_out_graph=Mock(),
            on_after_cuda_graph_warmup=Mock(),
        )
        runner.captured_req_width = 1
        runner.ragged_verify_mode = False
        runner.pp_size = 1
        runner.capture_prepare = Mock(return_value=(forward_batch, attn_backend, None))
        runner.tbo_plugin = SimpleNamespace(capture_one_batch_size=Mock())
        runner.deepep_adapter = SimpleNamespace(capture=Mock())
        runner.backend = SimpleNamespace(capture_one=Mock())
        runner._make_graph_key = Mock(return_value="shape-key")
        runner.model_runner = SimpleNamespace(
            server_args=SimpleNamespace(debug_cuda_graph=False),
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: False),
            is_draft_worker=False,
            capture_tail_hooks=[],
            canary_manager=None,
            attn_backend=attn_backend,
        )

        with (
            patch.object(
                runner_module,
                "forward_context",
                return_value=contextlib.nullcontext(),
            ),
            patch.object(
                runner_module,
                "maybe_flashinfer_autotune_speculative_draft",
            ),
        ):
            runner.capture_one_shape(2, Mock())

        self.assertIs(
            runner.backend.capture_one.call_args.kwargs["capture_inputs"],
            forward_batch,
        )


if __name__ == "__main__":
    unittest.main()
