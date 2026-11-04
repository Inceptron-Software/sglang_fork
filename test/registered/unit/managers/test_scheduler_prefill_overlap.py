"""Overlap decisions must use the previous forward's immutable batch mode."""

import copy
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.test_utils import maybe_stub_sgl_kernel
from sglang.test.ci.ci_register import register_cpu_ci

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _run_overlap_iteration(require_mlp_sync, disable_prefill_overlap):
    def make_batch(name, mode):
        batch = SimpleNamespace(
            name=name,
            forward_mode=mode,
            is_extend_in_batch=True,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        batch.copy = lambda: copy.copy(batch)
        return batch

    first = make_batch("first", ForwardMode.EXTEND)
    second = make_batch("second", ForwardMode.MIXED)
    events = []
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.gracefully_exit = False
    scheduler.request_receiver = SimpleNamespace(recv_requests=lambda: [])
    scheduler.process_input_requests = MagicMock()
    scheduler._engine_paused = False
    scheduler.running_batch = None
    scheduler.last_batch = None
    scheduler.require_mlp_sync = require_mlp_sync
    scheduler.enable_unified_memory = False
    scheduler.is_generation = False
    scheduler._apply_war_barrier = MagicMock()

    def plan(*, running_batch, last_batch):
        if last_batch is None:
            return SimpleNamespace(batch_to_run=first, running_batch=None)
        # get_next_batch_to_run reuses the previous prefill as running_batch,
        # then prepare_for_decode mutates its mode before the next mixed batch.
        assert last_batch is first
        first.forward_mode = ForwardMode.DECODE
        first.is_extend_in_batch = False
        return SimpleNamespace(batch_to_run=second, running_batch=first)

    def run(batch):
        events.append("run:" + batch.name)
        if batch is second:
            scheduler.gracefully_exit = True
        return batch.name

    scheduler.get_next_batch_to_run = plan
    scheduler.run_batch = run
    scheduler.process_batch_result = lambda batch, result: events.append(
        "process:" + result
    )
    with (
        envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(
            disable_prefill_overlap
        ),
        envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(0),
    ):
        scheduler.event_loop_overlap()

    return events


class TestPrefillOverlap(unittest.TestCase):
    def test_prefill_result_precedes_next_forward_after_batch_reuse(self):
        for synchronized, disabled in itertools.product((False, True), repeat=2):
            with self.subTest(require_mlp_sync=synchronized, overlap_disabled=disabled):
                expected = (
                    ["run:first", "process:first", "run:second"]
                    if disabled
                    else ["run:first", "run:second", "process:first"]
                )
                self.assertEqual(
                    _run_overlap_iteration(synchronized, disabled), expected
                )


if __name__ == "__main__":
    unittest.main()
