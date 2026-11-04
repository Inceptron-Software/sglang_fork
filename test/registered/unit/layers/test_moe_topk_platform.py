from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.layers.moe import topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMoeTopkPlatformHook(CustomTestCase):
    def test_platform_fallback_precedes_builtin_implementation(self):
        platform_impl = Mock(return_value="platform-result")
        platform = SimpleNamespace(
            get_biased_grouped_topk_impl=Mock(return_value=platform_impl)
        )

        with patch.object(topk, "current_platform", platform):
            result = topk._biased_grouped_topk_fallback("input", topk=2)

        self.assertEqual(result, "platform-result")
        platform_impl.assert_called_once_with("input", topk=2)

    def test_none_platform_fallback_preserves_builtin_implementation(self):
        platform = SimpleNamespace(get_biased_grouped_topk_impl=Mock(return_value=None))
        builtin = Mock(return_value="builtin-result")

        with (
            patch.object(topk, "current_platform", platform),
            patch.object(topk, "biased_grouped_topk_impl", builtin),
        ):
            result = topk._biased_grouped_topk_fallback("input", topk=2)

        self.assertEqual(result, "builtin-result")
        builtin.assert_called_once_with("input", topk=2)
