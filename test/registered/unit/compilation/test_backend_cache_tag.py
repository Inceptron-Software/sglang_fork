from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.compilation import backend
from sglang.srt.compilation.backend import _get_cache_tag
from sglang.srt.compilation.compiler_interface import EagerAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBackendCacheTag(CustomTestCase):
    def test_config_can_override_cache_tag(self):
        self.assertEqual(
            _get_cache_tag(SimpleNamespace(cache_tag="prefill")),
            "prefill",
        )

    def test_config_without_cache_tag_preserves_builtin_default(self):
        self.assertEqual(_get_cache_tag(SimpleNamespace()), "backbone")

    def test_platform_can_supply_piecewise_compiler(self):
        compiler = object()
        platform = SimpleNamespace(get_piecewise_compiler=Mock(return_value=compiler))
        config = SimpleNamespace(compiler="out_of_tree")

        with patch.object(backend, "current_platform", platform):
            result = backend.make_compiler(config)

        self.assertIs(result, compiler)
        platform.get_piecewise_compiler.assert_called_once_with(config)

    def test_none_platform_compiler_preserves_builtin_resolution(self):
        platform = SimpleNamespace(get_piecewise_compiler=Mock(return_value=None))

        with patch.object(backend, "current_platform", platform):
            result = backend.make_compiler(SimpleNamespace(compiler="eager"))

        self.assertIsInstance(result, EagerAdapter)
