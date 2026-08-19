from types import SimpleNamespace

from sglang.srt.compilation.backend import _get_cache_tag
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
