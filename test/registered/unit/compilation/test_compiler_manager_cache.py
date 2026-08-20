import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.compilation import backend
from sglang.srt.compilation.backend import CompilerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCompilerManagerCache(CustomTestCase):
    @staticmethod
    def make_compiler(compiled_graph, handle):
        compiler = SimpleNamespace(
            name="test_compiler",
            initialize_cache=Mock(),
            compile=Mock(return_value=(compiled_graph, handle)),
            load=Mock(return_value=compiled_graph),
        )
        return compiler

    @staticmethod
    def make_manager(compiler):
        with patch.object(backend, "make_compiler", return_value=compiler):
            return CompilerManager(SimpleNamespace())

    def test_cache_miss_compiles_and_persists_handle(self):
        compiled_graph = Mock()
        compiler = self.make_compiler(compiled_graph, "artifact-handle")
        manager = self.make_manager(compiler)
        graph = object()

        with tempfile.TemporaryDirectory() as cache_dir:
            manager.initialize_cache(cache_dir)
            result = manager.compile(graph, [], {}, runtime_shape=None)
            manager.save_to_file()

            self.assertIs(result, compiled_graph)
            compiler.load.assert_not_called()
            compiler.compile.assert_called_once_with(
                graph,
                [],
                {},
                None,
                "artifact_shape_None_subgraph_0",
            )
            self.assertTrue(manager.is_cache_updated)

    def test_persisted_handle_is_loaded_without_recompiling(self):
        for runtime_shape in (None, 8):
            with self.subTest(
                runtime_shape=runtime_shape
            ), tempfile.TemporaryDirectory() as cache_dir:
                graph = object()
                inputs = [object()]
                first_callable = Mock()
                first_compiler = self.make_compiler(
                    first_callable, "persisted-artifact-handle"
                )
                first_manager = self.make_manager(first_compiler)
                first_manager.initialize_cache(cache_dir)
                first_manager.compile(
                    graph, inputs, {"option": True}, runtime_shape=runtime_shape
                )
                first_manager.save_to_file()

                loaded_callable = Mock()
                second_compiler = self.make_compiler(loaded_callable, None)
                second_manager = self.make_manager(second_compiler)
                second_manager.initialize_cache(cache_dir)

                result = second_manager.compile(
                    graph, inputs, {"option": True}, runtime_shape=runtime_shape
                )

                self.assertIs(result, loaded_callable)
                second_compiler.compile.assert_not_called()
                second_compiler.load.assert_called_once_with(
                    "persisted-artifact-handle",
                    graph,
                    inputs,
                    0,
                    runtime_shape,
                )
                self.assertFalse(second_manager.is_cache_updated)
