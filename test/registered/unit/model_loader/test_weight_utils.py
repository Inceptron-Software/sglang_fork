"""Unit tests for srt/model_loader/weight_utils.py shard-index consistency."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_loader import weight_utils
from sglang.srt.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    get_quant_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

INDEX_NAME = "model.safetensors.index.json"


def _write_index(folder, weight_map):
    with open(os.path.join(folder, INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def _touch(folder, name):
    path = os.path.join(folder, name)
    open(path, "w").close()
    return path


class TestFilterDuplicateSafetensorsFiles(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_shard_raises(self):
        # Index lists two shards, only one on disk (interrupted download).
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        present = _touch(self.folder, "model-00001-of-00002.safetensors")

        with self.assertRaises(RuntimeError) as cm:
            filter_duplicate_safetensors_files(
                hf_weights_files=[present],
                hf_folder=self.folder,
                index_file=INDEX_NAME,
            )
        self.assertIn("model-00002-of-00002.safetensors", str(cm.exception))

    def test_complete_checkpoint_filters_non_indexed(self):
        # All indexed shards present; a non-indexed duplicate is still filtered out.
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        shard1 = _touch(self.folder, "model-00001-of-00002.safetensors")
        shard2 = _touch(self.folder, "model-00002-of-00002.safetensors")
        extra = _touch(self.folder, "consolidated.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[shard1, shard2, extra],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(sorted(result), sorted([shard1, shard2]))

    def test_single_file_model_no_index_returns_unchanged(self):
        # No index on disk (single-file / dummy / object-storage): early return.
        single = _touch(self.folder, "model.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[single],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(result, [single])


class TestQuantizationConfigHook(CustomTestCase):
    def test_file_based_modelopt_fp8_uses_platform_quantization_class(self):
        class PlatformModelOptFp8Config:
            @staticmethod
            def get_config_filenames():
                return ["hf_quant_config.json"]

            @classmethod
            def from_config(cls, config):
                return cls()

        with tempfile.TemporaryDirectory() as model_path:
            with open(os.path.join(model_path, "hf_quant_config.json"), "w") as f:
                json.dump(
                    {
                        "producer": {"name": "modelopt"},
                        "quantization": {"quant_algo": "FP8"},
                    },
                    f,
                )
            model_config = SimpleNamespace(
                quantization="modelopt_fp8",
                hf_config=SimpleNamespace(quantization_config=None, text_config=None),
                model_path=model_path,
                revision=None,
            )
            load_config = SimpleNamespace(model_loader_extra_config=None)

            with patch.object(
                weight_utils,
                "get_quantization_config",
                return_value=PlatformModelOptFp8Config,
            ):
                config = get_quant_config(model_config, load_config, {})

        self.assertIsInstance(config, PlatformModelOptFp8Config)


if __name__ == "__main__":
    unittest.main()
