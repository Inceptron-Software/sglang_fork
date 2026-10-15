from typing import Dict, Type

from sglang.benchmark.datasets.agentic_trace import AgenticTraceDataset
from sglang.benchmark.datasets.common import BaseDataset, DatasetRow
from sglang.benchmark.datasets.custom import CustomDataset
from sglang.benchmark.datasets.generated_shared_prefix import (
    GeneratedSharedPrefixDataset,
)
from sglang.benchmark.datasets.image import ImageDataset
from sglang.benchmark.datasets.longbench_v2 import LongBenchV2Dataset
from sglang.benchmark.datasets.mooncake import MooncakeDataset
from sglang.benchmark.datasets.openai_dataset import OpenAIDataset
from sglang.benchmark.datasets.random import RandomDataset
from sglang.benchmark.datasets.sharegpt import ShareGPTDataset
from sglang.benchmark.datasets.speed_bench import SpeedBenchDataset

# MMMU is the only dataset backed by the HF `datasets` package, whose import
# can fail in environments with incompatible `datasets`/`pyarrow` versions.
# Defer that failure to requests that actually select the mmmu dataset.
try:
    from sglang.benchmark.datasets.mmmu import MMMUDataset

    _MMMU_IMPORT_ERROR: Exception | None = None
except Exception as error:
    MMMUDataset = None
    _MMMU_IMPORT_ERROR = error

DATASET_MAPPING: Dict[str, Type[BaseDataset]] = {
    "agentic-trace": AgenticTraceDataset,
    "sharegpt": ShareGPTDataset,
    "custom": CustomDataset,
    "openai": OpenAIDataset,
    # TODO: "random" vs "random-ids" should be a flag (e.g. --random-source=sharegpt|integers),
    # not two separate dataset names sharing the same class.
    "random": RandomDataset,
    "random-ids": RandomDataset,
    "generated-shared-prefix": GeneratedSharedPrefixDataset,
    "mmmu": MMMUDataset,
    "image": ImageDataset,
    "mooncake": MooncakeDataset,
    "longbench_v2": LongBenchV2Dataset,
    "speed-bench": SpeedBenchDataset,
}


def get_dataset(args, tokenizer, model_id=None):
    dataset_name = args.dataset_name
    if dataset_name.startswith("random") and dataset_name not in DATASET_MAPPING:
        dataset_name = "random-ids"

    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    dataset_cls = DATASET_MAPPING[dataset_name]
    if dataset_cls is None:
        raise RuntimeError(
            f"Dataset {args.dataset_name!r} is unavailable: {_MMMU_IMPORT_ERROR}"
        )
    dataset = dataset_cls.from_args(args)
    return dataset.load(tokenizer=tokenizer, model_id=model_id)


__all__ = [
    "DATASET_MAPPING",
    "DatasetRow",
    "get_dataset",
]
