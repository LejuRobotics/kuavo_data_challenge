from typing import Any

import torch

from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import (
    make_groot_pre_post_processors as _base_make_groot_pre_post_processors,
)
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_groot_pre_post_processors(
    config: GrootConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Forward to the official LeRobot Groot pre/post processor factory.

    Keeping this thin wrapper preserves the public import path used in training
    scripts while delegating the actual logic to `lerobot`.
    """
    return _base_make_groot_pre_post_processors(config=config, dataset_stats=dataset_stats)
