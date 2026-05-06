from __future__ import annotations

from typing import Any

from lerobot.processor.converters import create_transition


def preprocess_observation_with_metadata(
    preprocessor: Any,
    observation: dict[str, Any],
    task: str | None = None,
    subtask: str | None = None,
) -> dict[str, Any]:
    """Run a preprocessor on an observation plus optional complementary metadata."""
    complementary_data: dict[str, Any] = {}
    if task is not None:
        complementary_data["task"] = task
    if subtask is not None:
        complementary_data["subtask"] = subtask

    if not complementary_data:
        return preprocessor(observation)

    transition = create_transition(observation=observation, complementary_data=complementary_data)
    transformed_transition = preprocessor._forward(transition)
    return preprocessor.to_output(transformed_transition)
