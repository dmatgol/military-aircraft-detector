from __future__ import annotations

from src.settings.general import ModelConfig


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def read_model_config(model_config_path: str) -> ModelConfig:

    model_config = ModelConfig.from_yaml(model_config_path)

    return model_config
