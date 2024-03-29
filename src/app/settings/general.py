from typing import Any, Dict, List

import torch
import yaml
from pydantic import BaseModel, BaseSettings


class Paths(BaseSettings):

    train_image_annotations: str = "src/app/data/train/image_annotations.csv"
    valid_image_annotations: str = "src/app/data/valid/image_annotations.csv"
    best_model: str = "src/app/model/best_model/checkpoints/"
    test_images: str = "src/app/data/test/test_images/"
    test_videos: str = "src/app/data/test/test_videos/"


class ModelConfig(BaseModel):
    model_name: str
    classes: List[str]
    model_parameters: Dict[str, Any]
    model_train_config: Dict[str, Any]

    @classmethod
    def from_yaml(cls, model_config_path: str):
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)

        if model_config is None:
            return cls(
                model_name="FasterRCNN",
                classes=["background", "military_aircraft"],
                model_parameters={
                    "num_classes": 2,
                },
                model_train_config={
                    "batch_size": 8,
                    "max_epochs": 10,
                    "num_workers": 0,
                    "img_width": 512,
                    "img_height": 512,
                },
            )

        return cls(**model_config)


DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
data_paths = Paths()
