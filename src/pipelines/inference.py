from __future__ import annotations

import os

import torch

from model.faster_rcnn import FasterRCNNModel
from pipelines.base import Pipeline
from settings.general import ModelConfig, data_paths


class Inference(Pipeline):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def run(self) -> None:

        trained_model = self._load_trained_model()
        for image in os.listdir(data_paths.test_images):
            test_image = os.path.join(data_paths.test_images, image)
            trained_model.predict_image(test_image)
        for video in os.listdir(data_paths.test_videos):
            test_video = os.path.join(data_paths.test_videos, video)
            trained_model.predict_video(test_video)

    def _load_trained_model(self):
        model = FasterRCNNModel(**self.model_config.model_parameters)
        checkpoint_path = os.listdir(data_paths.best_model)[0]
        checkpoint = torch.load(
            os.path.join(data_paths.best_model, checkpoint_path),
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
