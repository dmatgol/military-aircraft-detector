from __future__ import annotations

import os

from model.model_utils import load_trained_model
from pipelines.base import Pipeline
from settings.general import ModelConfig, data_paths


class Inference(Pipeline):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def run(self) -> None:

        trained_model = load_trained_model(self.model_config)
        for image in os.listdir(data_paths.test_images):
            test_image = os.path.join(data_paths.test_images, image)
            trained_model.predict_image(test_image)
        for video in os.listdir(data_paths.test_videos):
            test_video = os.path.join(data_paths.test_videos, video)
            trained_model.predict_video(test_video)
