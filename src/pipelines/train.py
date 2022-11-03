from __future__ import annotations

import os
from datetime import datetime

from pytorch_lightning import Trainer

from pipelines.base import Pipeline
from src.data.dataset.military_aircraft_data_module import MilitaryAircraftDataModule
from src.model.faster_rcnn import FasterRCNNModel
from src.settings.general import ModelConfig


class Train(Pipeline):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        folder = self._create_experiment_folder()
        self.tensorboard_results = os.path.join(
            os.getcwd(), f"src/model_experiments/{folder}"
        )

    def run(self) -> None:

        data_module = MilitaryAircraftDataModule(
            num_workers=self.model_config.model_train_config["num_workers"],
            batch_size=self.model_config.model_train_config["batch_size"],
            img_width=self.model_config.model_train_config["img_width"],
            img_height=self.model_config.model_train_config["img_height"],
        )
        model = FasterRCNNModel(**self.model_config.model_parameters)

        trainer = Trainer(
            max_epochs=self.model_config.model_train_config["max_epochs"],
            default_root_dir=self.tensorboard_results,
        )
        trainer.fit(model, data_module)

    def _create_experiment_folder(self):
        folder = (
            str(datetime.now())
            .replace(" ", "_")
            .replace(":", "_")
            .replace("-", "_")
            .split(".")[0]
        )
        os.mkdir(os.path.join(os.getcwd(), f"src/model_experiments/{folder}/"))
        return folder
