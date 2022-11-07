from __future__ import annotations

import os
from datetime import datetime

from data.dataset.military_aircraft_data_module import MilitaryAircraftDataModule
from model.faster_rcnn import FasterRCNNModel
from pipelines.base import Pipeline
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from settings.general import ModelConfig


class Train(Pipeline):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        folder = self._create_experiment_folder()
        self.tensorboard_results = os.path.join(
            os.getcwd(), f"src/app/model_experiments/{folder}"
        )

    def run(self) -> None:

        data_module = MilitaryAircraftDataModule(
            num_workers=self.model_config.model_train_config["num_workers"],
            batch_size=self.model_config.model_train_config["batch_size"],
            img_width=self.model_config.model_train_config["img_width"],
            img_height=self.model_config.model_train_config["img_height"],
        )
        model = FasterRCNNModel(**self.model_config.model_parameters)
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.tensorboard_results,
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        trainer = Trainer(
            max_epochs=self.model_config.model_train_config["max_epochs"],
            callbacks=[checkpoint_callback],
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
        os.makedirs(os.path.join(os.getcwd(), f"src/app/model_experiments/{folder}/"))
        return folder
