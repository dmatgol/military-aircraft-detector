import os

import torch
from model.faster_rcnn import FasterRCNNModel
from settings.general import DEVICE, data_paths


def load_trained_model(model_config):
    model = FasterRCNNModel(**model_config.model_parameters)
    checkpoint_dir = os.listdir(data_paths.best_model)
    for file in checkpoint_dir:
        if file.endswith(".ckpt"):
            checkpoint_path = file
    checkpoint = torch.load(
        os.path.join(data_paths.best_model, checkpoint_path),
        map_location=torch.device(DEVICE),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
