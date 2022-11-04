import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.data_augmentations import get_train_transform, get_valid_transform
from data.dataset.military_aircraft import MilitaryAircraftDataset
from settings.general import data_paths
from utils.utils import collate_fn


class MilitaryAircraftDataModule(LightningDataModule):
    """LightningDataModule based on MilitaryAircraft dataset.
    The module is used for training.
    Args:
        num_workers (int): number of workers to use for loading data
        batch_size (int): batch size
        img_size (int): image size to resize input data to during data
         augmentation
    """

    def __init__(
        self,
        num_workers: int,
        batch_size: int,
        img_width: int,
        img_height: int,
    ):
        super().__init__()
        self.train_transforms = get_train_transform()
        self.val_transforms = get_valid_transform()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height

    def train_dataset(self):
        return MilitaryAircraftDataset(
            data_paths.train_image_annotations,
            self.img_width,
            self.img_height,
            ["background", "military_aircraft"],
            get_train_transform(),
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

        return train_loader

    def val_dataset(self):
        return MilitaryAircraftDataset(
            data_paths.valid_image_annotations,
            self.img_width,
            self.img_height,
            ["background", "military_aircraft"],
            get_train_transform(),
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.val_dataset()
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

        return val_loader
