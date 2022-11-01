from torch.utils.data import DataLoader, Dataset

from acnato_classifier.pipelines.base import Pipeline


class Train(Pipeline):
    def __init__(self, train_dataset: Dataset, valid_dataset: Dataset):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def run(self) -> None:

        # train_dataloader = self.load_dataloaders(self.train_dataset, True)
        # validation_dataloader = self.load_dataloaders(self.valid_dataset, False)
        pass

    def load_dataloaders(self, dataset: Dataset, shuffle: bool):
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=shuffle,
            num_workers=0,
        )
        return dataloader
