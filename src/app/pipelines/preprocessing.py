import os
import shutil

import pandas as pd
from pipelines.base import Pipeline


class PreProcessing(Pipeline):
    def __init__(self, full_dataset_dir: str) -> None:
        self.full_dataset_dir = full_dataset_dir

    def run(self) -> None:
        data_dir = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )  # get src/app
        self.create_train_validation_folder(data_dir=data_dir)
        train_df, valid_df = self.train_valid_split(self.full_dataset_dir)
        self.copy_images_and_annotations_to_folder(train_df, f"{data_dir}/data/train")
        self.copy_images_and_annotations_to_folder(valid_df, f"{data_dir}/data/valid")

    @staticmethod
    def train_valid_split(dataset_dir_path: str, valid_frac: float = 0.20):

        images_annotations_path_list = []
        for file in os.listdir(dataset_dir_path):
            if not file.endswith(".csv"):
                file_name = file.split(".")[0]
                image_annotation = {
                    "AnnotationPath": os.path.join(
                        dataset_dir_path, f"{file_name}.csv"
                    ),
                    "ImagePath": os.path.join(dataset_dir_path, file),
                }
                images_annotations_path_list.append(image_annotation)

        images_annotations_path_df = pd.DataFrame(images_annotations_path_list)
        images_annotations_path_df.sample(frac=1, random_state=0)  # shuffle dataset
        train_split = int((1 - valid_frac) * len(images_annotations_path_df))

        train_df = images_annotations_path_df.iloc[:train_split]
        valid_df = images_annotations_path_df.iloc[train_split:]
        return train_df, valid_df

    @staticmethod
    def copy_images_and_annotations_to_folder(dataset: pd.DataFrame, dest_folder: str):
        for index, row in dataset.iterrows():
            for column in ["AnnotationPath", "ImagePath"]:
                shutil.copy(f"{row[column]}", dest_folder)
                # change path to either train or valid folder
                row[column] = row[column].replace(
                    "images_annotations", dest_folder.split("/")[-1]
                )

        dataset.to_csv(f"{dest_folder}/image_annotations.csv", index=False)

    @staticmethod
    def create_train_validation_folder(data_dir: str):
        os.makedirs(f"{data_dir}/data/train", exist_ok=True)
        os.makedirs(f"{data_dir}/data/valid", exist_ok=True)
