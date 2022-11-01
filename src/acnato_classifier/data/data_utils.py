import os
import shutil

import pandas as pd


class DataUtils:
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

        parent_dir = os.path.dirname(os.path.realpath(__file__))
        train_df = images_annotations_path_df.iloc[:train_split]
        train_df.to_csv(f"{parent_dir}/train/image_annotations.csv", index=False)
        valid_df = images_annotations_path_df.iloc[train_split:]
        valid_df.to_csv(f"{parent_dir}/valid/image_annotations.csv", index=False)
        return train_df, valid_df

    def copy_images_and_annotations_to_folder(dataset: pd.DataFrame, dest_folder: str):
        for index, row in dataset.iterrows():
            for column in ["AnnotationPath", "ImagePath"]:
                shutil.copy(f"{row[column]}", dest_folder)

    @staticmethod
    def create_train_validation_folder():
        data_dir = os.path.dirname(os.path.realpath(__file__))
        os.makedirs(f"{data_dir}/train", exist_ok=True)
        os.makedirs(f"{data_dir}/valid", exist_ok=True)


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.realpath(__file__))
    DataUtils.create_train_validation_folder()
    train_df, valid_df = DataUtils.train_valid_split(os.environ["DATASET_PATH"])
    DataUtils.copy_images_and_annotations_to_folder(train_df, f"{data_dir}/train")
    DataUtils.copy_images_and_annotations_to_folder(valid_df, f"{data_dir}/valid")
