from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MilitaryAircraftDataset(Dataset):
    def __init__(
        self,
        image_annotations_path: str,
        width: int,
        height: int,
        classes: list[str],
        transformations: A = None,
    ):
        self.image_annotations_df = pd.read_csv(image_annotations_path)
        self.width = width
        self.height = height
        self.classes = classes
        self.transformations = transformations

    def __getitem__(self, idx):
        image_path = self.image_annotations_df.iloc[idx].ImagePath
        annotation_path = self.image_annotations_df.iloc[idx].AnnotationPath

        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (self.width, self.height)).astype(np.float32)
        image_resized /= 255.0  # RGB code max 255

        boxes = []
        labels = []

        image_height = image.shape[0]
        image_width = image.shape[1]

        image_annotations = pd.read_csv(annotation_path)
        for index, row in image_annotations.iterrows():
            xmin_resized = (row["xmin"] / image_width) * self.width
            xmax_resized = (row["xmax"] / image_width) * self.width
            ymin_resized = (row["ymin"] / image_height) * self.height
            ymax_resized = (row["ymax"] / image_height) * self.height

            boxes.append([xmin_resized, ymin_resized, xmax_resized, ymax_resized])
            labels.append(
                self.classes.index("military_aircraft")
            )  # only want to identify military aircraft and not the type

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.transformations:
            print(labels)
            print("entrou")
            image_transformed = self.transformations(
                image=image_resized, bboxes=boxes, labels=labels
            )
            image_resized = image_transformed["image"]
            print(image_resized)
            boxes = image_transformed["bboxes"]

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "image_path": image_path,
        }
        return image_resized, target

    def __len__(self):
        return len(self.image_annotations_df)
