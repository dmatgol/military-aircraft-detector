from __future__ import annotations

import cv2
import torch

from settings.general import ModelConfig


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def read_model_config(model_config_path: str) -> ModelConfig:

    model_config = ModelConfig.from_yaml(model_config_path)

    return model_config


def draw_boxes(
    orig_image: torch.tensor,
    boxes: torch.tensor,
    pred_classes: list[str],
    resize: tuple[int, int] = None,
):
    # draw the bounding boxes and write the class name on top of it
    object_detected = False
    for j, box in enumerate(boxes):
        object_detected = True
        if resize is not None:
            box = [
                int(box[0] / resize[0]) * orig_image.shape[1],  # width
                int(box[1] / resize[1]) * orig_image.shape[0],  # height
                int(box[2] / resize[0]) * orig_image.shape[1],  # width
                int(box[3] / resize[1]) * orig_image.shape[1],  # height
            ]

        cv2.rectangle(
            orig_image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            orig_image,
            pred_classes[j],
            (int(box[0]), int(box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    return orig_image, object_detected


def draw_fps(orig_image: torch.tensor, fps: float):
    cv2.putText(
        orig_image,
        f"{fps:.1f} FPS",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )
    return orig_image
