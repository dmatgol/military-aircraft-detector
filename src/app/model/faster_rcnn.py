from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.utils import draw_boxes, draw_fps


class FasterRCNNModel(LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 0.0002,
    ):
        super().__init__()
        self.model = self._create_model(
            num_classes,
        )
        self.lr = learning_rate
        self.num_classes = num_classes
        self.map = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    def _create_model(self, num_classes: int):
        # load Faster RCNN pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def forward(self, images: torch.tensor):
        return self.model(images)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        batch_size = len(images)

        loss_dict = self.model(images, targets)

        # optimize for all losses - classifier, bbox,
        losses = sum(loss for loss in loss_dict.values())
        self.log(
            "train_loss",
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        batch_size = len(images)
        # set to train mode to get validation loss
        self.model.train()
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log(
            "val_loss",
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        # set to eval mode to get predictions
        self.model.eval()
        preds = self.model(images)
        self.map.update(preds=preds, target=targets)

    def validation_epoch_end(self, validation_step_outputs):
        mAPs = {"val_" + k: v for k, v in self.map.compute().items()}
        self.print(mAPs)
        self.map.reset()

    def predict_image(self, image_path: str | Any, iou_threshold: float = 0.85):
        self.model.eval()
        if isinstance(image_path, str):
            orig_image = cv2.imread(image_path)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        else:  # uploaded images from frontend
            arr = np.frombuffer(image_path, np.uint8)
            orig_image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        image = self._process_image_for_inference(orig_image)
        pred = self.model(image)
        # verify if we have predictions and if so draw boxes
        orig_image = self.draw_bboxes_if_detection(orig_image, pred, iou_threshold)
        return orig_image

    def _process_image_for_inference(
        self, orig_image: torch.tensor, resize: bool = False
    ):
        image = orig_image.copy().astype(np.float32)
        # make the pixel range between 0 and 1
        if resize:
            image = cv2.resize(image, (256, 256))
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float)
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        return image

    def draw_bboxes_if_detection(
        self,
        orig_image: torch.tensor,
        pred: list[dict[str, torch.tensor]],
        iou_threshold: float,
        resize: tuple[int, int] = None,
    ):
        boxes = pred[0]["boxes"].data.numpy()
        scores = pred[0]["scores"].data.numpy()
        # filter out boxes according to `iou_threshold`
        boxes = boxes[scores >= iou_threshold].astype(np.int32)
        # we only have one class
        pred_classes = ["military_aircraft" for i in range(len(boxes))]

        # draw the bounding boxes and write the class name on top of it
        orig_image = draw_boxes(
            orig_image=orig_image,
            boxes=boxes,
            pred_classes=pred_classes,
            resize=resize,
        )
        return orig_image

    def predict_video(self, video_path: str, iou_threshold: float = 0.85, stframe=None):
        self.model.eval()
        cap = self._read_video(video_path)
        cap = self.process_frames_and_predict(cap, iou_threshold, stframe)
        return cap

    def _read_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)

        if cap.isOpened() is False:
            logging.error(
                "Error trying to read video. Please verify if path is correct."
            )

        return cap

    def process_frames_and_predict(
        self, cap: cv2.VideoCapture, iou_threshold: float, stframe
    ):
        # Process entire video
        while cap.isOpened():
            # Capture each frame of the video
            frame_return_value, frame = cap.read()
            if frame_return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = self._process_image_for_inference(frame)
                # Compute prediction time to calculate frames per second
                start_time = time.time()
                with torch.no_grad():
                    # Compute predictions for the current frame
                    pred = self.model(image)
                end_time = time.time()

                fps = 1 / (end_time - start_time)
                if len(pred[0]["boxes"]):  # if there are predictions
                    frame, _ = self.draw_bboxes_if_detection(frame, pred, iou_threshold)
                frame = draw_fps(frame, fps)
                stframe.image(frame, use_column_width=True)

            else:
                break

        cap.release()
