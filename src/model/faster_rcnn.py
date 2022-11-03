import torch
import torchvision
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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

    def forward(self, images):
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
