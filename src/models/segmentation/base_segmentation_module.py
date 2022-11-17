from typing import Any, List

import torch
from torch_geometric.data import Batch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from src.models.base_module import BaseModule
from src.models.common import LrScheduler
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class BaseSegmentationModule(BaseModule):
    """LightningModule Docs:

    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        lr_scheduler: LrScheduler,
        num_classes: int,
    ):
        super().__init__(optimizer, lr_scheduler)

        # loss function
        self.criterion = criterion

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = MulticlassAccuracy(num_classes)
        self.val_acc = MulticlassAccuracy(num_classes)
        self.test_acc = MulticlassAccuracy(num_classes)
        self.train_iou = MulticlassJaccardIndex(num_classes)
        self.val_iou = MulticlassJaccardIndex(num_classes)
        self.test_iou = MulticlassJaccardIndex(num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # tracking best metrics, for use in hyperparameter optimization
        self.train_acc_best = MaxMetric()
        self.val_acc_best = MaxMetric()
        self.test_acc_best = MaxMetric()
        self.train_iou_best = MaxMetric()
        self.val_iou_best = MaxMetric()
        self.test_iou_best = MaxMetric()

    def forward(self, batch):
        raise NotImplementedError()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Batch):
        raise NotImplementedError()

    def training_step(self, batch: Batch, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_iou(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_iou, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        """Compute the best metrics at the end of the epoch for use in hyperparameter tuning. Log
        the value using the compute method rather than a metric object because otherwise, lightning
        will reset the metric after each epoch (and we want to track the best metrics across
        epochs)

        Args:
            outputs (List[Any]): outputs returned from the epoch step
        """
        acc = self.train_acc.compute()
        self.train_acc_best(acc)
        self.log("train/acc_best", self.train_acc_best.compute(), prog_bar=True)

        iou = self.train_iou.compute()
        self.train_iou_best(iou)
        self.log("train/iou_best", self.train_iou_best.compute(), prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_iou(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        """Compute the best metrics at the end of the epoch for use in hyperparameter tuning. Log
        the value using the compute method rather than a metric object because otherwise, lightning
        will reset the metric after each epoch (and we want to track the best metrics across
        epochs)

        Args:
            outputs (List[Any]): outputs returned from the epoch step
        """
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

        iou = self.val_iou.compute()
        self.val_iou_best(iou)
        self.log("val/iou_best", self.val_iou_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_iou(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        """Compute the best metrics at the end of the epoch for use in hyperparameter tuning. Log
        the value using the compute method rather than a metric object because otherwise, lightning
        will reset the metric after each epoch (and we want to track the best metrics across
        epochs)

        Args:
            outputs (List[Any]): outputs returned from the epoch step
        """
        acc = self.test_acc.compute()  # get current test acc
        self.test_acc_best(acc)  # update best so far test acc
        # log `test_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("test/acc_best", self.test_acc_best.compute(), prog_bar=True)

        iou = self.test_iou.compute()
        self.test_iou_best(iou)
        self.log("test/iou_best", self.test_iou_best.compute(), prog_bar=True)
