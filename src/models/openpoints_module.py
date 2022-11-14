from typing import Any, List

import torch
from torch_geometric.data import Batch
import omegaconf
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)

class OpenPointsModule(LightningModule):
    """
    LightningModule Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: omegaconf.DictConfig,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: dict, # todo: make this into a dataclass
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        cfg = EasyConfig()
        cfg.update(omegaconf.OmegaConf.to_container(net, resolve=True))
        self.net = build_model_from_cfg(cfg)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # tracking best metrics, for use in hyperparameter optimization
        self.train_acc_best = MaxMetric()
        self.val_acc_best = MaxMetric()
        self.test_acc_best = MaxMetric()

    def forward(self, batch):
        return self.net(batch)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Batch):
        pos, x, y = batch.pos, batch.x, batch.y
        if x:
            x = x.transpose(1, 2).contiguous()

        # print(y.shape)
        logits = self.forward({"pos": pos, "x": x})
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Batch, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        # print(preds, targets)
        self.train_acc(preds, targets)
        # print(self.train_acc)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        acc = self.train_acc.compute()  # get current test acc
        self.train_acc_best(acc)  # update best so far test acc
        # log `test_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("train/acc_best", self.train_acc_best.compute(), prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        acc = self.test_acc.compute()  # get current test acc
        self.test_acc_best(acc)  # update best so far test acc
        # log `test_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("test/acc_best", self.test_acc_best.compute(), prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.lr_scheduler is not None and self.hparams.lr_scheduler.scheduler is not None:
            # print(self.hparams.scheduler)
            scheduler = self.hparams.lr_scheduler.scheduler(optimizer=optimizer)
            scheduler_config = omegaconf.OmegaConf.to_container(self.hparams.lr_scheduler.config)
            scheduler_config["scheduler"] = scheduler
            log.info(f"Using LR Scheduler {type(scheduler)}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_config,
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
