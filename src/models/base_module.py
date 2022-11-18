import omegaconf
import torch
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch

from src.models.common import LrScheduler
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class BaseModule(LightningModule):
    """LightningModule Docs:

    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LrScheduler,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, batch):
        raise NotImplementedError()

    def step(self, batch: Batch):
        raise NotImplementedError()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        if self.lr_scheduler is not None and self.lr_scheduler.scheduler is not None:
            # print(self.hparams.scheduler)
            scheduler = self.lr_scheduler.scheduler(optimizer=optimizer)
            scheduler_config = omegaconf.OmegaConf.to_container(self.lr_scheduler.config)
            scheduler_config["scheduler"] = scheduler
            log.info(f"Using LR Scheduler {type(scheduler)}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_config,
            }
        return {"optimizer": optimizer}
