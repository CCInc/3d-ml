import omegaconf
import torch
from torch_geometric.data import Batch

from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig
from src.models.common import LrScheduler
from src.models.segmentation.base_segmentation_module import BaseSegmentationModule
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class OpenPointsModule(BaseSegmentationModule):
    def __init__(
        self,
        net: omegaconf.DictConfig,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        lr_scheduler: LrScheduler,
        num_classes: int,
    ):
        super().__init__(optimizer, criterion, lr_scheduler, num_classes)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        cfg = EasyConfig()
        cfg.update(omegaconf.OmegaConf.to_container(net, resolve=True))
        self.net = build_model_from_cfg(cfg)

    def forward(self, batch):
        return self.net(batch)

    def step(self, batch: Batch):
        pos, x, y = batch.pos, batch.x, batch.y
        if x is not None:
            # OpenPoints requires these channels to be flipped
            x = x.transpose(1, 2).contiguous()

        logits = self.forward({"pos": pos, "x": x})
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
