from typing import Callable, Dict

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from detcon.losses import DetConLoss

default_augs = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
    K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur(kernel_size=(32, 32), sigma=(0.1, 2.0), p=0.5),
    K.RandomResizedCrop(size=(256, 256), scale=(0.08, 1.0), resample="NEAREST"),
    K.RandomSolarize(thresholds=0.1, p=0.2),
)


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )


class DetConModule(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = False,
        augs: Callable = default_augs,
        loss_fn: nn.Module = DetConLoss(),
    ):
        super().__init__()
        self.augs = augs
        self.loss_fn = loss_fn
        self.encoder = getattr(torchvision.models, backbone)(pretrained)
        self.emb_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.Identity()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DetConSModule(DetConModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projector = MLP(input_dim=self.emb_dim, hidden_dim=2048, output_dim=128)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        x, y = batch
        x1, y1 = self.augs(x, y)
        x2, y2 = self.augs(x, y)
        z1 = self(x1)
        z2 = self(x2)
        loss = self.loss_fn(z1, z2, y1, y2)
        self.log("loss", loss)
        return loss
