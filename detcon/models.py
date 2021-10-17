from typing import Callable, Dict, Tuple

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
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )


class Encoder(nn.Module):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = False):
        self.model = getattr(torchvision.models, backbone)(pretrained)
        self.emb_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model.avgpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model
        return self.model(x)


class DetCon(pl.LightningModule):
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
        self.encoder = Encoder(backbone, pretrained)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class DetConS(DetCon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        proj_hidden_dim = 2048
        proj_dim = 128
        self.projector = MLP(self.encoder.emb_dim, proj_hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.encoder(x)
        z = self.projector(e)
        return e, z

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        x, y = batch
        x1, y1 = self.augs(x, y)
        x2, y2 = self.augs(x, y)
        e1, z1 = self(x1)
        e2, z2 = self(x2)
        loss = self.loss_fn(e1, e2, z1, z2, y1, y2)
        self.log("loss", loss)
        return loss
