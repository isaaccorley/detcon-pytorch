from typing import Dict, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from einops import rearrange, reduce
from torch_ema import ExponentialMovingAverage

from detcon.losses import DetConBLoss


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )


class Encoder(nn.Sequential):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = False) -> None:
        model = getattr(torchvision.models, backbone)(pretrained)
        self.emb_dim = model.fc.in_features
        model.fc = nn.Identity()
        model.avgpool = nn.Identity()
        super().__init__(*list(model.children()))


class MaskPooling(nn.Module):
    def __init__(
        self, num_classes: int, num_samples: int = 16, downsample: int = 32
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.mask_ids = torch.arange(num_classes)
        self.pool = nn.AvgPool2d(kernel_size=downsample, stride=downsample)

    def pool_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Create binary masks, average pool, flatten, argmax,"""
        masks = masks == self.mask_ids[None, :, None, None]
        masks = self.pool(masks.to(torch.float))
        masks = rearrange(masks, "b c h w -> b c (h w)")
        masks = torch.argmax(masks, dim=1)
        masks = torch.eye(self.num_classes)[masks]
        return masks

    def sample_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Samples which binary masks to use in the loss."""
        bs = masks.shape[0]
        mask_exists = torch.greater(masks.sum(dim=-1), 1e-3)
        sel_masks = mask_exists.to(torch.float) + 0.00000000001
        sel_masks = sel_masks / sel_masks.sum(dim=1, keepdim=True)
        sel_masks = torch.log(sel_masks)
        print(sel_masks)
        mask_ids = torch.multinomial(sel_masks, num_samples=self.num_samples)
        sampled_masks = torch.stack([masks[b][mask_ids[b]] for b in range(bs)])
        return sampled_masks, mask_ids

    def forward(self, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        binary_masks = self.pool_masks(masks)
        sampled_masks, sampled_mask_ids = self.sample_masks(binary_masks)
        area = sampled_masks.sum(dim=-1, keepdims=True)
        sampled_masks = sampled_masks / torch.maximum(area, 1.0)
        return sampled_masks, sampled_mask_ids


class Network(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = False,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_classes: int = 10,
        downsample: int = 32,
        num_samples: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(backbone, pretrained)
        self.projector = MLP(self.encoder.emb_dim, hidden_dim, output_dim)
        self.mask_pool = MaskPooling(num_classes, num_samples, downsample)

    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> Sequence[torch.Tensor]:
        m, mids = self.mask_pool(masks)
        e2d = self.encoder(x)
        e1d = reduce(e2d, "b c h w -> b c ()", reduction="mean")
        e1d = m * e1d.t()
        m, mids = self.mask_pool(masks)
        p = self.projector(e1d)
        return e1d, p, m, mids


class DetConB(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 21,
        num_samples: int = 5,
        backbone: str = "resnet50",
        pretrained: bool = False,
        downsample: int = 32,
        proj_hidden_dim: int = 128,
        proj_dim: int = 256,
        loss_fn: nn.Module = DetConBLoss(),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])
        self.loss_fn = loss_fn
        self.network = Network(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_classes=num_classes,
            downsample=downsample,
            num_samples=num_samples,
        )
        self.network_ema = ExponentialMovingAverage(
            self.network.parameters(), decay=0.995
        )
        self.temperature = nn.Parameter(torch.tensor(0.0))
        self.predictor = MLP(proj_dim, proj_hidden_dim, proj_dim)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_before_zero_grad(self, *args, **kwargs):
        """See https://forums.pytorchlightning.ai/t/adopting-exponential-moving-average-ema-for-pl-pipeline/488"""  # noqa: E501
        self.network_ema.update(self.network.parameters())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.network(x, y)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        (x1, x2), (y1, y2) = batch["image"], batch["mask"]

        # encode and project
        _, p1, _, mids1 = self(x1, y1)
        _, p2, _, mids2 = self(x2, y2)

        # ema encode and project
        _, pm1, _, midsm1 = self(x1, y1)
        _, pm2, _, midsm2 = self(x2, y2)

        # predict
        q1, q2 = self.predictor(p1), self.predictor(p2)

        # compute loss
        loss = self.loss_fn(
            pred1=q1,
            pred2=q2,
            target1=pm1.detach(),
            target2=pm2.detach(),
            pind1=mids1,
            pind2=mids2,
            tind1=midsm1,
            tind2=midsm2,
            temperature=self.temperature,
        )
        self.log("loss", loss)
        return loss
