from typing import Callable, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode

default_transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize(size=(256, 256), interpolation=InterpolationMode.BILINEAR),
    ]
)
default_target_transform = T.Compose(
    [
        T.PILToTensor(),
        T.Resize(size=(256, 256), interpolation=InterpolationMode.NEAREST),
    ]
)


class VOCSegmentationDataModule(pl.LightningDataModule):

    classes = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = default_transform,
        target_transform: Optional[Callable] = default_target_transform,
        transforms: Optional[Callable] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.num_classes = len(self.classes)
        self.idx2class = {i: c for i, c in enumerate(self.classes)}
        self.idx2class[255] = "ignore"

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = VOCSegmentation(
            root=self.root,
            year="2012",
            image_set="train",
            transform=self.transform,
            target_transform=self.target_transform,
            transforms=self.transforms,
        )
        self.val_dataset = VOCSegmentation(
            root=self.root,
            year="2012",
            image_set="val",
            transform=self.transform,
            target_transform=self.target_transform,
            transforms=self.transforms,
        )
        self.test_dataset = VOCSegmentation(
            root=self.root,
            year="2007",
            image_set="test",
            transform=self.transform,
            target_transform=self.target_transform,
            transforms=self.transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def on_before_batch_transfer(
        self, batch: Tuple[torch.Tensor, torch.Tensor], dataloader_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        y[y == 255] = 0
        return x, y
