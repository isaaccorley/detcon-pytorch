from typing import Callable, Dict, Optional, Tuple

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode


class RandomResizedCrop(K.RandomResizedCrop):
    def __init__(self, *args, **kwargs) -> None:
        if kwargs["align_corners"] is None:
            kwargs["align_corners"] = False
        super().__init__(*args, **kwargs)
        self.align_corners = None


default_transform = T.Compose(
    [T.ToTensor(), T.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR)]
)

default_target_transform = T.Compose(
    [
        T.PILToTensor(),
        T.Resize(size=(224, 224), interpolation=InterpolationMode.NEAREST),
    ]
)

default_augs = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5), data_keys=["input", "mask"]
)

default_ssl_augs = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
    K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0), p=0.5),
    RandomResizedCrop(
        size=(224, 224), scale=(0.08, 1.0), resample="NEAREST", align_corners=None
    ),
    K.RandomSolarize(thresholds=0.1, p=0.2),
    data_keys=["input", "mask"],
)


class VOCSegmentationBaseDataModule(pl.LightningDataModule):

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
        augmentations: K.AugmentationSequential = default_augs,
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = 2,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.num_classes = len(self.classes)
        self.idx2class = {i: c for i, c in enumerate(self.classes)}
        self.idx2class[255] = "ignore"

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def on_before_batch_transfer(
        self, batch: Tuple[torch.Tensor, torch.Tensor], dataloader_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        y[y == 255] = 0
        return {"image": x, "mask": y}


class VOCSegmentationDataModule(VOCSegmentationBaseDataModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

    def on_after_batch_transfer(
        self, batch: Dict[str, torch.Tensor], dataloader_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch["mask"] = batch["mask"].to(torch.float)
        batch["image"], batch["mask"] = self.augmentations(
            batch["image"], batch["mask"]
        )
        batch["mask"] = batch["mask"].to(torch.long)
        batch["mask"] = batch["mask"].squeeze(dim=1)
        return batch


class VOCSSLDataModule(VOCSegmentationBaseDataModule):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["augmentations"] = default_ssl_augs
        super().__init__(*args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = VOCSegmentation(
            root=self.root,
            year="2012",
            image_set="train",
            transform=self.transform,
            target_transform=self.target_transform,
            transforms=self.transforms,
        )

    def on_after_batch_transfer(
        self, batch: Dict[str, torch.Tensor], dataloader_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch["mask"] = batch["mask"].to(torch.float)
        image1, mask1 = self.augmentations(batch["image"], batch["mask"])
        image2, mask2 = self.augmentations(batch["image"], batch["mask"])
        mask1 = mask1.squeeze(dim=1).to(torch.long)
        mask2 = mask2.squeeze(dim=1).to(torch.long)
        batch = {"image": (image1, image2), "mask": (mask1, mask2)}
        return batch
