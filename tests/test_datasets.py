import torch

from detcon.datasets import VOCSegmentationDataModule

ROOT = "/mnt/e/data/"


def test_voc_segmentation_datamodule() -> None:
    dm = VOCSegmentationDataModule(
        root=ROOT,
        batch_size=1,
        num_workers=1,
        prefetch_factor=1,
        pin_memory=False,
    )
    dm.setup()

    # Train
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (1, 3, 256, 256)
    assert y.shape == (1, 1, 256, 256)
    assert x.dtype == torch.float32
    assert y.dtype == torch.uint8

    # Val
    x, y = next(iter(dm.val_dataloader()))
    assert x.shape == (1, 3, 256, 256)
    assert y.shape == (1, 1, 256, 256)
    assert x.dtype == torch.float32
    assert y.dtype == torch.uint8

    # Test
    x, y = next(iter(dm.test_dataloader()))
    assert x.shape == (1, 3, 256, 256)
    assert y.shape == (1, 1, 256, 256)
    assert x.dtype == torch.float32
    assert y.dtype == torch.uint8
