from detcon.datasets import VOCSegmentationDataModule, VOCSSLDataModule

ROOT = "data"
SHAPE = (224, 224)


def test_voc_segmentation_datamodule() -> None:
    dm = VOCSegmentationDataModule(root=ROOT)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    batch = dm.on_before_batch_transfer(batch, 0)
    batch = dm.on_after_batch_transfer(batch, 0)

    batch = next(iter(dm.val_dataloader()))
    batch = dm.on_before_batch_transfer(batch, 0)
    batch = dm.on_after_batch_transfer(batch, 0)

    batch = next(iter(dm.test_dataloader()))
    batch = dm.on_before_batch_transfer(batch, 0)
    batch = dm.on_after_batch_transfer(batch, 0)


def test_voc_ssl_segmentation_datamodule() -> None:
    dm = VOCSSLDataModule(root=ROOT)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    batch = dm.on_before_batch_transfer(batch, 0)
    batch = dm.on_after_batch_transfer(batch, 0)
