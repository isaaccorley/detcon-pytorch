import pytest
import pytorch_lightning as pl

from detcon.datasets import VOCSSLDataModule

ROOT = "data"


@pytest.fixture(scope="package")
def ssl_datamodule() -> pl.LightningDataModule:
    dm = VOCSSLDataModule(root=ROOT, batch_size=2)
    dm.setup()
    return dm
