from typing import Any, Generator

import pytest
import pytorch_lightning as pl
from _pytest.monkeypatch import MonkeyPatch

from detcon.models import DetConB


def mocked_log(*args: Any, **kwargs: Any) -> None:
    pass


class TestDetConB:
    @pytest.fixture
    def module(self, monkeypatch: Generator[MonkeyPatch, None, None]) -> DetConB:
        module = DetConB()
        monkeypatch.setattr(module, "log", mocked_log)
        return module

    def test_configure_optimizers(self, module: DetConB) -> None:
        module.configure_optimizers()

    def test_training(
        self, ssl_datamodule: pl.LightningDataModule, module: DetConB
    ) -> None:
        batch = next(iter(ssl_datamodule.train_dataloader()))
        batch = ssl_datamodule.on_before_batch_transfer(batch, 0)
        batch = ssl_datamodule.on_after_batch_transfer(batch, 0)
        module.training_step(batch, 0)
