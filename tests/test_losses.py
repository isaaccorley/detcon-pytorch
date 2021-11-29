from typing import Dict

import pytest
import torch

from detcon.losses import DetConBLoss


@pytest.fixture
def loss_inputs() -> Dict[str, torch.Tensor]:
    num_samples = 16
    batch_size = 2
    dim = 256
    num_classes = 21
    return {
        "pred1": torch.randn(batch_size, num_samples, dim),
        "pred2": torch.randn(batch_size, num_samples, dim),
        "target1": torch.randn(batch_size, num_samples, dim),
        "target2": torch.randn(batch_size, num_samples, dim),
        "pind1": torch.randint(low=0, high=num_classes, size=(batch_size, num_samples)),
        "pind2": torch.randint(low=0, high=num_classes, size=(batch_size, num_samples)),
        "tind1": torch.randint(low=0, high=num_classes, size=(batch_size, num_samples)),
        "tind2": torch.randint(low=0, high=num_classes, size=(batch_size, num_samples)),
    }


def test_detconb_loss(loss_inputs: Dict[str, torch.Tensor]) -> None:
    loss = DetConBLoss()
    loss(**loss_inputs)
