from typing import Dict

import pytest
import torch

from detcon.losses import DetConBLoss


@pytest.fixture
def loss_inputs() -> Dict[str, torch.Tensor]:
    return {
        "pred1": torch.randn(2, 5, 256),
        "pred2": torch.randn(2, 5, 256),
        "target1": torch.randn(2, 5, 256),
        "target2": torch.randn(2, 5, 256),
        "pind1": torch.randint(low=0, high=10, size=(2, 5)),
        "pind2": torch.randint(low=0, high=10, size=(2, 5)),
        "tind1": torch.randint(low=0, high=10, size=(2, 5)),
        "tind2": torch.randint(low=0, high=10, size=(2, 5)),
        "temperature": torch.tensor(0.0),
    }


def test_detconb_loss(loss_inputs: Dict[str, torch.Tensor]) -> None:
    loss = DetConBLoss()
    loss(**loss_inputs)
