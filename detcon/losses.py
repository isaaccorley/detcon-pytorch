import torch
import torch.nn as nn
import torch.nn.functional as F


class DetConLoss(nn.Module):
    def forward(self, z1, z2, p1, p2) -> torch.Tensor:
        p1, p2 = F.normalize(p1, p=2), p1 = F.normalize(p1, p=2)

        pred2 = helpers.l2_normalize(pred2, axis=-1)
        target1 = helpers.l2_normalize(target1, axis=-1)
        target2 = helpers.l2_normalize(target2, axis=-1)
