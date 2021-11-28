import torch
import torch.nn as nn
import torch.nn.functional as F


def manual_cross_entropy(
    labels: torch.Tensor, logits: torch.Tensor, weight: torch.Tensor
):
    ce = -weight * torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    return torch.mean(ce)


class DetConBLoss(nn.Module):
    """Modified from https://github.com/deepmind/detcon/blob/main/utils/losses.py."""

    def forward(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
        target1: torch.Tensor,
        target2: torch.Tensor,
        pind1: torch.Tensor,
        pind2: torch.Tensor,
        tind1: torch.Tensor,
        tind2: torch.Tensor,
        temperature: torch.Tensor,
        local_negatives: bool = True,
    ) -> torch.Tensor:
        """Compute the NCE scores from pairs of predictions and targets.

        This implements the batched form of the loss described in
        Section 3.1, Equation 3 in https://arxiv.org/pdf/2103.10957.pdf.

        Args:
            pred1: (b, num_samples, d) the prediction from first view.
            pred2: (b, num_samples, d) the prediction from second view.
            target1: (b, num_samples, d) the projection from first view.
            target2: (b, num_samples, d) the projection from second view.
            pind1: (b, num_samples) mask indices for first view's prediction.
            pind2: (b, num_samples) mask indices for second view's prediction.
            tind1: (b, num_samples) mask indices for first view's projection.
            tind2: (b, num_samples) mask indices for second view's projection.
            temperature: (float) the temperature to use for the NCE loss.
            local_negatives (bool): whether to include local negatives

        Returns:
            A single scalar loss for the XT-NCE objective.
        """
        bs, num_samples, num_features = pred1.shape
        infinity_proxy = 1e9  # Used for masks to proxy a very large number.

        def make_same_obj(ind_0, ind_1):
            same_obj = torch.eq(
                ind_0.reshape([bs, num_samples, 1]), ind_1.reshape([bs, 1, num_samples])
            )
            same_obj = same_obj.unsqueeze(2).to(torch.float)
            return same_obj

        same_obj_aa = make_same_obj(pind1, tind1)
        same_obj_ab = make_same_obj(pind1, tind2)
        same_obj_ba = make_same_obj(pind2, tind1)
        same_obj_bb = make_same_obj(pind2, tind2)

        # L2 normalize the tensors to use for the cosine-similarity
        pred1 = F.normalize(pred1, dim=-1)
        pred2 = F.normalize(pred2, dim=-1)
        target1 = F.normalize(target1, dim=-1)
        target2 = F.normalize(target2, dim=-1)

        labels = F.one_hot(torch.arange(bs), num_classes=bs).to(pred1.device)
        labels = labels.unsqueeze(dim=2).unsqueeze(dim=1)

        # Do our matmuls and mask out appropriately.
        logits_aa = torch.einsum("abk,uvk->abuv", pred1, target1) / temperature
        logits_bb = torch.einsum("abk,uvk->abuv", pred2, target2) / temperature
        logits_ab = torch.einsum("abk,uvk->abuv", pred1, target2) / temperature
        logits_ba = torch.einsum("abk,uvk->abuv", pred2, target1) / temperature

        labels_aa = labels * same_obj_aa
        labels_ab = labels * same_obj_ab
        labels_ba = labels * same_obj_ba
        labels_bb = labels * same_obj_bb

        logits_aa = logits_aa - infinity_proxy * labels * same_obj_aa
        logits_bb = logits_bb - infinity_proxy * labels * same_obj_bb
        labels_aa = 0.0 * labels_aa
        labels_bb = 0.0 * labels_bb

        if not local_negatives:
            logits_aa = logits_aa - infinity_proxy * labels * (1 - same_obj_aa)
            logits_ab = logits_ab - infinity_proxy * labels * (1 - same_obj_ab)
            logits_ba = logits_ba - infinity_proxy * labels * (1 - same_obj_ba)
            logits_bb = logits_bb - infinity_proxy * labels * (1 - same_obj_bb)

        labels_abaa = torch.cat([labels_ab, labels_aa], dim=2)
        labels_babb = torch.cat([labels_ba, labels_bb], dim=2)

        labels_0 = labels_abaa.reshape((bs, num_samples, -1))
        labels_1 = labels_babb.reshape((bs, num_samples, -1))

        num_positives_0 = torch.sum(labels_0, dim=-1, keepdim=True)
        num_positives_1 = torch.sum(labels_1, dim=-1, keepdim=True)

        labels_0 = labels_0 / torch.maximum(num_positives_0, torch.tensor(1.0))
        labels_1 = labels_1 / torch.maximum(num_positives_1, torch.tensor(1.0))

        obj_area_0 = torch.sum(make_same_obj(pind1, pind1), dim=(2, 3))
        obj_area_1 = torch.sum(make_same_obj(pind2, pind2), dim=(2, 3))

        weights_0 = torch.greater(num_positives_0[..., 0], 1e-3).to(torch.float)
        weights_0 = weights_0 / obj_area_0
        weights_1 = torch.greater(num_positives_1[..., 0], 1e-3).to(torch.float)
        weights_1 = weights_1 / obj_area_1

        logits_abaa = torch.cat([logits_ab, logits_aa], dim=2)
        logits_babb = torch.cat([logits_ba, logits_bb], dim=2)

        logits_abaa = logits_abaa.reshape((bs, num_samples, -1))
        logits_babb = logits_babb.reshape((bs, num_samples, -1))

        loss_a = manual_cross_entropy(logits_abaa, labels_0, weight=weights_0)
        loss_b = manual_cross_entropy(logits_babb, labels_1, weight=weights_1)
        loss = loss_a + loss_b
        return loss
