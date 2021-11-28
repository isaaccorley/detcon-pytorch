import torch
import torch.nn as nn
import torch.nn.functional as F


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
            pred1: the prediction from first view.
            pred2: the prediction from second view.
            target1: the projection from first view.
            target2: the projection from second view.
            pind1: mask indices for first view's prediction.
            pind2: mask indices for second view's prediction.
            tind1: mask indices for first view's projection.
            tind2: mask indices for second view's projection.
            temperature: the temperature to use for the NCE loss.
            local_negatives (bool): whether to include local negatives

        Returns:
            A single scalar loss for the XT-NCE objective.
        """
        batch_size = pred1.shape[0]
        num_rois = pred1.shape[1]
        infinity_proxy = 1e9  # Used for masks to proxy a very large number.

        def make_same_obj(ind_0, ind_1):
            same_obj = torch.equal(
                ind_0.reshape([batch_size, num_rois, 1]),
                ind_1.reshape([batch_size, 1, num_rois]),
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

        target1_large = target1
        target2_large = target2
        labels_local = F.one_hot(torch.arange(batch_size), num_classes=batch_size)
        labels_ext = F.one_hot(torch.arange(batch_size), num_classes=batch_size * 2)

        labels_local = labels_local.unsqueeze(dim=2).unsqueeze(dim=1)
        labels_ext = labels_ext.unsqueeze(dim=2).unsqueeze(dim=1)

        # Do our matmuls and mask out appropriately.
        logits_aa = torch.einsum("abk,uvk->abuv", pred1, target1_large) / temperature
        logits_bb = torch.einsum("abk,uvk->abuv", pred2, target2_large) / temperature
        logits_ab = torch.einsum("abk,uvk->abuv", pred1, target2_large) / temperature
        logits_ba = torch.einsum("abk,uvk->abuv", pred2, target1_large) / temperature

        labels_aa = labels_local * same_obj_aa
        labels_ab = labels_local * same_obj_ab
        labels_ba = labels_local * same_obj_ba
        labels_bb = labels_local * same_obj_bb

        logits_aa = logits_aa - infinity_proxy * labels_local * same_obj_aa
        logits_bb = logits_bb - infinity_proxy * labels_local * same_obj_bb
        labels_aa = 0.0 * labels_aa
        labels_bb = 0.0 * labels_bb

        if not local_negatives:
            logits_aa = logits_aa - infinity_proxy * labels_local * (1 - same_obj_aa)
            logits_ab = logits_ab - infinity_proxy * labels_local * (1 - same_obj_ab)
            logits_ba = logits_ba - infinity_proxy * labels_local * (1 - same_obj_ba)
            logits_bb = logits_bb - infinity_proxy * labels_local * (1 - same_obj_bb)

        labels_abaa = torch.cat([labels_ab, labels_aa], dim=2)
        labels_babb = torch.cat([labels_ba, labels_bb], dim=2)

        labels_0 = labels_abaa.reshape((batch_size, num_rois, -1))
        labels_1 = labels_babb.reshape((batch_size, num_rois, -1))

        num_positives_0 = torch.sum(labels_0, dim=-1, keepdim=True)
        num_positives_1 = torch.sum(labels_1, dim=-1, keepdim=True)

        labels_0 = labels_0 / torch.maximum(num_positives_0, 1)
        labels_1 = labels_1 / torch.maximum(num_positives_1, 1)

        obj_area_0 = torch.sum(make_same_obj(pind1, pind1), dim=(2, 3))
        obj_area_1 = torch.sum(make_same_obj(pind2, pind2), dim=(2, 3))

        weights_0 = torch.greater(num_positives_0[..., 0], 1e-3).to(torch.float)
        weights_0 = weights_0 / obj_area_0
        weights_1 = torch.greater(num_positives_1[..., 0], 1e-3).to(torch.float)
        weights_1 = weights_1 / obj_area_1

        logits_abaa = torch.cat([logits_ab, logits_aa], dim=2)
        logits_babb = torch.cat([logits_ba, logits_bb], dim=2)

        logits_abaa = logits_abaa.reshape((batch_size, num_rois, -1))
        logits_babb = logits_babb.reshape((batch_size, num_rois, -1))

        loss_a = F.cross_entropy(logits_abaa, labels_0, weight=weights_0)
        loss_b = F.cross_entropy(logits_babb, labels_1, weight=weights_1)
        loss = loss_a + loss_b
        return loss
