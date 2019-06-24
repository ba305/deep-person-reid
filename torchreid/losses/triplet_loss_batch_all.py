from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class TripletLossBatchAll(nn.Module):
    """Triplet loss WITHOUT hard positive/negative mining (i.e., "batch all" rather than
    "batch hard.") It calculates the loss based on ALL combinations of possible triplets
    rather than just using the hardest positive and the hardest negative for each anchor.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLossBatchAll, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        # Note: the top few lines of code below could be calculated with
        # the scipy.spatial.distance.cdist function
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find ALL combinations of triplets (i.e., all possible
        # groupings with a positive and a negative)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            pos = dist[i][mask[i]]
            neg = dist[i][mask[i] == 0]
            for p in pos:
                for n in neg:
                    dist_ap.append(p.unsqueeze(0))
                    dist_an.append(n.unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)