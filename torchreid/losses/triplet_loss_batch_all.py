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

        # Must ignore diagonals (set them equal to 7, which is just an arbitrary number)
        # so that the anchor and positive will not be the exact same image
        for i in range(len(dist)):
            mask[i][i] = 7

        dist_ap, dist_an = [], []
        for i in range(n):
            pos = dist[i][mask[i] == 1]
            neg = dist[i][mask[i] == 0]

            num_pos, num_neg = len(pos), len(neg)

            # Find every possible pair between the positive and negatives, in order
            # to make every possible triplet. This is equivalent to using a double
            # for loop to loop through pos and neg, but faster.
            pos = pos.view(-1, 1).repeat(1, num_neg).view(1,-1).squeeze()
            neg = neg.repeat(1, num_pos).squeeze()
            dist_ap.extend(pos.tolist())
            dist_an.extend(neg.tolist())

        dist_ap = torch.tensor(dist_ap).cuda()
        dist_an = torch.tensor(dist_an).cuda()

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)