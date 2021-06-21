import torch
import torch.nn as nn

from ..registry import HEADS


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@HEADS.register_module
class BarlowTwinsHead(nn.Module):
    """Head for Barlow Twins.

    Args:
        lambd (float): Weight on off-diagonal terms.
            Default: 0.0051.
    """

    def __init__(self, lambd=0.0051, sizes=[2048]):
        super(BarlowTwinsHead, self).__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, z_a, z_b):
        """Forward head.

        Args:
            z_a (Tensor): NxD representation from one randomly augmented image.
            z_b (Tensor): NxD representation from another version of augmentation.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N, D = z_a.shape
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D).cuda()).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool).cuda()] *= self.lambd
        loss = c_diff.sum()

        # # empirical cross-correlation matrix
        # c = self.bn(z_a).T @ self.bn(z_b)
        # # sum the cross-correlation matrix between all gpus
        # c.div_(N)
        # torch.distributed.all_reduce(c)
        # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # off_diag = off_diagonal(c).pow_(2).sum()
        # loss = on_diag + self.lambd * off_diag

        losses = dict()
        losses['loss'] = loss
        return losses
