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

    def __init__(self, lambd=0.0051, sizes=[2048], dimension="D"):
        super(BarlowTwinsHead, self).__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        assert dimension in ("D", "N")
        self.dimension = dimension

    def forward(self, z_a, z_b):
        """Forward head.

        Args:
            z_a (Tensor): NxD representation from one randomly augmented image.
            z_b (Tensor): NxD representation from another version of augmentation.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N, D = z_a.shape

        # # normalize repr. along the batch dimension
        # z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        # z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD
        # if self.dimension == 'D':
        #     # cross-correlation matrix
        #     c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        #     # loss
        #     c_diff = (c - torch.eye(D).cuda()).pow(2) # DxD
        #     # multiply off-diagonal elems of c_diff by lambda
        #     c_diff[~torch.eye(D, dtype=bool).cuda()] *= self.lambd
        # elif self.dimension == 'N':
        #     # auto-correlation matrix
        #     c = torch.mm(z_a_norm, z_b_norm.T) / N # NxN
        #     # loss
        #     c_diff = (c - torch.eye(N).cuda()).pow(2) # NxN
        #     # multiply off-diagonal elems of c_diff by lambda
        #     c_diff[~torch.eye(N, dtype=bool).cuda()] *= self.lambd
        # loss = c_diff.sum()

        # empirical cross-correlation matrix
        c = self.bn(z_a).T @ self.bn(z_b)
        # sum the cross-correlation matrix between all gpus
        c.div_(N)
        torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        losses = dict()
        losses["loss"] = loss
        return losses


@HEADS.register_module
class BarlowTwinsHeadV2(nn.Module):
    """Head for Barlow Twins.

    Args:
        lambd (float): Weight on off-diagonal terms.
            Default: 0.0051.
    """

    def __init__(
        self, lambd=0.0051, dimension=2048, norm="fro", rank_lambd=1, supcon=False
    ):
        super(BarlowTwinsHeadV2, self).__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(dimension, affine=False)
        self.norm = norm
        self.rank_lambd = rank_lambd
        self.supcon = supcon

    def forward(self, z_a, z_b, gt_label):
        """Forward head.

        Args:
            z_a (Tensor): NxD representation from one randomly augmented image.
            z_b (Tensor): NxD representation from another version of augmentation.
            gt_label (Tensor): Ground-truth labels.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N, D = z_a.shape
        # empirical cross-correlation matrix
        c = self.bn(z_a).T @ self.bn(z_b)  # DxD
        # sum the cross-correlation matrix between all gpus
        c.div_(N)
        torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        c_ = self.bn(z_a) @ self.bn(z_b).T  # NxN
        # FIXME: torch.norm is deprecated and may be removed in a future PyTorch release.
        rank = self.rank_lambd * torch.norm(c_, p=self.norm)
        loss -= rank

        if self.supcon:
            gt_label = torch.flatten(gt_label).view(-1, 1)
            mask = torch.eq(gt_label, gt_label.T).float()
            # TODO: label mask

        losses = dict()
        losses["loss"] = loss
        return losses


@HEADS.register_module
class BtSimClrHead(nn.Module):
    """Head for Barlow Twins with SimCLR.

    Args:
        lambd (float): Weight on off-diagonal terms.
            Default: 0.0051.
    """

    def __init__(self, lambd=0.0051, dimension=2048, temperature=0.1):
        super(BtSimClrHead, self).__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(dimension, affine=False)
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, z_a, z_b, pos, neg):
        """Forward head.

        Args:
            z_a (Tensor): NxD representation from one randomly augmented image.
            z_b (Tensor): NxD representation from another version of augmentation.
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N, D = z_a.shape

        # empirical cross-correlation matrix
        c = self.bn(z_a).T @ self.bn(z_b)
        # sum the cross-correlation matrix between all gpus
        c.div_(N)
        torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        bt_loss = on_diag + self.lambd * off_diag

        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        simclr_loss = self.criterion(logits, labels)

        losses = dict()
        losses["loss"] = bt_loss + simclr_loss
        return losses
