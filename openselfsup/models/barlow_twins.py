import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .utils import GatherLayer


@MODELS.register_module
class BarlowTwins(nn.Module):
    """Barlow Twins.

    Implementation of "Barlow Twins: Self-Supervised Learning via Redundancy Reduction
	(https://arxiv.org/abs/2103.03230v1)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(BarlowTwins, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log("load model from: {}".format(pretrained), logger="root")
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear="kaiming")

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, "Input must have 5 dims, got: {}".format(img.dim())
        img = img.reshape(img.size(0) * 2, img.size(2), img.size(3), img.size(4))
        x = self.forward_backbone(img)  # [2NxDx?x?]
        z = self.neck(x)[0]  # 2NxDx1x1
        z = z.squeeze()  # 2NxD

        N = z.size(0)
        mask_a = torch.tensor([i for i in range(N) if i % 2 == 0])
        mask_b = torch.tensor([i for i in range(N) if i % 2 != 0])
        z_a = z[mask_a, :]
        z_b = z[mask_b, :]

        losses = self.head(z_a, z_b)
        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode="train", **kwargs):
        if mode == "train":
            return self.forward_train(img, **kwargs)
        elif mode == "test":
            return self.forward_test(img, **kwargs)
        elif mode == "extract":
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class BarlowTwinsV2(nn.Module):
    """Barlow Twins.

    Implementation of "Barlow Twins: Self-Supervised Learning via Redundancy Reduction
	(https://arxiv.org/abs/2103.03230v1)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(BarlowTwinsV2, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log("load model from: {}".format(pretrained), logger="root")
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear="kaiming")

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): Ground-truth labels.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, "Input must have 5 dims, got: {}".format(img.dim())
        img = img.reshape(img.size(0) * 2, img.size(2), img.size(3), img.size(4))
        x = self.forward_backbone(img)  # [2NxDx?x?]
        z = self.neck(x)[0]  # 2NxDx1x1
        z = z.squeeze()  # 2NxD

        N = z.size(0)
        mask_a = torch.tensor([i for i in range(N) if i % 2 == 0])
        mask_b = torch.tensor([i for i in range(N) if i % 2 != 0])
        z_a = z[mask_a, :]
        z_b = z[mask_b, :]

        losses = self.head(z_a, z_b, gt_label)
        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode="train", **kwargs):
        if mode == "train":
            return self.forward_train(img, **kwargs)
        elif mode == "test":
            return self.forward_test(img, **kwargs)
        elif mode == "extract":
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class BtSimClr(nn.Module):
    """Barlow Twins.

    Implementation of "Barlow Twins: Self-Supervised Learning via Redundancy Reduction
	(https://arxiv.org/abs/2103.03230v1)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(BtSimClr, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log("load model from: {}".format(pretrained), logger="root")
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear="kaiming")

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    @staticmethod
    def _create_buffer(N):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (
            torch.arange(N * 2).cuda(),
            2
            * torch.arange(N, dtype=torch.long)
            .unsqueeze(1)
            .repeat(1, 2)
            .view(-1, 1)
            .squeeze()
            .cuda(),
        )
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, "Input must have 5 dims, got: {}".format(img.dim())
        img = img.reshape(img.size(0) * 2, img.size(2), img.size(3), img.size(4))
        x = self.forward_backbone(img)  # [2NxDx?x?]
        z = self.neck(x)[0]  # 2NxDx1x1
        z = z.squeeze()  # 2NxD

        N = z.size(0)
        mask_a = torch.tensor([i for i in range(N) if i % 2 == 0])
        mask_b = torch.tensor([i for i in range(N) if i % 2 != 0])
        z_a = z[mask_a, :]
        z_b = z[mask_b, :]

        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)

        losses = self.head(z_a, z_b, positive, negative)
        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode="train", **kwargs):
        if mode == "train":
            return self.forward_train(img, **kwargs)
        elif mode == "test":
            return self.forward_test(img, **kwargs)
        elif mode == "extract":
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
