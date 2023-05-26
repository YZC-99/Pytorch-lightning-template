import numpy as np

import torch.nn as nn
import torch.nn.functional as F


from segment.losses.seg.dice_loss import *

def cross_entropy(logits, targets, reduction='mean'):
    loss = F.cross_entropy(logits, targets, reduction=reduction)
    return loss

def binary_cross_entropy(logits, targets, reduction='mean'):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)
    return loss

def mask_cross_entropy(logits, targets, mask, reduction='mean'):
    loss = F.cross_entropy(logits, targets, weight=mask, reduction=reduction)
    return loss


def get_class_grw_weight(class_weight, num_classes=2, exp_scale=1, dataset_type="ddr"):
    """
    Caculate the Generalized Re-weight for Loss Computation
    0	0.0234	1200	300
    1	0.0432	2400	600
    2	0.0345	1800	450
    ...

    """
    assert class_weight.endswith("txt"), class_weight
    if dataset_type is "ddr":
        txt_info = open(class_weight, "r").readlines()
        data_info = dict()
        for idx in range(num_classes):
            item = txt_info[idx+1]
            data = item.strip().split("\t")
            key = data[-1].split(",")[0]
            # assert result[0] == key, "key:{}, result key:{}".format(key,result[0])
            
            data_info[key] = {
                "idx": int(data[0]),
                "ratio": float(data[1]),
                "train": int(data[2]),
                "val": int(data[3]),
            }
        
        ratio = [item['ratio'] for item in data_info.values()]

        class_weight = 1 / (np.array(ratio)**exp_scale)
        class_weight = class_weight / np.sum(class_weight) * num_classes
    else:
        raise NotImplementedError
    return class_weight


class GRWCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 num_classes = 2,
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce',
                 exp_scale=1):
        super(GRWCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_grw_weight(num_classes=num_classes,class_weight=class_weight, exp_scale=exp_scale)

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * nn.functional.cross_entropy(cls_score,label,weight=class_weight)
        # loss_cls = self.loss_weight * self.cls_criterion(
        #     cls_score,
        #     label,
        #     weight=class_weight,
        #     reduction=reduction,
        #     avg_factor=avg_factor,
        #     **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class Dice_GRWCrossEntropyLoss(GRWCrossEntropyLoss):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 num_classes = 2,
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce',
                 exp_scale=1):
        super(GRWCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_grw_weight(num_classes=num_classes,class_weight=class_weight, exp_scale=exp_scale)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper)

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * nn.functional.cross_entropy(cls_score,label,weight=class_weight)
        dc_loss = self.dc(cls_score, label)
        return loss_cls + 1 + dc_loss

    @property
    def loss_name(self):
        return self._loss_name



