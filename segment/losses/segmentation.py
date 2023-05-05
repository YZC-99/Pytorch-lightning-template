# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch
from .dice_loss import *
from .dice_coefficient_loss import dice_loss, build_target


class CE_DiceLoss(nn.Module):
    def __init__(self):
        super(CE_DiceLoss, self).__init__()
    def forward(self,input, target):
        loss_weight = torch.as_tensor([1.0, 2.0], device='cuda:0')
        ce_loss = nn.functional.cross_entropy(input,target.long(),weight=loss_weight)
        dice_target = build_target(target.long(), num_classes=2)
        dc_loss = dice_loss(input, dice_target, multiclass=True)
        return ce_loss + dc_loss