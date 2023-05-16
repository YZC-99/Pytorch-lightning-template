# from base import BaseModel
import torch
import torch.nn as nn
import torchvision.models.segmentation as seg
from omegaconf import OmegaConf
from typing import List,Tuple, Dict, Any, Optional

from segment.modules.fcn import Res50_FCN

class IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x


class Align_Res50_FCN(Res50_FCN):
    def __init__(self,
                 image_key: str,
                 in_channels: int,
                 num_classes: int,
                 weight_decay: float,
                 loss: OmegaConf,
                 scheduler: Optional[OmegaConf] = None,
                 ):
        super(Align_Res50_FCN, self).__init__(
                 image_key,
                 in_channels,
                 num_classes,
                 weight_decay,
                 loss,
                 scheduler,)

        self.backbone = seg.fcn_resnet50(pretrained=True)
        self.backbone.aux_classifier[4] = torch.nn.Conv2d(
            in_channels=self.backbone.aux_classifier[4].in_channels,
            out_channels=self.num_classes,
            kernel_size=self.backbone.aux_classifier[4].kernel_size,
            stride=self.backbone.aux_classifier[4].stride,
            padding=self.backbone.aux_classifier[4].padding,
        )

        self.classifier = torch.nn.Conv2d(
            in_channels=self.backbone.classifier[4].in_channels,
            out_channels=self.num_classes,
            kernel_size=self.backbone.classifier[4].kernel_size,
            stride=self.backbone.classifier[4].stride,
            padding=self.backbone.classifier[4].padding,
        )
        self.confidence_layer = nn.Sequential(
            nn.Conv2d(self.backbone.classifier[0].out_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.backbone.classifier[4] = IdentityModule()
        # Magnitude and Margin of DisAlign+
        self.logit_scale = nn.Parameter(torch.ones(1, self.num_classes, 1, 1))
        self.logit_bias = nn.Parameter(torch.zeros(1, self.num_classes, 1, 1))
        # Confidence function，类似于linear层，是一个1*1得卷积

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.backbone(x)

        output_out = output['out']
        confidence = self.confidence_layer(output_out).sigmoid()
        output_out = self.classifier(output_out)
        # only adjust the foreground classification scores
        scores_tmp = confidence * (output_out * self.logit_scale + self.logit_bias)
        output_out = scores_tmp + (1 - confidence) * output_out

        output['out'] = output_out
        return output







