import torch
import torchvision.models.segmentation as seg
from omegaconf import OmegaConf
from typing import List,Tuple, Dict, Any, Optional

from .base import BaseModel

class Res50_FCN(BaseModel):
    def __init__(self,
                 image_key: str,
                 in_channels: int,
                 num_classes: int,
                 weight_decay: float,
                 loss: OmegaConf,
                 scheduler: Optional[OmegaConf] = None,
                 ):
        super(Res50_FCN, self).__init__(
                 image_key,
                 in_channels,
                 num_classes,
                 weight_decay,
                 loss,
                 scheduler,)
        self.backbone = seg.fcn_resnet50(pretrained=False)
        self.backbone.classifier[4] = torch.nn.Conv2d(
            in_channels = self.backbone.classifier[4].in_channels,
            out_channels = self.num_classes,
            kernel_size=self.backbone.classifier[4].kernel_size,
            stride=self.backbone.classifier[4].stride,
            padding=self.backbone.classifier[4].padding,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.backbone(x)
        return output['out']



input = torch.randn(2,3,128,128)
