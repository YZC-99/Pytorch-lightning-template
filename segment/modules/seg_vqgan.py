import torchvision.models.segmentation as seg
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score

from omegaconf import OmegaConf
from typing import List,Tuple, Dict, Any, Optional

from segment.modules.base import BaseModel
from segment.modules.vqgan.diff_model import Encoder,Decoder
from segment.modules.vqgan.quantize import VectorQuantizer2 as VectorQuantizer

class Res50_FCN_VQGAN(BaseModel):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 image_key: str,
                 in_channels: int,
                 num_classes: int,
                 weight_decay: float,
                 loss: OmegaConf,
                 ckpt_path: str = None,
                 ignore_keys: list = [],
                 scheduler: Optional[OmegaConf] = None,
                 remap=None,
                 sane_index_shape=False,
                 ):
        super(Res50_FCN_VQGAN, self).__init__(
                 image_key,
                 in_channels,
                 num_classes,
                 weight_decay,
                 loss,
                 scheduler,)

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # 冻结参数
        for name, param in self.named_parameters():
            if name != 'encoder':
                param.requires_grad = False

        self.image_key = image_key

        # 重新设置encoder
        self.encoder = seg.fcn_resnet50(pretrained=True).backbone

    def encode(self, x):

        h = self.encoder(x)['out']
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        quant, diff, _ = self.encode(x)
        dec = self.decode(quant)
        return dec







