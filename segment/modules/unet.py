from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List,Tuple, Dict, Any, Optional
from omegaconf import OmegaConf
from segment.utils.general import initialize_from_config
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from torchmetrics import JaccardIndex,Dice

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Encoder(pl.LightningModule):
    def __init__(self,in_channels,base_c,bilinear=True):
        super(Encoder,self).__init__()
        factor = 2 if bilinear else 1
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

    def forward(self,x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return {"x1":x1,"x2":x2,"x3":x3,"x4":x4,"x5":x5}

class Decoder(pl.LightningModule):
    def __init__(self,base_c,bilinear=True):
        super(Decoder,self).__init__()
        factor = 2 if bilinear else 1
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)


    def forward(self,x_dict):
        x = self.up1(x_dict['x5'],x_dict['x4'])
        x = self.up2(x,x_dict['x3'])
        x = self.up3(x,x_dict['x2'])
        x = self.up4(x,x_dict['x1'])
        return x

class UNet(pl.LightningModule):
    def __init__(self,
                 image_key: str,
                 in_channels: int,
                 num_classes: int,
                 bilinear: bool,
                 base_c: int,
                 weight_decay: float,
                 loss: OmegaConf,
                 scheduler: Optional[OmegaConf] = None,
                 ):
        super(UNet, self).__init__()
        self.weight_decay = weight_decay
        self.image_key = image_key
        self.loss = initialize_from_config(loss)
        self.scheduler = scheduler
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.encoder = Encoder(in_channels,base_c,bilinear=True)
        self.decoder = Decoder(base_c,bilinear=True)
        self.out_conv = OutConv(base_c, num_classes)


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_dict = self.encoder(x)
        x = self.decoder(x_dict)
        logits = self.out_conv(x)
        return logits

    def init_from_ckpt(self,path: str,ignore_keys: List[str] = list()):
        sd = torch.load(path,map_location='cpu')['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def get_input(self, batch: Tuple[Any, Any], key: str = 'image') -> Any:
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()
        return x.contiguous()

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        x = self.get_input(batch, self.image_key)
        y = batch['label']
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = self.get_input(batch, self.image_key)
        y = batch['label']
        logits = self(x)
        preds = nn.functional.softmax(logits).argmax(1)
        jaccard = JaccardIndex(num_classes=2,task='binary')
        jaccard = jaccard.to(self.device)
        iou = jaccard(preds,y)

        dice = Dice(num_classes=2,average='macro')
        dice = dice.to(self.device)
        dice_score = dice(preds,y)

        loss = self.loss(logits, y)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/iou", iou, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/dice_score", dice_score, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss


    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        # optimizers = [torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)]
        optimizers = [torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)]
        # optimizers = [torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)]

        warmup = True
        warmup_epochs = 1
        num_step = 9
        warmup_factor = 1e-3
        epochs = self.trainer.max_epochs
        lr_decay_rate = 0.8

        def f(x):
            """
             根据step数返回一个学习率倍率因子，
             注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
             """
            if warmup is True and x <= (warmup_epochs * num_step):
                alpha = float(x) / (warmup_epochs * num_step)
                # warmup过程中lr倍率因子从warmup_factor -> 1
                return warmup_factor * (1 - alpha) + alpha
            else:
                # warmup后lr倍率因子从1 -> 0
                # 参考deeplab_v2: Learning rate policy
                return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

        def lr_scheduler_fn(epoch,lr):
            if epoch < warmup_epochs:
                return (epoch / warmup_epochs) * lr
            else:
                return lr * (lr_decay_rate ** (epoch - warmup_epochs))

        schedulers = [
            {
                # 'scheduler': lr_scheduler.LambdaLR(optimizers[0],lr_lambda=lambda epoch: lr_scheduler_fn(epoch, lr)),
                'scheduler': lr_scheduler.LambdaLR(optimizers[0],lr_lambda=f),
                'interval': 'step',
                'frequency': 1
            }
        ]

        return optimizers, schedulers



    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        y = batch['label']
        # log["originals"] = x
        out = self(x)
        out = torch.nn.functional.softmax(out,dim=1)
        predict = out.argmax(1)
        log["image"] = x
        log["label"] = y
        log["predict"] = predict
        return log


