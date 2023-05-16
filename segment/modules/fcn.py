import torchvision.models.segmentation as seg
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score

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
        self.backbone = seg.fcn_resnet50(pretrained=True)
        self.backbone.classifier[4] = torch.nn.Conv2d(
            in_channels = self.backbone.classifier[4].in_channels,
            out_channels = self.num_classes,
            kernel_size=self.backbone.classifier[4].kernel_size,
            stride=self.backbone.classifier[4].stride,
            padding=self.backbone.classifier[4].padding,
        )
        self.backbone.aux_classifier[4] = torch.nn.Conv2d(
            in_channels = self.backbone.aux_classifier[4].in_channels,
            out_channels = self.num_classes,
            kernel_size=self.backbone.aux_classifier[4].kernel_size,
            stride=self.backbone.aux_classifier[4].stride,
            padding=self.backbone.aux_classifier[4].padding,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.backbone(x)
        return output

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        x = self.get_input(batch, self.image_key)
        y = batch['label']
        output = self(x)
        loss = self.loss(output['out'], y) + 0.5 * self.loss(output['aux'], y)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = self.get_input(batch, self.image_key)
        y = batch['label']
        logits = self(x)['out']

        preds = nn.functional.softmax(logits, dim=1).argmax(1)
        y_true = y.cpu().numpy().flatten()
        y_pred = preds.cpu().numpy().flatten()

        if self.num_classes == 2:
            task = 'binary'
            y_probs = nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy().flatten()
            precision, recall, _ = precision_recall_curve(y_true, y_probs)
            aupr = auc(recall, precision)
            roc_auc = roc_auc_score(y_true, y_pred)
            average_precision = average_precision_score(y_true, y_probs)
        else:
            task = 'multiclass'
            aupr, roc_auc, average_precision = None, None, None

        jaccard = JaccardIndex(num_classes=self.num_classes,task=task)
        jaccard = jaccard.to(self.device)
        iou = jaccard(preds,y)
        dice = Dice(num_classes=self.num_classes,average='macro')
        dice = dice.to(self.device)
        dice_score = dice(preds,y)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        eps = 1e-6
        pr = tp / (tp + fp + eps)
        se = tp / (tp + fn + eps)
        sp = tn / (tn + fp + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)

        loss = self.loss(logits, y)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/iou", iou, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/dice_score", dice_score, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/pr", pr, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/se", se, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/sp", sp, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if self.num_classes == 2:
            self.log("val/aupr", aupr, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/roc_auc", roc_auc, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/average_precision", average_precision, prog_bar=True, logger=True, on_step=False,
                     on_epoch=True, sync_dist=True)
        return loss

    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        y = batch['label']
        # log["originals"] = x
        out = self(x)['out']
        out = torch.nn.functional.softmax(out,dim=1)
        predict = out.argmax(1)

        log["image"] = x
        log["label"] = y
        log["predict"] = predict
        return log



