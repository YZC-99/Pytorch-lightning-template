import torchvision.models.segmentation as seg
import torch
import torch.nn as nn

import numpy as np
from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score,confusion_matrix

from utils.general import initialize_from_config
from omegaconf import OmegaConf
from typing import List,Tuple, Dict, Any, Optional

from .base import BaseModel

class Res50_FCN(BaseModel):
    def __init__(self,
                 image_key: str,
                 in_channels: int,
                 num_classes: int,
                 weight_decay: float,
                 loss: Optional[OmegaConf] = None,
                 scheduler: Optional[OmegaConf] = None,
                 ckpt_path: str = None,
                 ignore_keys: list = [],
                 ):
        super(Res50_FCN, self).__init__(
                 image_key,
                 in_channels,
                 num_classes,
                 weight_decay,
                 scheduler,
        )
        if loss is not None:
            self.loss = initialize_from_config(loss)
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
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.color_map = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128]}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.backbone(x)
        return output
    
    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        optimizers = [torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)]

        total_epochs = self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=total_epochs)

        schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        ]

        return optimizers, schedulers

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

        jaccard = JaccardIndex(num_classes=self.num_classes,task='binary' if self.num_classes ==  2 else 'multiclass')
        jaccard = jaccard.to(self.device)
        iou = jaccard(preds, y)

        dice = Dice(num_classes=self.num_classes, average='macro')
        dice = dice.to(self.device)
        dice_score = dice(preds, y)

        if self.num_classes == 2:
            y_probs = nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy().flatten()
            precision, recall, _ = precision_recall_curve(y_true, y_probs)
            aupr = auc(recall, precision)
            roc_auc = roc_auc_score(y_true, y_pred)
            average_precision = average_precision_score(y_true, y_probs)
            self.log("val/aupr", aupr, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/roc_auc", roc_auc, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/average_precision", average_precision, prog_bar=True, logger=True, on_step=False,
                     on_epoch=True, sync_dist=True)

        # Calculate metrics for each class
        for i in range(self.num_classes):
            binary_y_true = (y_true == i)
            binary_y_pred = (y_pred == i)

            # 计算dice、iou
            jaccard_i = JaccardIndex(num_classes=2, task='binary')
            # jaccard_i = jaccard_i.to(self.device)
            iou_i = jaccard_i(torch.from_numpy(binary_y_pred), torch.from_numpy(binary_y_true))

            dice_i = Dice(num_classes=2, average='macro')
            # dice_i = dice_i.to(self.device)
            dice_score_i = dice_i(torch.from_numpy(binary_y_pred), torch.from_numpy(binary_y_true))


            conf_matrix = confusion_matrix(binary_y_true, binary_y_pred)
            # Ensure the confusion matrix is 2x2.
            if conf_matrix.size == 1:
                conf_matrix = conf_matrix.reshape((1, 1))
                conf_matrix = np.pad(conf_matrix, ((0, 1), (0, 1)), 'constant')

            tn, fp, fn, tp = conf_matrix.ravel()

            eps = 1e-6
            se = tp / (tp + fn + eps)
            sp = tn / (tn + fp + eps)
            acc = (tp + tn) / (tp + tn + fp + fn + eps)

            # 计算AUC_PR、AUC_ROC
            y_probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
            y_true_i = (y_true == i)
            y_probs_i = y_probs[:, i].flatten()

            if len(np.unique(y_true_i)) > 1:
                precision, recall, _ = precision_recall_curve(y_true_i, y_probs_i)
                auc_pr_i = auc(recall, precision)
                auc_roc_i = roc_auc_score(y_true_i, y_probs_i)
                self.log(f"val/class_{i}/auc_pr", auc_pr_i, prog_bar=True, logger=True, on_step=False,
                         on_epoch=True, sync_dist=True)
                self.log(f"val/class_{i}/auc_roc", auc_roc_i, prog_bar=True, logger=True, on_step=False,
                         on_epoch=True, sync_dist=True)


            # Log metrics

            self.log(f"val/class_{i}/iou", iou_i, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val/class_{i}/dice", dice_score_i, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val/class_{i}/se", se, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val/class_{i}/sp", sp, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val/class_{i}/acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True)

        self.log(f"val/dice_score", dice_score, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log("val/iou", iou, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        loss = self.loss(logits, y)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        y = batch['label']
        # log["originals"] = x
        out = self(x)['out']
        out = torch.nn.functional.softmax(out,dim=1)
        predict = out.argmax(1)

        y_color, predict_color = self.gray2rgb(y, predict)
        log["image"] = x
        log["label"] = y_color
        log["predict"] = predict_color
        return log



