import torchvision.models.segmentation as seg
import torch
import torch.nn as nn

import numpy as np
from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score,confusion_matrix
from segment.utils.general import initialize_from_config
from omegaconf import OmegaConf
from typing import List,Tuple, Dict, Any, Optional

from segment.modules.fcn import Res50_FCN
from segment.modules.unet import UNet

class FCN_Unet(Res50_FCN):
    def __init__(self,
                 image_key: str,
                 in_channels: int,
                 num_classes: int,
                 weight_decay: float,
                 loss: OmegaConf,
                 scheduler: Optional[OmegaConf] = None,
                 ckpt_path: str = None,
                 ignore_keys: list = [],
                 concat: bool = False,
                 ):
        super(FCN_Unet, self).__init__(
                 image_key,
                 in_channels,
                 num_classes,
                 weight_decay,
                 loss,
                 scheduler,
                 ckpt_path,
                 ignore_keys,
        )
        self.concat = concat
        for name, param in self.named_parameters():
            if name == 'backbone':
                param.requires_grad = False

        self.unet = UNet(in_channels=1,
                         num_classes=self.num_classes,
                         weight_decay=self.weight_decay,
                         loss=self.loss)
        self.color_map = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128]}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.backbone(x)['out']
        if self.num_classes == 2:
            stage1_output = torch.nn.functional.sigmoid(logits)
        else:
            stage1_output = torch.nn.functional.softmax(logits,dim=1)

        stage2_input = torch.argmax(stage1_output)
        output = self.unet(stage2_input)
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
        loss = self.loss(output, y)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = self.get_input(batch, self.image_key)
        y = batch['label']
        logits = self(x)

        preds = nn.functional.softmax(logits, dim=1).argmax(1)
        y_true = y.cpu().numpy().flatten()
        y_pred = preds.cpu().numpy().flatten()

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

            # Log metrics
            self.log(f"val/class_{i}/dice_score", dice_score, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True)
            self.log(f"val/class_{i}/se", se, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val/class_{i}/sp", sp, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val/class_{i}/acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True)

        loss = self.loss(logits, y)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        y = batch['label']
        # log["originals"] = x
        out = self(x)
        out = torch.nn.functional.softmax(out,dim=1)
        predict = out.argmax(1)

        # Convert labels and predictions to color images.
        y_color = torch.zeros(y.size(0), 3, y.size(1), y.size(2), device=self.device)
        predict_color = torch.zeros(predict.size(0), 3, predict.size(1), predict.size(2), device=self.device)
        for label, color in self.color_map.items():
            mask_y = (y == int(label))
            mask_p = (predict == int(label))
            for i in range(3):  # apply each channel individually
                y_color[mask_y, i] = color[i]
                predict_color[mask_p, i] = color[i]

        log["image"] = x
        log["label"] = y_color
        log["predict"] = predict_color
        return log

class End2End_FCN_Unet(Res50_FCN):
    def __init__(self,
                 image_key: str,
                 in_channels: int,
                 base_c: int,
                 num_classes: int,
                 weight_decay: float,
                 losses: OmegaConf,
                 scheduler: Optional[OmegaConf] = None,
                 ckpt_path: str = None,
                 ignore_keys: list = [],
                 concat: bool = False,
                 bilinear: bool = True
                 ):
        super(End2End_FCN_Unet, self).__init__(
                 image_key,
                 in_channels,
                 num_classes,
                 weight_decay,
                 # losses.stage1_loss,
                 scheduler,
                 ignore_keys,
        )
        self.concat = concat
        stage2_input_channels = self.num_classes + in_channels if self.concat else self.num_classes

        self.unet = UNet(
                         image_key=self.image_key,
                         in_channels=stage2_input_channels,
                         base_c = base_c,
                         num_classes=self.num_classes,
                         weight_decay=self.weight_decay,
                         bilinear = bilinear,
                        )

        self.loss1 = initialize_from_config(losses.stage1_loss)
        self.loss2 = initialize_from_config(losses.stage1_loss)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.color_map = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128]}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(x)
        stage1_out = out['out']
        if self.concat:
            stage2_input = torch.concat(x,stage1_out,dim=1)
        else :
            stage2_input = stage1_out
        logits = self.unet(stage2_input)
        return logits,out

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        x = self.get_input(batch, self.image_key)
        y = batch['label']
        logits,stage1_out = self(x)

        loss1 = self.loss1(stage1_out['out'], y) + 0.5 * self.loss2(stage1_out['aux'], y)
        loss2 = self.loss2(logits, y)
        loss = loss1 + loss2
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = self.get_input(batch, self.image_key)
        y = batch['label']
        logits, stage1_out = self(x)

        preds = nn.functional.softmax(logits, dim=1).argmax(1)
        y_true = y.cpu().numpy().flatten()
        y_pred = preds.cpu().numpy().flatten()

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

            # Log metrics
            self.log(f"val/class_{i}/dice_score", dice_score, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True)
            self.log(f"val/class_{i}/se", se, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val/class_{i}/sp", sp, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val/class_{i}/acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True)

        loss1 = self.loss1(stage1_out['out'], y) + 0.5 * self.loss2(stage1_out['aux'], y)
        loss2 = self.loss2(logits, y)
        loss = loss1 + loss2
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        y = batch['label']
        # log["originals"] = x
        out = self(x)[0]
        out = torch.nn.functional.softmax(out,dim=1)
        predict = out.argmax(1)

        # Convert labels and predictions to color images.
        y_color = torch.zeros(y.size(0), 3, y.size(1), y.size(2), device=self.device)
        predict_color = torch.zeros(predict.size(0), 3, predict.size(1), predict.size(2), device=self.device)
        for label, color in self.color_map.items():
            mask_y = (y == int(label))
            mask_p = (predict == int(label))
            for i in range(3):  # apply each channel individually
                y_color[mask_y, i] = color[i]
                predict_color[mask_p, i] = color[i]

        log["image"] = x
        log["label"] = y_color
        log["predict"] = predict_color
        return log


