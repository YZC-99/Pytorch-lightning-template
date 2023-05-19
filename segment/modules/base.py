import torch
import torch.nn as nn
from typing import List,Tuple, Dict, Any, Optional
from omegaconf import OmegaConf
from segment.utils.general import initialize_from_config
from torch.optim import lr_scheduler
import pytorch_lightning as pl

from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score


class BaseModel(pl.LightningModule):
    def __init__(self,
                 image_key: str,
                 in_channels: int,
                 num_classes: int,
                 weight_decay: float,
                 loss: OmegaConf,
                 scheduler: Optional[OmegaConf] = None,
                 ):
        super(BaseModel, self).__init__()
        self.weight_decay = weight_decay
        self.image_key = image_key
        self.loss = initialize_from_config(loss)
        self.scheduler = scheduler
        self.in_channels = in_channels
        self.num_classes = num_classes


        self.color_map = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128]}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

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
        # lr = self.learning_rate
        #
        # # optimizers = [torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)]
        # optimizers = [torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)]
        # # optimizers = [torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)]
        #
        # warmup = True
        # warmup_epochs = 1
        # num_step = 9
        # warmup_factor = 1e-3
        # epochs = self.trainer.max_epochs
        # lr_decay_rate = 0.8
        #
        # def f(x):
        #     """
        #      根据step数返回一个学习率倍率因子，
        #      注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        #      """
        #     if warmup is True and x <= (warmup_epochs * num_step):
        #         alpha = float(x) / (warmup_epochs * num_step)
        #         # warmup过程中lr倍率因子从warmup_factor -> 1
        #         return warmup_factor * (1 - alpha) + alpha
        #     else:
        #         # warmup后lr倍率因子从1 -> 0
        #         # 参考deeplab_v2: Learning rate policy
        #         return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
        #
        # def lr_scheduler_fn(epoch,lr):
        #     if epoch < warmup_epochs:
        #         return (epoch / warmup_epochs) * lr
        #     else:
        #         return lr * (lr_decay_rate ** (epoch - warmup_epochs))
        #
        # schedulers = [
        #     {
        #         # 'scheduler': lr_scheduler.LambdaLR(optimizers[0],lr_lambda=lambda epoch: lr_scheduler_fn(epoch, lr)),
        #         'scheduler': lr_scheduler.LambdaLR(optimizers[0],lr_lambda=f),
        #         'interval': 'step',
        #         'frequency': 1
        #     }
        # ]

        return optimizers, schedulers



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
