import torch
import torch.nn as nn
import numpy as np
from typing import List,Tuple, Dict, Any, Optional
from omegaconf import OmegaConf
from segment.utils.general import initialize_from_config
from torch.optim import lr_scheduler
import torchmetrics
import pytorch_lightning as pl
from torchvision.models import resnet18,resnet50
from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import precision_score,accuracy_score,roc_auc_score, recall_score,f1_score


class Resnet50(pl.LightningModule):
    def __init__(self,
                 image_key,
                 num_classes: int,
                 weight_decay,
                 ckpt_path: str = None,
                 ignore_keys: list = [],
                 ):
        super(Resnet50, self).__init__()
        self.weight_decay = weight_decay
        self.image_key = image_key
        self.model = resnet50(pretrained=True)
        # 冻结模型的参数
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # 替换最后的全连接层
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

        # Create loss module
        self.loss = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        task = 'multiclass' if num_classes > 2 else 'binary'
        # metrics
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs = self.get_input(batch, self.image_key)
        labels = batch['class_label']

        preds = self.model(imgs)
        loss = self.loss(preds, labels)




        #
        # # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        # self.log("train/acc", acc, prog_bar=True, logger=True, on_epoch=True)
        # self.log("train/mean_pre", mean_pre, prog_bar=True, logger=True, on_epoch=True)
        # self.log("train/recall", recall, prog_bar=True, logger=True, on_epoch=True)
        # self.log("train/f1", f1, prog_bar=True, logger=True, on_epoch=True)
        # self.log("train/sp", sp, prog_bar=True, logger=True, on_epoch=True)
        # self.log("train/auroc", auroc, prog_bar=True, logger=True, on_epoch=True)

        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = self.get_input(batch, self.image_key)
        labels = batch['class_label']
        logits = self.model(imgs)
        preds = nn.functional.softmax(logits,dim=1).argmax(1)

        loss = self.loss(logits, labels)
        output = {'y_true':labels,'y_pred':preds,'loss_step':loss}
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return output

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        y_true = []
        y_pred = []

        for output in outputs:
            y_true.extend(output['y_true'].cpu().numpy().flatten())
            y_pred.extend(output['y_pred'].cpu().numpy().flatten())
        if len(np.unique(y_true)) < 2:
            # 处理只有一个类别的情况
            self.log("val/auc", 0.0, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        else:
            preci_score = precision_score(y_true, y_pred)
            acc_score = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            self.log("val/preci_score", preci_score, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/acc_score", acc_score, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/auc", auc, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/recall", recall, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/f1", f1, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

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



    # def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
    #     log = dict()
    #     x = self.get_input(batch, self.image_key).to(self.device)
    #     y = batch['label']
    #     # log["originals"] = x
    #     out = self(x)
    #     out = torch.nn.functional.softmax(out,dim=1)
    #     predict = out.argmax(1)
    #
    #     y_color, predict_color = self.gray2rgb(y, predict)
    #
    #     log["image"] = x
    #     log["label"] = y_color
    #     log["predict"] = predict_color
    #     return log
