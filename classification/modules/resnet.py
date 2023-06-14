import torch
import torch.nn as nn
import numpy as np
from typing import List,Tuple, Dict, Any, Optional
import torch.nn.functional as F
from omegaconf import OmegaConf
from utils.general import initialize_from_config
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from torchvision.models import resnet18,resnet50
# from torchmetrics.functional import accuracy,precision,f1_score,recall,auroc
from torchmetrics import Accuracy,Precision,F1Score,Recall,AUROC
import torchmetrics

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
        self.task = 'multiclass' if num_classes > 2 else 'binary'

        self.acc = Accuracy(self.task)
        self.pre = Precision(self.task)
        self.f1 = F1Score(self.task)
        self.recall = Recall(self.task)
        self.auroc = AUROC(self.task)

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
        preds = nn.functional.softmax(preds, dim=1).argmax(1)

        self.acc(preds, labels)
        self.pre(preds, labels)
        self.f1(preds, labels)
        self.recall(preds,labels)
        self.auroc(preds, labels)

        self.log(f"train/acc", self.acc, prog_bar=True)
        self.log(f"train/pre", self.pre, prog_bar=True)
        self.log(f"train/f1", self.f1, prog_bar=True)
        self.log(f"train/recall", self.recall, prog_bar=True)
        self.log(f"train/auroc", self.auroc, prog_bar=True)

        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def evaluate(self, batch, stage=None):
        imgs = self.get_input(batch, self.image_key)
        labels = batch['class_label']
        logits = self.model(imgs)
        loss = self.loss(logits, labels)
        logits = F.log_softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)


        self.acc(preds, labels)
        self.pre(preds, labels)
        self.f1(preds, labels)
        self.recall(preds,labels)
        self.auroc(preds, labels)

        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/acc", self.acc, prog_bar=True)
            self.log(f"{stage}/pre", self.pre, prog_bar=True)
            self.log(f"{stage}/f1", self.f1, prog_bar=True)
            self.log(f"{stage}/recall", self.recall, prog_bar=True)
            self.log(f"{stage}/auroc", self.auroc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch,"val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch,"test")


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
