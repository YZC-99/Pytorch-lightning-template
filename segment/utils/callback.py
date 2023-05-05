# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------
from datetime import datetime
import pandas as pd
import os
import wandb
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from typing import Tuple, Generic, Dict
import yaml
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import Callback


class SetupCallback(Callback):
    def __init__(self, config: OmegaConf, exp_config: OmegaConf, basedir: Path, logdir: str = "log", ckptdir:str = "ckpt") -> None:
        super().__init__()
        self.logdir = basedir / logdir
        self.ckptdir = basedir / ckptdir
        self.config = config
        self.exp_config = exp_config
        
    def on_fit_start(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            
            print("Experiment config")
            print(OmegaConf.to_yaml(self.exp_config))

            print("Model config")
            print(OmegaConf.to_yaml(self.config))

            # hparams = OmegaConf.to_container(OmegaConf.merge(self.config, self.exp_config), resolve=True)
            # hparams_file = self.logdir / "hparams.yaml"
            # with open(hparams_file, "w") as f:
            #     yaml.dump(hparams, f)

    def on_train_start(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank == 0:
            # Save exp_config and config to hparams.yaml
            hparams_path = Path(pl_module.logger.log_dir) / "hparams.yaml"
            with open(hparams_path, "w") as f:
                f.write("experiment_config:\n")
                f.write(OmegaConf.to_yaml(self.exp_config))
                f.write("\n")
                f.write("model_config:\n")
                f.write(OmegaConf.to_yaml(self.config))

class ImageLogger(Callback):
    def __init__(self, batch_frequency: int, max_images: int, clamp: bool = True, increase_log_steps: bool =True) -> None:
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TensorBoardLogger: self._tensorboard,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        #raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir: str, split: str, images: Dict,
                  global_step: int, current_epoch: int, batch_idx: int) -> None:
        root = os.path.join(save_dir, "results", split)
        os.makedirs(root, exist_ok=True)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)
            
    def log_img(self, pl_module: pl.LightningModule, batch: Tuple[torch.LongTensor, torch.FloatTensor], batch_idx: int, split: str = "train") -> None:
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N].detach().cpu()
                if self.clamp:
                    images[k] = images[k].clamp(0, 1)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx: int) -> bool:
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule,
                           outputs: Generic, batch: Tuple[torch.LongTensor, torch.FloatTensor], batch_idx: int) -> None:
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule,
                                outputs: Generic, batch: Tuple[torch.LongTensor, torch.FloatTensor],
                                dataloader_idx: int, batch_idx: int) -> None:
        self.log_img(pl_module, batch, batch_idx, split="val")

class CSVLogger(Callback):
    def __init__(self, logdir, filename):
        super().__init__()
        self.logdir = logdir
        self.filename = filename
        self.df = pd.DataFrame(columns=['lr-Adam', 'created_at', 'train_loss_step', 'epoch', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1_score', 'val_auc', 'train_loss_epoch'])
        self.counter = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]['lr']
        loss = trainer.callback_metrics['val_loss']
        acc = trainer.callback_metrics['val_acc']
        precision = trainer.callback_metrics['val_precision']
        recall = trainer.callback_metrics['val_recall']
        f1_score = trainer.callback_metrics['val_f1_score']
        auc = trainer.callback_metrics['val_auc']
        epoch = pl_module.current_epoch
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        step = self.counter + 1
        train_loss_epoch = trainer.callback_metrics['train_loss_epoch']
        self.df.loc[step] = [lr, now, step, epoch, loss, acc, precision, recall, f1_score, auc, train_loss_epoch]
        self.counter += 1

    def on_train_end(self, trainer, pl_module):
        filename = os.path.join(self.logdir, self.filename)
        self.df.to_csv(filename, index=False)