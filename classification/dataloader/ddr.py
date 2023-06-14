import os
import numpy as np
import cv2
import albumentations
import glob
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root,
                 size=None,
                 n_labels=2, train=True
                 ):
        self.n_labels = n_labels
        self.data_csv = data_csv
        self.data_root = data_root

        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l.split(' ')[0] for l in self.image_paths],
            "label": [l.split(' ')[1] for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l.split(' ')[0])
                           for l in self.image_paths],
        }
        self.train = train
        self.size = size

        base_size = 565
        crop_size = size
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        mean = (0.318, 0.199, 0.0831)
        std = (0.2991, 0.191, 0.0851)
        # ==================
        if self.train:
            self.transforms = T.Compose([
                                        T.RandomRotation(90),
                                        T.Resize((crop_size,crop_size)),
                                        T.RandomHorizontalFlip(0.5),
                                        T.RandomVerticalFlip(0.5),
                                        # T.CenterCrop(crop_size),
                                        # T.RandomCrop(crop_size),
                                        T.ToTensor(),
                                        T.Normalize(mean=mean, std=std),
            ])
        else:
            self.transforms = T.Compose([
                                         T.Resize(crop_size),
                                         T.ToTensor(),
                                         T.Normalize(mean=mean, std=std),
        ])


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        label = int(example["label"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        img = self.transforms(image)

        example["image"] = img
        example["label"] = torch.tensor(label)
        return example


class DDRGradTrain(SegmentationBase):
    def __init__(self,
                 data_csv,
                 data_root,
                 size=None):
        super().__init__(data_csv='F:/DL-Data/eyes/DDR/DDR-dataset.zip/DDR-dataset/DR_grading/train.txt',
                         data_root='F:/DL-Data/eyes/DDR/DDR-dataset.zip/DDR-dataset/DR_grading/train',
                         size=size,train=True)


class DDRGradEval(SegmentationBase):
    def __init__(self,
                 data_csv,
                 data_root,
                 size=None):
        super().__init__(data_csv='F:/DL-Data/eyes/DDR/DDR-dataset.zip/DDR-dataset/DR_grading/valid.txt',
                         data_root='F:/DL-Data/eyes/DDR/DDR-dataset.zip/DDR-dataset/DR_grading/valid',
                         size=size,train=False)


