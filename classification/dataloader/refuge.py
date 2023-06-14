import os
import numpy as np
import cv2
import albumentations
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
from . import transforms as T
import torchvision.transforms as transforms


class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root,
                 size=None, interpolation="bicubic",
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
            "file_path_": [os.path.join(self.data_root, l.split(' ')[0])
                           for l in self.image_paths],
            'label': [l.split(' ')[1] for l in self.image_paths],
        }
        self.train = train
        self.size = size
        # ==================
        base_size = 565
        crop_size = size
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        mean = (0.28179467, 0.16782693, 0.08382585)
        std = (0.17760678, 0.10207705, 0.05228512)
        # ==================
        if self.train:
            self.transforms = T.Compose([
                                        # T.RandomResize(min_size, max_size),
                                        T.Resize(crop_size),
                                        T.RandomHorizontalFlip(0.5),
                                        T.RandomVerticalFlip(0.5),
                                        T.RandomRotation(0.5,90),
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

        self.org_transforms = T.Compose([
                                         T.Resize(crop_size),
                                         T.ToTensor(),
        ])
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image_tensor,_ = self.org_transforms(image,image)
        example["original_image"] = image_tensor

        img, mask = self.transforms(image, image)

        example["image"] = img
        example["class_label"] = int(example["label"])
        return example


class REFUGEClassTrain(SegmentationBase):
    def __init__(self, size=None, train=True,interpolation="bicubic",
                 data_csv='data/REFUGE/class_train.txt',
                 data_root='data/REFUGE',
                 ):
        super().__init__(data_csv=data_csv,
                         data_root=data_root,
                         size=size, interpolation=interpolation, train=train,
                         n_labels=2)


class REFUGEClassEval(SegmentationBase):
    def __init__(self,
                 size=None,
                 train=False,
                 interpolation="bicubic",
                 data_csv='data/REFUGE/class_eval.txt',
                 data_root='data/REFUGE',
                 ):
        super().__init__(data_csv=data_csv,
                         data_root=data_root,
                         size=size, interpolation=interpolation, train=train,
                         n_labels=2)

class REFUGEClassTest(SegmentationBase):
    def __init__(self,
                 size=None,
                 train=False,
                 interpolation="bicubic",
                 data_csv='data/REFUGE/class_test.txt',
                 data_root='data/REFUGE',
                 ):
        super().__init__(data_csv=data_csv,
                         data_root=data_root,
                         size=size, interpolation=interpolation, train=train,
                         n_labels=2)
