import os
import numpy as np
import cv2
import albumentations
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
from . import transforms as T

def preprocess_mask(img):
    mask = np.zeros_like(img)
    mask[img >= 150] = 1
    return mask



class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, segmentation_root,
                 size=None, interpolation="bicubic",
                 n_labels=2, shift_segmentation=False, train=True
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "segmentation_path_": [os.path.join(self.segmentation_root, l).replace("tif", "gif")
                                   for l in self.image_paths]
        }
        self.train = train
        self.size = size

        # ==================
        base_size = 565
        crop_size = size
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # ==================
        if self.train:
            self.transforms = T.Compose([
                                        # T.RandomResize(min_size, max_size),
                                        T.RandomHorizontalFlip(0.5),
                                        T.RandomVerticalFlip(0.5),
                                        T.RandomCrop(crop_size),
                                        T.ToTensor(),
                                        T.Normalize(mean=mean, std=std),
            ])
        else:
            self.transforms = T.Compose([T.ToTensor(),
                                         T.Normalize(mean=mean, std=std),
        ])


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        segmentation = Image.open(example["segmentation_path_"])
        if not segmentation.mode == "L":
            segmentation = segmentation.convert("L")
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        # Preprocess
        segmentation = preprocess_mask(segmentation)
        segmentation = Image.fromarray(segmentation)

        img, mask = self.transforms(image, segmentation)
        example["image"] = img
        example["label"] = mask
        return example


class DRIVESegTrain(SegmentationBase):
    def __init__(self, size=None, train=True, interpolation="bicubic"):
        super().__init__(data_csv='data/DRIVE/drive_train.txt',
                         data_root='data/DRIVE/images',
                         segmentation_root='data/DRIVE/masks',
                         size=size, interpolation=interpolation, train=train,
                         n_labels=2)


class DRIVESegEval(SegmentationBase):
    def __init__(self, size=None, train=False, interpolation="bicubic"):
        super().__init__(data_csv='data/DRIVE/drive_eval.txt',
                         data_root='data/DRIVE/images',
                         segmentation_root='data/DRIVE/masks',
                         size=size, interpolation=interpolation, train=train,
                         n_labels=2)


