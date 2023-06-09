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
    mask[img >= 1] = 1
    return mask

def preprocess_multi_mask(img):
    mask = np.zeros_like(img)
    mask[img == 15] = 1
    mask[img == 38] = 2
    mask[img == 75] = 3
    mask[img == 113] = 4
    return mask

class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, segmentation_root,
                 size=None, interpolation="bicubic",
                 n_labels=2, shift_segmentation=False, train=True,seg_object='5. Optic Disc'
                 ):
        self.seg_object = seg_object
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = os.path.join(segmentation_root,self.seg_object)

        endwith = ''
        if self.seg_object == '5. Optic Dis':
            endwith = '_OD.tif'
        elif self.seg_object == 'multi_seg' or self.seg_object == 'dr_single':
            endwith = '_gt.png'

        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "segmentation_path_": [os.path.join(self.segmentation_root, l).replace(".jpg", endwith)
                                   for l in self.image_paths]
        }
        self.train = train
        self.size = size

        # ==================
        base_size = 565
        crop_size = size
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        mean = (0.318, 0.199, 0.0831)
        std = (0.2991, 0.191, 0.0851)
        # ==================
        if self.train:
            self.transforms = T.Compose([
                                        T.RandomResize(min_size, max_size),
                                        # T.Resize(crop_size),
                                        T.RandomHorizontalFlip(0.5),
                                        T.RandomVerticalFlip(0.5),
                                        T.CenterCrop(crop_size),
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
        if not image.mode == "RGB":
            image = image.convert("RGB")
        segmentation = Image.open(example["segmentation_path_"])
        if not self.train and self.seg_object == "5. Optic Disc":
            segmentation = segmentation.transpose(Image.ROTATE_90)
        if not segmentation.mode == "L":
            segmentation = segmentation.convert("L")
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        # Preprocess
        if self.seg_object == 'multi_seg':
            segmentation = preprocess_multi_mask(segmentation)
        else:
            segmentation = preprocess_mask(segmentation)
        segmentation = Image.fromarray(segmentation)

        img, mask = self.transforms(image, segmentation)
        example["image"] = img
        example["label"] = mask
        return example


class DDRSegTrain(SegmentationBase):
    def __init__(self, size=None, train=True,seg_object='5. Optic Disc', interpolation="bicubic"):
        super().__init__(data_csv='data/DDR/ddr_train.txt',
                         data_root='data/DDR/images',
                         segmentation_root='data/DDR/masks',
                         size=size, interpolation=interpolation, train=train,
                         n_labels=2,seg_object=seg_object)


class DDRSegEval(SegmentationBase):
    def __init__(self, size=None, train=False,seg_object='5. Optic Disc', interpolation="bicubic"):
        super().__init__(data_csv='data/DDR/ddr_eval.txt',
                         data_root='data/DDR/images',
                         segmentation_root='data/DDR/masks',
                         size=size, interpolation=interpolation, train=train,
                         n_labels=2,seg_object=seg_object)


