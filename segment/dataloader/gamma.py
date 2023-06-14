import os
import numpy as np
import cv2
import albumentations
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
from . import transforms as T

def preprocess_mask(img,label_type):
    od_mask = np.zeros_like(img)
    oc_mask = np.zeros_like(img)
    od_oc_mask = np.zeros_like(img)
    if label_type == 'od':
        od_mask[img == 128] = 1
        od_mask[img == 0] = 1
    elif label_type == 'oc':
        oc_mask[img == 0] = 1
    elif label_type == 'od_oc':
        od_oc_mask[img == 128] = 1
        od_oc_mask[img == 0] = 2

    return {'od':od_mask,
            'oc':oc_mask,
            'od_oc':od_oc_mask,
    }


class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, segmentation_root,
                 size=None, interpolation="bicubic",
                 n_labels=2, shift_segmentation=False, train=True,seg_object='od'
                 ):
        self.seg_object = seg_object
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
            "segmentation_path_": [os.path.join(self.segmentation_root, l).replace(".jpg", '.png')
                                   for l in self.image_paths]
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


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image_tensor,_ = self.org_transforms(image,image)
        example["original_image"] = image_tensor
        segmentation = Image.open(example["segmentation_path_"])
        if not segmentation.mode == "L":
            segmentation = segmentation.convert("L")
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        # Preprocess
        segmentation_dict = preprocess_mask(segmentation)
        segmentation = segmentation_dict[self.seg_object]
        segmentation = Image.fromarray(segmentation)

        img, mask = self.transforms(image, segmentation)

        _,od_mask = self.transforms(image, Image.fromarray(segmentation_dict['od']))
        _,oc_mask = self.transforms(image, Image.fromarray(segmentation_dict['oc']))
        _,od_oc_mask = self.transforms(image, Image.fromarray(segmentation_dict['od_oc']))

        example["od_mask"] = od_mask
        example["oc_mask"] = oc_mask
        example["od_oc_mask"] = od_oc_mask

        example["image"] = img
        example["label"] = mask
        return example


class GAMMASegTrain(SegmentationBase):
    def __init__(self,data_csv='',data_root='',segmentation_root='',size=None, train=True,seg_object='od', interpolation="bicubic"):
        super().__init__(data_csv=data_csv,
                         data_root=data_root,
                         segmentation_root=segmentation_root,
                         size=size, interpolation=interpolation, train=train,
                         n_labels=2,seg_object=seg_object)


class GAMMASegEval(SegmentationBase):
    def __init__(self,data_csv='',data_root='',segmentation_root='', size=None, train=False,seg_object='od', interpolation="bicubic"):
        super().__init__(data_csv=data_csv,
                         data_root=data_root,
                         segmentation_root=segmentation_root,
                         size=size, interpolation=interpolation, train=train,
                         n_labels=2,seg_object=seg_object)


