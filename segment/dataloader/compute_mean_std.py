import os
from PIL import Image
import numpy as np

def DRIVE():
    img_channels = 3
    img_dir = "F:/DL-Data/eyes/Vessel/DRIVE/training/images"
    roi_dir = "F:/DL-Data/eyes/Vessel/DRIVE/training/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".tif")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".tif", "_mask.gif"))
        img = np.array(Image.open(img_path)) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))

        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")

def HRF():
    img_channels = 3
    img_dir = "F:/DL-Data/eyes/Vessel/HRF/images"
    roi_dir = "F:/DL-Data/eyes/Vessel/HRF/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".jpg", "_mask.tif"))
        img = np.array(Image.open(img_path)) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))

        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")

def IDRID():
    img_channels = 3
    img_dir = "F:/DL-Data/eyes/IDRID/training/images"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        img = np.array(Image.open(img_path)) / 255.
        cumulative_mean += img.mean(axis=(0,1))
        cumulative_std += img.std(axis=(0,1))

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")

def DDR():
    img_channels = 3
    img_dir = "F:/DL-Data/eyes/Multi_seg/DDR/training/images"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        img = np.array(Image.open(img_path)) / 255.
        cumulative_mean += img.mean(axis=(0,1))
        cumulative_std += img.std(axis=(0,1))

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")

def REFUGE():
    img_channels = 3
    img_dir = "F:/DL-Data/eyes/OD_OC/REFUGE/training/images"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        img = np.array(Image.open(img_path)) / 255.
        cumulative_mean += img.mean(axis=(0,1))
        cumulative_std += img.std(axis=(0,1))

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")
def main():
    REFUGE()
    # HRF()



if __name__ == '__main__':
    main()
