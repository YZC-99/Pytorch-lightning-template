# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score,confusion_matrix,classification_report

import argparse, os, sys, datetime, glob, importlib
import torch
import torch.nn as nn

from segment.utils.general import get_config_from_file, initialize_from_config, setup_callbacks

import numpy as np
import matplotlib.pyplot as plt

def gray2rgb(y, predict):
    color_map = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128]}
    # Convert labels and predictions to color images.
    y_color = torch.zeros(y.size(0), 3, y.size(1), y.size(2), device=y.device)
    predict_color = torch.zeros(predict.size(0), 3, predict.size(1), predict.size(2), device=y.device)
    for label, color in color_map.items():
        mask_y = (y == int(label))
        mask_p = (predict == int(label))
        for i in range(3):  # apply each channel individually
            y_color[mask_y, i] = color[i]
            predict_color[mask_p, i] = color[i]
    return y_color, predict_color

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def visual(input_image,y_true,y_pred):
    y_color,predict_color = gray2rgb(y_true,y_pred)
    y_color,predict_color = torch.squeeze(y_color),torch.squeeze(predict_color)
    # 将标签图和预测图转换为灰度图像
    # label_image = y_true.reshape(y_true.shape)  # 假设image_shape为标签图的形状
    # pred_image = y_pred.reshape(y_true.shape)  # 假设image_shape为预测图的形状
    input_image = np.transpose(torch.squeeze(input_image).cpu().numpy())
    label_image = np.transpose(y_color.cpu().numpy(),(1,2,0))
    pred_image = np.transpose(predict_color.cpu().numpy(),(1,2,0))
    # 创建一个图像对象并绘制标签图和预测图
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # 绘制图
    axes[0].imshow(input_image)
    axes[0].set_title('Image')
    # 绘制标签图
    axes[1].imshow(label_image)
    axes[1].set_title('Label Image')

    # 绘制预测图
    axes[2].imshow(pred_image)
    axes[2].set_title('Predicted Image')

    # 设置图像标题和坐标轴标签
    fig.suptitle('Label vs. Predicted Images')
    axes[0].set_xlabel('X-axis')
    axes[0].set_ylabel('Y-axis')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')
    axes[2].set_xlabel('X-axis')
    axes[2].set_ylabel('Y-axis')
    # 显示图像
    plt.show()

if __name__ == '__main__':
    # Load configuration
    config = get_config_from_file("../refer/refuge_pretrained/refuge_od_unet_for_refuge.yaml")
    device = 'cuda:0'
    num_classes= 2
    # Build model
    model = initialize_from_config(config.model)
    model.learning_rate = config.model.base_learning_rate
    # Build data modules
    data = initialize_from_config(config.dataset)
    data.prepare_data()
    data.setup()
    test_dl = data._test_dataloader()
    model.to(device)
    model.eval()

    for data in test_dl:
        input,y = data['image'].to(device),data['label'].to(device)
        logits = model(input)['out'].detach()
        preds = nn.functional.softmax(logits, dim=1).argmax(1)
        visual(data['original_image'],y,preds)

        y_true = y.cpu().numpy().flatten()
        y_pred = preds.cpu().numpy().flatten()

        jaccard = JaccardIndex(num_classes=num_classes, task='binary' if num_classes == 2 else 'multiclass')
        jaccard = jaccard.to(device)
        mean_iou = jaccard(preds, y)

        dice = Dice(num_classes=num_classes, average='macro')
        dice = dice.to(device)
        mean_dice_score = dice(preds, y)

        if num_classes == 2:
            y_probs = nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy().flatten()
            precision, recall, _ = precision_recall_curve(y_true, y_probs)
            aupr = auc(recall, precision)
            roc_auc = roc_auc_score(y_true, y_pred)
            average_precision = average_precision_score(y_true, y_probs)

        # Calculate metrics for each class
        for i in range(num_classes):
            binary_y_true = (y_true == i)
            binary_y_pred = (y_pred == i)

            # 计算dice、iou
            jaccard_i = JaccardIndex(num_classes=2, task='binary')
            # jaccard_i = jaccard_i.to(device)
            iou_i = jaccard_i(torch.from_numpy(binary_y_pred), torch.from_numpy(binary_y_true))

            dice_i = Dice(num_classes=2, average='macro')
            # dice_i = dice_i.to(device)
            dice_score_i = dice_i(torch.from_numpy(binary_y_pred), torch.from_numpy(binary_y_true))

            conf_matrix = confusion_matrix(binary_y_true, binary_y_pred)
            # Ensure the confusion matrix is 2x2.
            if conf_matrix.size == 1:
                conf_matrix = conf_matrix.reshape((1, 1))
                conf_matrix = np.pad(conf_matrix, ((0, 1), (0, 1)), 'constant')

            tn, fp, fn, tp = conf_matrix.ravel()

            eps = 1e-6
            se = tp / (tp + fn + eps)
            sp = tn / (tn + fp + eps)
            acc = (tp + tn) / (tp + tn + fp + fn + eps)

            # 计算AUC_PR、AUC_ROC
            y_probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
            y_true_i = (y_true == i)
            y_probs_i = y_probs[:, i].flatten()

            if len(np.unique(y_true_i)) > 1:
                precision, recall, _ = precision_recall_curve(y_true_i, y_probs_i)
                auc_pr_i = auc(recall, precision)
                auc_roc_i = roc_auc_score(y_true_i, y_probs_i)
        print(mean_iou)
        print(mean_dice_score)
        break
