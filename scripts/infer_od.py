# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score,confusion_matrix,classification_report
from math import log
import argparse, os, sys, datetime, glob, importlib
import torch
import torch.nn as nn

from segment.utils.general import get_config_from_file, initialize_from_config, setup_callbacks
from segment.dataloader.refuge import *
from segment.dataloader.gamma import *
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

def visual(input_image,y_true,y_pred,logits,od_oc_mask):
    logits = nn.functional.softmax(logits, dim=1)
    logits,max_indices = torch.max(logits, dim=1)
    logits = torch.squeeze(logits)
    max_indices = torch.squeeze(max_indices)

    prob = max_indices*logits

    y_color,predict_color = gray2rgb(y_true,y_pred)
    y_color,predict_color = torch.squeeze(y_color),torch.squeeze(predict_color)

    od_oc_mask_color,_ = gray2rgb(od_oc_mask,y_pred)
    od_oc_mask_color = torch.squeeze(od_oc_mask_color)
    # 将标签图和预测图转换为灰度图像
    # label_image = y_true.reshape(y_true.shape)  # 假设image_shape为标签图的形状
    # pred_image = y_pred.reshape(y_true.shape)  # 假设image_shape为预测图的形状
    input_image = np.transpose(torch.squeeze(input_image).cpu().numpy())
    label_image = np.transpose(y_color.cpu().numpy(),(1,2,0))
    pred_image = np.transpose(predict_color.cpu().numpy(),(1,2,0))
    od_oc_mask_image = np.transpose(od_oc_mask_color.cpu().numpy(),(1,2,0))

    prob = np.transpose(prob.cpu().numpy())
    prob = np.rot90(prob, k=1, axes=(0, 1))
    # 创建一个图像对象并绘制标签图和预测图
    fig, axes = plt.subplots(1, 6, figsize=(16, 9))

    # 绘制图
    axes[0].imshow(input_image)
    axes[0].set_title('Image')
    # 绘制标签图
    axes[1].imshow(label_image)
    axes[1].set_title('OD Label Image')

    # 绘制预测图
    axes[2].imshow(pred_image)
    axes[2].set_title('Predicted OD Image')

    # 绘制概率图
    pseudo_oc = np.zeros_like(prob)
    pseudo_oc[prob == 1] = 1
    # pseudo_oc = np.rot90(pseudo_oc, k=1, axes=(0, 1))
    axes[3].imshow(pseudo_oc)
    axes[3].set_title('pseudo OC Image')
    # axes[3].tight_layout()

    # od_oc
    axes[4].imshow(od_oc_mask_image)
    axes[4].set_title('od_oc_mask_image')

    # true_oc - pseudo_oc
    od_oc_mask = torch.squeeze(od_oc_mask).cpu().numpy()

    true_oc = np.zeros_like(od_oc_mask)
    true_oc[od_oc_mask == 1] = 1
    oc_true_minus_pseudo = true_oc - pseudo_oc
    # od_oc
    mapped_value = np.log(prob - np.min(prob) + 1) / np.log(np.max(prob) - np.min(prob) + 1) * (255 - 0) + 0
    mapped_value = np.clip(mapped_value, 0, 255)  # 截断映射后的值，限制在0到255范围内
    axes[5].imshow(mapped_value)
    axes[5].set_title('mapped_value')


    # 关闭所有子图的坐标刻度
    plt.setp(axes, xticks=[], yticks=[])
    # 设置图像标题和坐标轴标签
    fig.suptitle('Label vs. Predicted Images')
    # 显示图像
    plt.show()


def testing(dataloader,
            model,
            num_classes):

    device = 'cuda:0'
    model.to(device)
    model.eval()
    num = len(dataloader)
    total_iou = 0
    total_dice = 0
    for data in dataloader:
        input,y = data['image'].to(device),data['label'].to(device)
        logits = model(input)['out'].detach()
        preds = nn.functional.softmax(logits, dim=1).argmax(1)

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
        total_iou += mean_iou.detach().cpu().numpy()
        total_dice += mean_dice_score.detach().cpu().numpy()
    return {'mean_iou':total_iou/num,
            'mean_dice_score':total_dice/num}

if __name__ == '__main__':
    # Load configuration
    config_path = "../refer/refuge_pretrained/refuge_od_unet.yaml"
    config = get_config_from_file(config_path)
    num_classes= 2
    device = 'cuda:0'
    # Build model
    model = initialize_from_config(config.model)
    model.learning_rate = config.model.base_learning_rate

    dl_dict = {}
    # Build data modules
    refuge_dataset = REFUGESegEval(
                                data_csv='F:/DL-Data/eyes/glaucoma_OD_OC/REFUGE/refuge_eval.txt',
                                data_root='F:/DL-Data/eyes/glaucoma_OD_OC/REFUGE/images',
                                segmentation_root='F:/DL-Data/eyes/glaucoma_OD_OC/REFUGE/ground_truths',
                                size=512,
                                seg_object='od')
    refuge_dl = torch.utils.data.DataLoader(refuge_dataset,batch_size=1)
    dl_dict['refuge_dl']=refuge_dl

    gamma_dataset = GAMMASegEval(
                                data_csv='F:/DL-Data/eyes/glaucoma_OD_OC/GAMMA/gamma_all.txt',
                                data_root='F:/DL-Data/eyes/glaucoma_OD_OC/GAMMA/images',
                                segmentation_root='F:/DL-Data/eyes/glaucoma_OD_OC/GAMMA/ground_truths',
                                size=512,
                                seg_object='od')
    gamma_dl = torch.utils.data.DataLoader(gamma_dataset,batch_size=1)
    dl_dict['gamma_dl']=gamma_dl

    model.to(device)
    model.eval()
    # threshold = 3
    image_number = 30
    for i in np.arange(4.1,4.7,0.1):
        threshold = i
        img_folder = './threshold_{}'.format(threshold)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        for id,data in enumerate(refuge_dl):
            input,y,original_image = data['image'].to(device),data['label'].to(device),data['original_image'].to(device)
            od_oc_mask,oc_mask = data['od_oc_mask'].to(device),data['oc_mask'].to(device)

            logits = model(input)['out'].detach()
            logits_softmaxed = nn.functional.softmax(logits, dim=1)
            prob,preds = torch.max(logits, dim=1)
            prob_softmax,preds_softmax = torch.max(logits_softmaxed, dim=1)
            prob = preds * prob
            ##
            prob_array = prob.squeeze().detach().cpu().numpy()
            prob_array[prob_array > threshold] = 10

            fig,axes = plt.subplots(1,7,figsize=(16,9))
            # 可视化original_image
            axes[0].imshow(original_image.squeeze().permute(1,2,0).cpu().numpy())
            axes[0].axis('off')  # 去除坐标轴
            axes[0].set_title('original_image')

            # 可视化y
            axes[1].imshow(y.squeeze().cpu().numpy())
            axes[1].axis('off')  # 去除坐标轴
            axes[1].set_title('Ground Truth')

            # 可视化preds
            axes[2].imshow(preds.squeeze().cpu().numpy())
            axes[2].axis('off')  # 去除坐标轴
            axes[2].set_title('Predictions')

            # 可视化prob
            axes[3].imshow(prob.squeeze().cpu().numpy())
            axes[3].axis('off')  # 去除坐标轴
            axes[3].set_title('prob')

            # 可视化
            overlay = oc_mask.squeeze().cpu().numpy() + prob.squeeze().cpu().numpy()
            axes[4].imshow(overlay)
            axes[4].axis('off')  # 去除坐标轴
            axes[4].set_title('oc_mask over prob')

            # 可视化threshold
            axes[5].imshow(prob_array)
            axes[5].axis('off')  # 去除坐标轴
            axes[5].set_title('threshold')

            # 可视化od_oc_mask
            axes[6].imshow(od_oc_mask.squeeze().cpu().numpy())
            axes[6].axis('off')  # 去除坐标轴
            axes[6].set_title('od_oc_mask')

            plt.tight_layout()  # 调整子图布局
            # plt.show()
            plt.savefig('{}/概率对比图{}.jpg'.format(img_folder,id),dpi=500)
            if id > image_number:
                break
        # visual(data['original_image'],y,preds,logits,od_oc_mask)


    #
    # log_path = '../refer/logs/{}.txt'.format(os.path.basename(config_path).split('.')[0])
    # for name,dl in dl_dict.items():
    #     metrics = testing(dl, model, num_classes)
    #     with open(log_path,'a') as file:
    #         file.write(config_path + '\n')
    #         file.write(config.model.params.ckpt_path + '\n')
    #         file.write(name + ':\n')
    #         for key,value in metrics.items():
    #             file.write(f"{key}:{value}\n")
    #         file.write("\n")