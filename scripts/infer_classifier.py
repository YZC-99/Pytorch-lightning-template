# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score,confusion_matrix
import importlib
import torch
import torch.nn as nn

from utils.general import get_config_from_file, initialize_from_config
from classification.dataloader.refuge import *
import numpy as np


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
    config_path = "../refer/refuge_pretrained/refuge_resnet50.yaml"
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

    # gamma_dataset = GAMMASegEval(
    #                             data_csv='F:/DL-Data/eyes/glaucoma_OD_OC/GAMMA/gamma_all.txt',
    #                             data_root='F:/DL-Data/eyes/glaucoma_OD_OC/GAMMA/images',
    #                             segmentation_root='F:/DL-Data/eyes/glaucoma_OD_OC/GAMMA/ground_truths',
    #                             size=512,
    #                             seg_object='od')
    # gamma_dl = torch.utils.data.DataLoader(gamma_dataset,batch_size=1)
    # dl_dict['gamma_dl']=gamma_dl

    model.to(device)
    model.eval()

    for id,data in enumerate(refuge_dl):
        input,y = data['image'].to(device),data['class_label'].to(device)
        logits = model(input).detach()
        logits_softmaxed = nn.functional.softmax(logits, dim=1)
        print(logits_softmaxed)
        break;
