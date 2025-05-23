import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal
import gdown
from sklearn.metrics import confusion_matrix




def ConLoss(logit, target):
    target = target.permute(0,3,1,2)
    loss = nn.BCEWithLogitsLoss()(logit, target)
    return loss

def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss




def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def MIoU(predictions, targets, num_classes=2, eps=1e-7):
    """
    Calculate Mean Intersection over Union (MIoU)
    
    Args:
        predictions: Model predictions tensor of shape (N, C, H, W)
        targets: Ground truth tensor of shape (N, H, W)
        num_classes: Number of classes (default=2 for binary segmentation)
        eps: Small value to avoid division by zero
        
    Returns:
        miou: Mean IoU across all classes
        class_ious: IoU for each class
    """
    # Convert predictions to class indices
    predictions = torch.argmax(predictions, dim=1)  # (N, H, W)
    
    # Initialize confusion matrix
    conf_matrix = torch.zeros((num_classes, num_classes), device=predictions.device)
    
    # Flatten predictions and targets
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)
    
    # Calculate confusion matrix
    for i in range(num_classes):
        for j in range(num_classes):
            conf_matrix[i, j] = torch.sum((pred_flat == i) & (target_flat == j))
            
    # Calculate IoU for each class
    class_ious = torch.zeros(num_classes, device=predictions.device)
    for i in range(num_classes):
        intersection = conf_matrix[i, i]
        union = (torch.sum(conf_matrix[i, :]) + 
                torch.sum(conf_matrix[:, i]) - 
                conf_matrix[i, i])
        class_ious[i] = (intersection + eps) / (union + eps)
    
    # Calculate mean IoU
    miou = torch.mean(class_ious)
    
    return miou





def precision_recall_f1(predictions, targets, num_classes=2, eps=1e-7):
    """
    Calculate Mean Precision, Mean Recall, and Mean F1-Score for a segmentation task
    
    Args:
        predictions: Model predictions tensor of shape (N, C, H, W)
        targets: Ground truth tensor of shape (N, H, W)
        num_classes: Number of classes (default=2 for binary segmentation)
        eps: Small value to avoid division by zero
        
    Returns:
        mean_precision: Mean Precision across all classes
        mean_recall: Mean Recall across all classes
        mean_f1_score: Mean F1-Score across all classes
    """
    # Convert predictions to class indices
    predictions = torch.argmax(predictions, dim=1)  # (N, H, W)
    
    # Initialize confusion matrix
    conf_matrix = torch.zeros((num_classes, num_classes), device=predictions.device)
    
    # Flatten predictions and targets
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)
    
    # Calculate confusion matrix
    for i in range(num_classes):
        for j in range(num_classes):
            conf_matrix[i, j] = torch.sum((pred_flat == i) & (target_flat == j))
    
    # Calculate precision, recall, and F1-score for each class
    precision = torch.zeros(num_classes, device=predictions.device)
    recall = torch.zeros(num_classes, device=predictions.device)
    f1_score = torch.zeros(num_classes, device=predictions.device)
    
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = torch.sum(conf_matrix[:, i]) - TP
        FN = torch.sum(conf_matrix[i, :]) - TP
        
        # Calculate Precision and Recall for the current class
        precision[i] = (TP + eps) / (TP + FP + eps)
        recall[i] = (TP + eps) / (TP + FN + eps)
        
        # Calculate F1-Score for the current class
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + eps)
    
    # Calculate mean precision, mean recall, and mean F1-Score
    mean_precision = torch.mean(precision)
    mean_recall = torch.mean(recall)
    mean_f1_score = torch.mean(f1_score)
    
    return mean_precision.item(), mean_recall.item(), mean_f1_score.item()



def calculate_eval_metrics(predictions, labels, num_classes=2):
    # Flatten the arrays
    preds = predictions.flatten()
    labels = labels.flatten()

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds, labels=np.arange(num_classes))

    # Compute mIoU and mDice
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / union
    miou = np.nanmean(iou)  # Ignore NaN values (for classes that don't appear in the dataset)

    # mDice
    dice = (2 * intersection) / (cm.sum(axis=1) + cm.sum(axis=0))
    mdice = np.nanmean(dice)

    return miou, mdice




