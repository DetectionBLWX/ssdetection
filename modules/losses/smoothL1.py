'''
Function:
	define the smooth l1 losses
Author:
	Charles
'''
import torch
import numpy as np


'''smooth l1 loss with beta'''
def betaSmoothL1Loss(bbox_preds, bbox_targets, beta=1, size_average=True, loss_weight=1.0):
	diff = torch.abs(bbox_preds - bbox_targets)
	loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
	loss = loss.mean() if size_average else loss.sum()
	return loss * loss_weight


'''balanced smooth l1 Loss, I borrow the code from mmdet'''
def balancedSmoothL1Loss(bbox_preds, bbox_targets, beta=1.0, alpha=0.5, gamma=1.5, size_average=True, loss_weight=1.0):
	assert (beta > 0.) and (bbox_preds.size() == bbox_targets.size()) and (bbox_targets.numel() > 0)
	diff = torch.abs(bbox_preds - bbox_targets)
	b = np.e ** (gamma / alpha) - 1
	loss = torch.where(diff < beta, alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff, gamma * diff + gamma / b - alpha * beta)
	loss = loss.mean() if size_average else loss.sum()
	return loss * loss_weight