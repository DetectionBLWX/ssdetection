'''define the smooth l1 loss'''
import torch


'''smooth l1 loss with beta'''
def betaSmoothL1Loss(bbox_preds, bbox_targets, beta=1, size_average=True, loss_weight=1.0):
	diff = torch.abs(bbox_preds - bbox_targets)
	loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
	loss = loss.mean() if size_average else loss.sum()
	return loss * loss_weight