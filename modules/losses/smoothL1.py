'''define the smooth l1 loss'''
import torch


'''smooth l1 loss'''
def smoothL1Loss(bbox_preds, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
	sigma_2 = sigma ** 2
	box_diff = bbox_preds - bbox_targets
	in_box_diff = bbox_inside_weights * box_diff
	abs_in_box_diff = torch.abs(in_box_diff)
	smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
	in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
	out_loss_box = bbox_outside_weights * in_loss_box
	loss_box = out_loss_box
	for i in sorted(dim, reverse=True):
		loss_box = loss_box.sum(i)
	loss_box = loss_box.mean()
	return loss_box


'''smooth l1 loss with beta'''
def betaSmoothL1Loss(bbox_preds, bbox_targets, beta=1, size_average=True):
	diff = torch.abs(bbox_preds - bbox_targets)
	loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
	loss = loss.mean() if size_average else loss.sum()
	return loss