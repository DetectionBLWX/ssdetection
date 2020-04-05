'''
Function:
	define the iou losses
Author:
	Charles
'''
import torch


'''giou loss, I borrow the code from mmdet'''
def GIoULoss(bbox_preds, bbox_targets, eps=1e-7, size_average=True, loss_weight=1.0, avg_factor=None):
	# overlap
	lt = torch.max(bbox_preds[:, :2], bbox_targets[:, :2])
	rb = torch.min(bbox_preds[:, 2:], bbox_targets[:, 2:])
	wh = (rb - lt + 1).clamp(min=0)
	overlap = wh[:, 0] * wh[:, 1]
	# union
	ap = (bbox_preds[:, 2] - bbox_preds[:, 0] + 1) * (bbox_preds[:, 3] - bbox_preds[:, 1] + 1)
	ag = (bbox_targets[:, 2] - bbox_targets[:, 0] + 1) * (bbox_targets[:, 3] - bbox_targets[:, 1] + 1)
	union = ap + ag - overlap + eps
	# IoU
	ious = overlap / union
	# enclose area
	enclose_x1y1 = torch.min(bbox_preds[:, :2], bbox_targets[:, :2])
	enclose_x2y2 = torch.max(bbox_preds[:, 2:], bbox_targets[:, 2:])
	enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
	enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps
	# GIoU
	gious = ious - (enclose_area - union) / enclose_area
	loss = 1 - gious
	# summary and return the loss
	if avg_factor is None:
		loss = loss.mean() if size_average else loss.sum()
	else:
		loss = (loss.sum() / avg_factor) if size_average else loss.sum()
	return loss * loss_weight