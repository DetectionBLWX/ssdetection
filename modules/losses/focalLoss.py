'''
Function:
	define the focal loss
Author:
	Charles
'''
import torch.nn.functional as F


'''sigmoid focal loss'''
def pySigmoidFocalLoss(preds, targets, loss_weight=1.0, gamma=2.0, alpha=0.25, size_average=True):
	preds_sigmoid = preds.sigmoid()
	targets = targets.type_as(preds)
	pt = (1 - preds_sigmoid) * targets + preds_sigmoid * (1 - targets)
	focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * pt.pow(gamma)
	loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none') * focal_weight
	loss = loss.mean() if size_average else loss.sum()
	return loss * loss_weight