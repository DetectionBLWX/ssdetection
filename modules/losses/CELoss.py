'''
Function:
	define the cross entropy loss
Author:
	Charles
'''
import torch.nn.functional as F


'''cross entropy loss'''
def CrossEntropyLoss(preds, targets, loss_weight=1.0, size_average=True, avg_factor=None):
	loss = F.cross_entropy(preds, targets, reduction='none')
	if avg_factor is None:
		loss = loss.mean() if size_average else loss.sum()
	else:
		loss = (loss.sum() / avg_factor) if size_average else loss.sum()
	return loss * loss_weight