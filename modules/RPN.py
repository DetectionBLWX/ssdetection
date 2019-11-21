'''
Function:
	region proposal net
Author:
	Charles
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from libs.nms.nms_wrapper import nms
from modules.utils.utils import BBoxFunctions
from modules.losses.smoothL1 import smoothL1Loss, betaSmoothL1Loss


'''
Function:
	define the proposal layer for rpn
Init Input:
	--feature_stride: stride now.
	--anchors: A x 4
	--mode: flag about TRAIN or TEST.
	--cfg: config file.
Forward Input:
	--x_cls_pred: N x 2 x H x W
	--x_loc_pred: N x 4 x H x W
	--img_info: (height, width, scale_factor)
'''
class rpnProposalLayer(nn.Module):
	def __init__(self, feature_stride, anchor_scales, anchor_ratios, mode, cfg, **kwargs):
		super(rpnProposalLayer, self).__init__()
		self.feature_stride = feature_stride
		self.anchors = RegionProposalNet.generateAnchors(scales=anchor_scales, ratios=anchor_ratios)
		self.num_anchors = self.anchors.size(0)
		if mode == 'TRAIN':
			self.pre_nms_topN = cfg.TRAIN_RPN_PRE_NMS_TOP_N
			self.post_nms_topN = cfg.TRAIN_RPN_POST_NMS_TOP_N
			self.nms_thresh = cfg.TRAIN_RPN_NMS_THRESH
		elif mode == 'TEST':
			self.pre_nms_topN = cfg.TEST_RPN_PRE_NMS_TOP_N
			self.post_nms_topN = cfg.TEST_RPN_POST_NMS_TOP_N
			self.nms_thresh = cfg.TEST_RPN_NMS_THRESH	
		else:
			raise ValueError('Unkown mode <%s> in rpnProposalLayer...' % mode)
	def forward(self, x):
		# prepare
		probs, x_loc_pred, img_info = x
		batch_size = probs.size(0)
		feature_height, feature_width = probs.size(2), probs.size(3)
		# get bg and fg probs
		bg_probs = probs[:, :self.num_anchors, :, :]
		fg_probs = probs[:, self.num_anchors:, :, :]
		# get shift
		shift_x = np.arange(0, feature_width) * self.feature_stride
		shift_y = np.arange(0, feature_height) * self.feature_stride
		shift_x, shift_y = np.meshgrid(shift_x, shift_y)
		shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose())
		shifts = shifts.contiguous().type_as(fg_probs).float()
		# get anchors
		anchors = self.anchors.type_as(fg_probs)
		anchors = anchors.view(1, self.num_anchors, 4) + shifts.view(shifts.size(0), 1, 4)
		anchors = anchors.view(1, self.num_anchors*shifts.size(0), 4).expand(batch_size, self.num_anchors*shifts.size(0), 4)
		# format x_loc_pred
		bbox_deltas = x_loc_pred.permute(0, 2, 3, 1).contiguous()
		bbox_deltas = bbox_deltas.view(batch_size, -1, 4)
		# format fg_probs
		fg_probs = fg_probs.permute(0, 2, 3, 1).contiguous()
		fg_probs = fg_probs.view(batch_size, -1)
		# convert anchors to proposals
		proposals = BBoxFunctions.anchors2Proposals(anchors, bbox_deltas)
		# clip predicted boxes to image
		proposals = BBoxFunctions.clipBoxes(proposals, img_info)
		# do nms
		scores = fg_probs
		_, order = torch.sort(scores, 1, True)
		output = scores.new(batch_size, self.post_nms_topN, 5).zero_()
		for i in range(batch_size):
			proposals_single = proposals[i]
			scores_single = scores[i]
			order_single = order[i]
			if self.pre_nms_topN > 0 and self.pre_nms_topN < scores.numel():
				order_single = order_single[:self.pre_nms_topN]
			proposals_single = proposals_single[order_single, :]
			scores_single = scores_single[order_single].view(-1, 1)
			keep_idxs = nms(torch.cat((proposals_single, scores_single), 1), self.nms_thresh, force_cpu=False)
			keep_idxs = keep_idxs.long().view(-1)
			if self.post_nms_topN > 0:
				keep_idxs = keep_idxs[:self.post_nms_topN]
			proposals_single = proposals_single[keep_idxs, :]
			scores_single = scores_single[keep_idxs, :]
			num_proposals = proposals_single.size(0)
			output[i, :, 0] = i
			output[i, :num_proposals, 1:] = proposals_single
		return output
	def backward(self, *args):
		pass


'''build target layer for rpn'''
class rpnBuildTargetLayer(nn.Module):
	def __init__(self, feature_stride, anchor_scales, anchor_ratios, mode, cfg, **kwargs):
		super(rpnBuildTargetLayer, self).__init__()
		self.feature_stride = feature_stride
		self.anchors = RegionProposalNet.generateAnchors(scales=anchor_scales, ratios=anchor_ratios)
		if mode == 'TRAIN':
			self.rpn_negative_overlap = cfg.TRAIN_RPN_NEGATIVE_OVERLAP
			self.rpn_positive_overlap = cfg.TRAIN_RPN_POSITIVE_OVERLAP
			self.rpn_fg_fraction = cfg.TRAIN_RPN_FG_FRACTION
			self.rpn_batch_size = cfg.TRAIN_RPN_BATCHSIZE
		elif mode == 'TEST':
			self.rpn_negative_overlap = cfg.TEST_RPN_NEGATIVE_OVERLAP
			self.rpn_positive_overlap = cfg.TEST_RPN_POSITIVE_OVERLAP
			self.rpn_fg_fraction = cfg.TEST_RPN_FG_FRACTION
			self.rpn_batch_size = cfg.TEST_RPN_BATCHSIZE
		else:
			raise ValueError('Unkown mode <%s> in rpnBuildTargetLayer...' % mode)
		self.num_anchors = self.anchors.size(0)
		self.allowed_border = 0
		self.bbox_inside_weights = 1.
	def forward(self, x):
		# prepare
		x_cls_pred, gt_boxes, img_info, num_gt_boxes = x
		batch_size = gt_boxes.size(0)
		feature_height, feature_width = x_cls_pred.size(2), x_cls_pred.size(3)
		# get shift
		shift_x = np.arange(0, feature_width) * self.feature_stride
		shift_y = np.arange(0, feature_height) * self.feature_stride
		shift_x, shift_y = np.meshgrid(shift_x, shift_y)
		shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose())
		shifts = shifts.contiguous().type_as(x_cls_pred).float()
		# get anchors
		anchors = self.anchors.type_as(gt_boxes)
		anchors = anchors.view(1, self.num_anchors, 4) + shifts.view(shifts.size(0), 1, 4)
		anchors = anchors.view(self.num_anchors*shifts.size(0), 4)
		total_anchors_ori = anchors.size(0)
		# make sure anchors are in the image
		keep_idxs = ((anchors[:, 0] >= -self.allowed_border) &
					 (anchors[:, 1] >= -self.allowed_border) &
					 (anchors[:, 2] < int(img_info[0][1])+self.allowed_border) &
					 (anchors[:, 3] < int(img_info[0][0])+self.allowed_border))
		keep_idxs = torch.nonzero(keep_idxs).view(-1)
		anchors = anchors[keep_idxs, :]
		# prepare labels: 1 is positive, 0 is negative, -1 means ignore
		labels = gt_boxes.new(batch_size, keep_idxs.size(0)).fill_(-1)
		bbox_inside_weights = gt_boxes.new(batch_size, keep_idxs.size(0)).zero_()
		bbox_outside_weights = gt_boxes.new(batch_size, keep_idxs.size(0)).zero_()
		# calc ious
		overlaps = BBoxFunctions.calcIoUs(anchors, gt_boxes)
		max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
		gt_max_overlaps, _ = torch.max(overlaps, 1)
		# assign labels
		labels[max_overlaps < self.rpn_negative_overlap] = 0
		gt_max_overlaps[gt_max_overlaps==0] = 1e-5
		keep_idxs_label = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)
		if torch.sum(keep_idxs_label) > 0:
			labels[keep_idxs_label > 0] = 1
		labels[max_overlaps >= self.rpn_positive_overlap] = 1
		max_num_fg = int(self.rpn_fg_fraction * self.rpn_batch_size)
		num_fg = torch.sum((labels == 1).int(), 1)
		num_bg = torch.sum((labels == 0).int(), 1)
		for i in range(batch_size):
			if num_fg[i] > max_num_fg:
				fg_idxs = torch.nonzero(labels[i] == 1).view(-1)
				rand_num = torch.from_numpy(np.random.permutation(fg_idxs.size(0))).type_as(gt_boxes).long()
				disable_idxs = fg_idxs[rand_num[:fg_idxs.size(0)-max_num_fg]]
				labels[i][disable_idxs] = -1
			max_num_bg = self.rpn_batch_size - torch.sum((labels == 1).int(), 1)[i]
			if num_bg[i] > max_num_bg:
				bg_idxs = torch.nonzero(labels[i] == 0).view(-1)
				rand_num = torch.from_numpy(np.random.permutation(bg_idxs.size(0))).type_as(gt_boxes).long()
				disable_idxs = bg_idxs[rand_num[:bg_idxs.size(0)-max_num_bg]]
				labels[i][disable_idxs] = -1
		offsets = torch.arange(0, batch_size) * gt_boxes.size(1)
		argmax_overlaps = argmax_overlaps + offsets.view(batch_size, 1).type_as(argmax_overlaps)
		gt_rois = gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5)
		bbox_targets = BBoxFunctions.encodeBboxes(anchors, gt_rois[..., :4])
		bbox_inside_weights[labels==1] = self.bbox_inside_weights
		num_examples = torch.sum(labels[i] >= 0)
		bbox_outside_weights[labels==1] = 1.0 / num_examples.item()
		bbox_outside_weights[labels==0] = 1.0 / num_examples.item()
		# unmap
		labels = rpnBuildTargetLayer.unmap(labels, total_anchors_ori, keep_idxs, batch_size, fill=-1)
		bbox_targets = rpnBuildTargetLayer.unmap(bbox_targets, total_anchors_ori, keep_idxs, batch_size, fill=0)
		bbox_inside_weights = rpnBuildTargetLayer.unmap(bbox_inside_weights, total_anchors_ori, keep_idxs, batch_size, fill=0)
		bbox_outside_weights = rpnBuildTargetLayer.unmap(bbox_outside_weights, total_anchors_ori, keep_idxs, batch_size, fill=0)
		# format return values
		outputs = []
		labels = labels.view(batch_size, feature_height, feature_width, self.num_anchors).permute(0, 3, 1, 2).contiguous()
		labels = labels.view(batch_size, 1, self.num_anchors*feature_height, feature_width)
		outputs.append(labels)
		bbox_targets = bbox_targets.view(batch_size, feature_height, feature_width, self.num_anchors*4).permute(0, 3, 1, 2).contiguous()
		outputs.append(bbox_targets)
		bbox_inside_weights = bbox_inside_weights.view(batch_size, total_anchors_ori, 1).expand(batch_size, total_anchors_ori, 4)
		bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, feature_height, feature_width, 4*self.num_anchors).permute(0, 3, 1, 2).contiguous()
		outputs.append(bbox_inside_weights)
		bbox_outside_weights = bbox_outside_weights.view(batch_size, total_anchors_ori, 1).expand(batch_size, total_anchors_ori, 4)
		bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, feature_height, feature_width, 4*self.num_anchors).permute(0, 3, 1, 2).contiguous()
		outputs.append(bbox_outside_weights)
		return outputs
	@staticmethod
	def unmap(data, count, inds, batch_size, fill=0):
		if data.dim() == 2:
			ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
			ret[:, inds] = data
		else:
			ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
			ret[:, inds, :] = data
		return ret
	def backward(self, *args):
		pass


'''region proposal net'''
class RegionProposalNet(nn.Module):
	def __init__(self, in_channels, feature_stride, mode, cfg, **kwargs):
		super(RegionProposalNet, self).__init__()
		# prepare
		self.anchor_scales = cfg.ANCHOR_SCALES
		self.anchor_ratios = cfg.ANCHOR_RATIOS
		self.feature_stride = feature_stride
		self.in_channels = in_channels
		self.mode = mode
		self.cfg = cfg
		# define rpn conv
		self.rpn_conv_trans = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
											nn.ReLU(inplace=True))
		self.out_channels_cls = len(self.anchor_scales) * len(self.anchor_ratios) * 2
		self.out_channels_loc = len(self.anchor_scales) * len(self.anchor_ratios) * 4
		self.rpn_conv_cls = nn.Conv2d(in_channels=512, out_channels=self.out_channels_cls, kernel_size=1, stride=1, padding=0)
		self.rpn_conv_loc = nn.Conv2d(in_channels=512, out_channels=self.out_channels_loc, kernel_size=1, stride=1, padding=0)
		# proposal layer
		self.rpn_proposal_layer = rpnProposalLayer(feature_stride=self.feature_stride, anchor_scales=self.anchor_scales, anchor_ratios=self.anchor_ratios, mode=self.mode, cfg=self.cfg)
		# build target layer
		self.rpn_build_target_layer = rpnBuildTargetLayer(feature_stride=self.feature_stride, anchor_scales=self.anchor_scales, anchor_ratios=self.anchor_ratios, mode=self.mode, cfg=self.cfg)
	def forward(self, x, gt_boxes, img_info, num_gt_boxes):
		batch_size = x.size(0)
		# do base classifiction and location
		x = self.rpn_conv_trans(x)
		x_cls = self.rpn_conv_cls(x)
		x_loc = self.rpn_conv_loc(x)
		# do softmax to get probs
		x_cls_reshape = x_cls.view(x_cls.size(0), 2, -1, x_cls.size(3))
		probs = F.softmax(x_cls_reshape, 1)
		probs = probs.view(x_cls.size())
		# get RoIs
		rois = self.rpn_proposal_layer((probs.data, x_loc.data, img_info))
		# define loss
		rpn_cls_loss = torch.Tensor([0]).type_as(x)
		rpn_loc_loss = torch.Tensor([0]).type_as(x)
		# while training, calculate loss
		if self.mode == 'TRAIN' and gt_boxes is not None:
			targets = self.rpn_build_target_layer((x_cls.data, gt_boxes, img_info, num_gt_boxes))
			# --classification loss
			cls_scores_pred = x_cls_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
			labels = targets[0].view(batch_size, -1)
			keep_idxs = labels.view(-1).ne(-1).nonzero().view(-1)
			cls_scores_pred_keep = torch.index_select(cls_scores_pred.view(-1, 2), 0, keep_idxs.data)
			labels_keep = torch.index_select(labels.view(-1), 0, keep_idxs.data)
			labels_keep = labels_keep.long()
			if self.cfg.RPN_CLS_LOSS_SET['type'] == 'cross_entropy':
				rpn_cls_loss = F.cross_entropy(cls_scores_pred_keep, labels_keep, size_average=self.cfg.RPN_CLS_LOSS_SET['cross_entropy']['size_average'])
				rpn_cls_loss = rpn_cls_loss * self.cfg.RPN_CLS_LOSS_SET['cross_entropy']['weight']
			else:
				raise ValueError('Unkown classification loss type <%s>...' % self.cfg.RPN_CLS_LOSS_SET['type'])
			# --regression loss
			bbox_targets, bbox_inside_weights, bbox_outside_weights = targets[1:]
			if self.cfg.RPN_REG_LOSS_SET['type'] == 'smoothL1Loss':
				rpn_loc_loss = smoothL1Loss(x_loc, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=3, dim=[1, 2, 3])
				rpn_loc_loss = rpn_loc_loss * self.cfg.RPN_REG_LOSS_SET['smoothL1Loss']['weight']
			elif self.cfg.RPN_REG_LOSS_SET['type'] == 'betaSmoothL1Loss':
				rpn_loc_loss = betaSmoothL1Loss(x_loc[bbox_inside_weights>0].view(-1, 4), bbox_targets[bbox_inside_weights>0].view(-1, 4), beta=self.cfg.RPN_REG_LOSS_SET['betaSmoothL1Loss']['beta'], size_average=self.cfg.RPN_REG_LOSS_SET['betaSmoothL1Loss']['size_average'])
				rpn_loc_loss = rpn_loc_loss * self.cfg.RPN_REG_LOSS_SET['betaSmoothL1Loss']['weight']
			else:
				raise ValueError('Unkown regression loss type <%s>...' % self.cfg.RPN_REG_LOSS_SET['type'])
		return rois, rpn_cls_loss, rpn_loc_loss
	'''
	Function:
		generate anchors.
	Input:
		--base_size(int): the base anchor size (16 in faster RCNN).
		--scales(list): scales for anchor boxes.
		--ratios(list): ratios for anchor boxes.
	Return:
		--anchors(np.array): [nA, 4], the format is (x1, y1, x2, y2).
	'''
	@staticmethod
	def generateAnchors(size_base=16, scales=2**np.arange(3, 6), ratios=[0.5, 1, 2]):
		def getWHCxCy(anchor):
			w = anchor[2] - anchor[0] + 1
			h = anchor[3] - anchor[1] + 1
			cx = anchor[0] + 0.5 * (w - 1)
			cy = anchor[1] + 0.5 * (h - 1)
			return w, h, cx, cy
		def makeAnchors(ws, hs, cx, cy):
			ws = ws[:, np.newaxis]
			hs = hs[:, np.newaxis]
			anchors = np.hstack((cx - 0.5 * (ws - 1),
								 cy - 0.5 * (hs - 1),
								 cx + 0.5 * (ws - 1),
								 cy + 0.5 * (hs - 1)))
			return anchors
		scales = np.array(scales)
		ratios = np.array(ratios)
		anchor_base = np.array([1, 1, size_base, size_base]) - 1
		w, h, cx, cy = getWHCxCy(anchor_base)
		size = w * h
		size_ratios = size / ratios
		ws = np.round(np.sqrt(size_ratios))
		hs = np.round(ws * ratios)
		anchors = makeAnchors(ws, hs, cx, cy)
		tmp = list()
		for i in range(anchors.shape[0]):
			w, h, cx, cy = getWHCxCy(anchors[i, :])
			ws = w * scales
			hs = h * scales
			tmp.append(makeAnchors(ws, hs, cx, cy))
		anchors = np.vstack(tmp)
		return torch.from_numpy(anchors).float()