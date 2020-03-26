'''
Function:
	define the faster RCNN
Author:
	Charles
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.backbones import *
from modules.utils.utils import *
from modules.losses.smoothL1 import *
from modules.RPN import RegionProposalNet
from libs.roi_pool.roi_pool import roi_pool
from libs.roi_align.roi_align import roi_align


'''build proposal target layer'''
class buildProposalTargetLayer(nn.Module):
	def __init__(self, mode, cfg, **kwargs):
		super(buildProposalTargetLayer, self).__init__()
		self.mode = mode
		self.cfg = cfg
		if mode == 'TRAIN':
			self.bbox_normalize_means = torch.FloatTensor(cfg.TRAIN_BBOX_NORMALIZE_MEANS)
			self.bbox_normalize_stds = torch.FloatTensor(cfg.TRAIN_BBOX_NORMALIZE_STDS)
			self.roi_batchsize = cfg.TRAIN_ROI_BATCHSIZE
			self.roi_fg_fraction = cfg.TRAIN_ROI_FG_FRACTION
			self.roi_fg_thresh = cfg.TRAIN_ROI_FG_THRESH
			self.roi_bg_thresh_hi = cfg.TRAIN_ROI_BG_THRESH_HI
			self.roi_bg_thresh_lo = cfg.TRAIN_ROI_BG_THRESH_LO
		elif mode == 'TEST':
			self.bbox_normalize_means = torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_MEANS)
			self.bbox_normalize_stds = torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_STDS)
			self.roi_batchsize = cfg.TEST_ROI_BATCHSIZE
			self.roi_fg_fraction = cfg.TEST_ROI_FG_FRACTION
			self.roi_fg_thresh = cfg.TEST_ROI_FG_THRESH
			self.roi_bg_thresh_hi = cfg.TEST_ROI_BG_THRESH_HI
			self.roi_bg_thresh_lo = cfg.TEST_ROI_BG_THRESH_LO
		else:
			raise ValueError('Unkown mode <%s> in buildProposalTargetLayer...' % mode)
	'''forward'''
	def forward(self, all_rois, gt_boxes, num_gt_boxes):
		self.bbox_normalize_means = self.bbox_normalize_means.type_as(gt_boxes)
		self.bbox_normalize_stds = self.bbox_normalize_stds.type_as(gt_boxes)
		gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
		gt_boxes_append[..., 1:5] = gt_boxes[..., :4]
		all_rois = torch.cat([all_rois, gt_boxes_append], 1)
		num_rois_per_image = self.roi_batchsize
		num_rois_fg_per_image = int(np.round(self.roi_fg_fraction * num_rois_per_image))
		rois, labels, bbox_targets = self.sampleRoIs(all_rois, gt_boxes, num_rois_fg_per_image, num_rois_per_image)
		return rois, labels, bbox_targets
	'''sample from proposals and obtain rois'''
	def sampleRoIs(self, all_rois, gt_boxes, num_rois_fg_per_image, num_rois_per_image):
		overlaps = BBoxFunctions.calcIoUs(all_rois, gt_boxes)
		max_overlaps, gt_assignment = torch.max(overlaps, 2)
		batch_size = overlaps.size(0)
		offsets = torch.arange(0, batch_size) * gt_boxes.size(1)
		offsets = offsets.view(-1, 1).type_as(gt_assignment) + gt_assignment
		labels = gt_boxes[..., 4].contiguous().view(-1)[(offsets.view(-1),)].view(batch_size, -1)
		labels_batch = labels.new(batch_size, num_rois_per_image).zero_()
		rois_batch = all_rois.new(batch_size, num_rois_per_image, 5).zero_()
		gt_rois_batch = all_rois.new(batch_size, num_rois_per_image, 5).zero_()
		for i in range(batch_size):
			fg_idxs = torch.nonzero(max_overlaps[i] >= self.roi_fg_thresh).view(-1)
			num_rois_fg = fg_idxs.numel()
			bg_idxs = torch.nonzero((max_overlaps[i] < self.roi_bg_thresh_hi) & (max_overlaps[i] >= self.roi_bg_thresh_lo)).view(-1)
			num_rois_bg = bg_idxs.numel()
			if num_rois_fg > 0 and num_rois_bg > 0:
				num_rois_fg_this_image = min(num_rois_fg_per_image, num_rois_fg)
				rand_num = torch.from_numpy(np.random.permutation(num_rois_fg)).type_as(gt_boxes).long()
				fg_idxs = fg_idxs[rand_num[:num_rois_fg_this_image]]
				num_rois_bg_this_image = num_rois_per_image - num_rois_fg_this_image
				rand_num = np.floor(np.random.rand(num_rois_bg_this_image) * num_rois_bg)
				rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
				bg_idxs = bg_idxs[rand_num]
			elif num_rois_fg > 0 and num_rois_bg == 0:
				rand_num = np.floor(np.random.rand(num_rois_per_image) * num_rois_fg)
				rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
				fg_idxs = fg_idxs[rand_num]
				num_rois_fg_this_image = num_rois_per_image
				num_rois_bg_this_image = 0
			elif num_rois_fg == 0 and num_rois_bg > 0:
				rand_num = np.floor(np.random.rand(num_rois_per_image) * num_rois_bg)
				rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
				bg_idxs = bg_idxs[rand_num]
				num_rois_fg_this_image = 0
				num_rois_bg_this_image = num_rois_per_image
			else:
				raise ValueError('num_rois_fg = 0 and num_rois_bg = 0, your program should be wrong somewhere...')
			keep_idxs = torch.cat([fg_idxs, bg_idxs], 0)
			labels_batch[i].copy_(labels[i][keep_idxs])
			if num_rois_fg_this_image < num_rois_per_image:
				labels_batch[i][num_rois_fg_this_image:] = 0
			rois_batch[i] = all_rois[i][keep_idxs]
			rois_batch[i, :, 0] = i
			gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_idxs]]
		bbox_targets = self.computeTargets(rois_batch[..., 1:5], gt_rois_batch[..., :4])
		bbox_targets = self.getBBoxRegressionTargets(bbox_targets, labels_batch)
		return rois_batch, labels_batch, bbox_targets
	'''compute targets'''
	def computeTargets(self, ex_rois, gt_rois):
		assert ex_rois.size(1) == gt_rois.size(1)
		assert ex_rois.size(2) == 4
		assert gt_rois.size(2) == 4
		batch_size = ex_rois.size(0)
		num_rois_per_image = ex_rois.size(1)
		targets = BBoxFunctions.encodeBboxes(ex_rois, gt_rois)
		targets = ((targets - self.bbox_normalize_means.expand_as(targets)) / self.bbox_normalize_stds.expand_as(targets))
		return targets
	'''get reg targets'''
	def getBBoxRegressionTargets(self, bbox_targets, labels_batch):
		batch_size = labels_batch.size(0)
		num_rois_per_image = labels_batch.size(1)
		bbox_targets_new = bbox_targets.new(batch_size, num_rois_per_image, 4).zero_()
		for b in range(batch_size):
			if labels_batch[b].sum() == 0:
				continue
			idxs = torch.nonzero(labels_batch[b] > 0).view(-1)
			for i in range(idxs.numel()):
				idx = idxs[i]
				bbox_targets_new[b, idx, :] = bbox_targets[b, idx, :]
		return bbox_targets_new
	'''disable backward'''
	def backward(self, *args):
		pass


'''base model for faster rcnn'''
class fasterRCNNBase(nn.Module):
	def __init__(self, feature_stride, mode, cfg, **kwargs):
		super(fasterRCNNBase, self).__init__()
		self.num_classes = cfg.NUM_CLASSES
		self.is_class_agnostic = cfg.IS_CLASS_AGNOSTIC
		self.feature_stride = feature_stride
		self.mode = mode
		self.cfg = cfg
		if self.mode == 'TRAIN':
			self.pooling_method = cfg.TRAIN_POOLING_METHOD
			self.pooling_size = cfg.TRAIN_POOLING_SIZE
			self.pooling_sample_num = cfg.TRAIN_POOLING_SAMPLE_NUM
		elif self.mode == 'TEST':
			self.pooling_method = cfg.TEST_POOLING_METHOD
			self.pooling_size = cfg.TEST_POOLING_SIZE
			self.pooling_sample_num = cfg.TEST_POOLING_SAMPLE_NUM
		else:
			raise ValueError('Unkown mode <%s> in fasterRCNNBase...' % mode)
		# base model
		self.base_model = None
		# RPN
		self.rpn_net = None
		self.build_proposal_target_layer = None
		# top model
		self.top_model = None
		# final results
		self.fc_cls = None
		self.fc_reg = None
	'''forward'''
	def forward(self, x, gt_boxes, img_info, num_gt_boxes):
		batch_size = x.size(0)
		# extract features using backbone network
		x = self.base_model(x)
		# obtain rois
		rois, rpn_cls_loss, rpn_reg_loss = self.rpn_net(x, gt_boxes, img_info, num_gt_boxes)
		# if train
		if self.mode == 'TRAIN' and gt_boxes is not None:
			rois, rois_labels, rois_bbox_targets = self.build_proposal_target_layer(rois, gt_boxes, num_gt_boxes)
			rois_labels = rois_labels.view(-1).long()
			rois_bbox_targets = rois_bbox_targets.view(-1, rois_bbox_targets.size(2))
		else:
			rois_labels = None
			rois_bbox_targets = None
		# roi pooling based on obtained rois
		if self.pooling_method == 'align':
			pooled_features = roi_align(x, rois.view(-1, 5), self.pooling_size, 1./self.feature_stride, self.pooling_sample_num)
		elif self.pooling_method == 'pool':
			pooled_features = roi_pool(x, rois.view(-1, 5), self.pooling_size, 1./self.feature_stride, self.pooling_sample_num)
		else:
			raise ValueError('Unkown pooling_method <%s> in fasterRCNNBase...' % self.pooling_method)
		# feed into top model
		pooled_features = self.top_model(pooled_features)
		if len(pooled_features.size()) == 4:
			pooled_features = pooled_features.mean(3).mean(2)
		# do regression
		x_reg = self.fc_reg(pooled_features)
		if self.mode == 'TRAIN' and not self.is_class_agnostic:
			x_reg = x_reg.view(x_reg.size(0), -1, 4)
			x_reg = torch.gather(x_reg, 1, rois_labels.view(rois_labels.size(0), 1, 1).expand(rois_labels.size(0), 1, 4))
			x_reg = x_reg.squeeze(1)
		# do classification
		x_cls = self.fc_cls(pooled_features)
		cls_probs = F.softmax(x_cls, 1)
		# calculate loss
		loss_cls = torch.Tensor([0]).type_as(x)
		loss_reg = torch.Tensor([0]).type_as(x)
		if self.mode == 'TRAIN':
			# --classification loss
			if self.cfg.RCNN_CLS_LOSS_SET['type'] == 'cross_entropy':
				loss_cls = F.cross_entropy(x_cls, rois_labels, size_average=self.cfg.RCNN_CLS_LOSS_SET['cross_entropy']['size_average'])
				loss_cls = loss_cls * self.cfg.RCNN_CLS_LOSS_SET['cross_entropy']['weight']
			else:
				raise ValueError('Unkown classification loss type <%s>...' % self.cfg.RCNN_CLS_LOSS_SET['type'])
			# --regression loss
			if self.cfg.RCNN_REG_LOSS_SET['type'] == 'betaSmoothL1Loss':
				mask = rois_labels.unsqueeze(1).expand(rois_labels.size(0), 4)
				loss_reg = betaSmoothL1Loss(x_reg[mask>0].view(-1, 4), 
											rois_bbox_targets[mask>0].view(-1, 4), 
											beta=self.cfg.RCNN_REG_LOSS_SET['betaSmoothL1Loss']['beta'], 
											size_average=self.cfg.RCNN_REG_LOSS_SET['betaSmoothL1Loss']['size_average'],
											loss_weight=self.cfg.RCNN_REG_LOSS_SET['betaSmoothL1Loss']['weight'])
			else:
				raise ValueError('Unkown regression loss type <%s>...' % self.cfg.RCNN_REG_LOSS_SET['type'])
		cls_probs = cls_probs.view(batch_size, rois.size(1), -1)
		bbox_preds = x_reg.view(batch_size, rois.size(1), -1)
		return rois, cls_probs, bbox_preds, rpn_cls_loss, rpn_reg_loss, loss_cls, loss_reg
	'''initialize except for backbone network'''
	def initializeAddedModules(self, init_method):
		raise NotImplementedError('fasterRCNNBase.initializeAddedModules is not implemented...')
	'''set bn fixed'''
	@staticmethod
	def setBnFixed(m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm') != -1:
			for p in m.parameters():
				p.requires_grad = False
	'''set bn eval'''
	@staticmethod
	def setBnEval(m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm') != -1:
			m.eval()


'''faster rcnn using resnet backbones'''
class FasterRCNNResNets(fasterRCNNBase):
	feature_stride = 16.0
	def __init__(self, mode, cfg, logger_handle, **kwargs):
		fasterRCNNBase.__init__(self, FasterRCNNResNets.feature_stride, mode, cfg)
		self.logger_handle = logger_handle
		self.backbone_type = cfg.BACKBONE_TYPE
		self.pretrained_model_path = cfg.PRETRAINED_MODEL_PATH
		self.backbone = ResNets(resnet_type=self.backbone_type, pretrained=False)
		if cfg.WEIGHTS_NEED_INITIALIZE and mode == 'TRAIN':
			self.initializeBackbone()
		self.backbone.avgpool = None
		self.backbone.fc = None
		# base model
		self.base_model = nn.Sequential(*[self.backbone.conv1,
										  self.backbone.bn1,
										  self.backbone.relu,
										  self.backbone.maxpool,
										  self.backbone.layer1,
										  self.backbone.layer2,
										  self.backbone.layer3])
		# RPN
		in_channels = 256 if self.backbone_type in ['resnet18', 'resnet34'] else 1024
		self.rpn_net = RegionProposalNet(in_channels=in_channels, feature_stride=self.feature_stride, mode=mode, cfg=cfg)
		self.build_proposal_target_layer = buildProposalTargetLayer(mode, cfg)
		# define top model
		self.top_model = nn.Sequential(*[self.backbone.layer4])
		# final results
		in_features = 512 if self.backbone_type in ['resnet18', 'resnet34'] else 2048
		self.fc_cls = nn.Linear(in_features, self.num_classes)
		if self.is_class_agnostic:
			self.fc_reg = nn.Linear(in_features, 4)
		else:
			self.fc_reg = nn.Linear(in_features, 4*self.num_classes)
		if cfg.ADDED_MODULES_WEIGHT_INIT_METHOD and mode == 'TRAIN':
			self.initializeAddedModules(cfg.ADDED_MODULES_WEIGHT_INIT_METHOD)
		# fix some first layers following original implementation
		if cfg.FIXED_FRONT_BLOCKS:
			for p in self.base_model[0].parameters():
				p.requires_grad = False
			for p in self.base_model[1].parameters():
				p.requires_grad = False
			for p in self.base_model[4].parameters():
				p.requires_grad = False
		self.base_model.apply(fasterRCNNBase.setBnFixed)
		self.top_model.apply(fasterRCNNBase.setBnFixed)
	'''set train mode'''
	def setTrain(self):
		nn.Module.train(self, True)
		if self.cfg.FIXED_FRONT_BLOCKS:
			self.base_model.eval()
			self.base_model[5].train()
			self.base_model[6].train()
		self.base_model.apply(fasterRCNNBase.setBnEval)
		self.top_model.apply(fasterRCNNBase.setBnEval)
	'''initialize model'''
	def initializeBackbone(self):
		if self.pretrained_model_path:
			self.backbone.load_state_dict({k:v for k,v in torch.load(self.pretrained_model_path).items() if k in self.backbone.state_dict()})
			self.logger_handle.info('Loading pretrained weights from %s for backbone network...' % self.pretrained_model_path)
		else:
			self.backbone = ResNets(resnet_type=self.backbone_type, pretrained=True)