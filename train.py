'''
Function:
	train the model
Author:
	Charles
'''
import torch
import warnings
import argparse
import torch.nn as nn
import torch.optim as optim
from modules.utils.utils import *
from modules.utils.datasets import *
from modules.fasterRCNN import FasterRCNNResNets
from cfgs.getcfg import getCfgByDatasetAndBackbone
warnings.filterwarnings("ignore")


'''parse arguments for training'''
def parseArgs():
	parser = argparse.ArgumentParser(description='Faster R-CNN')
	parser.add_argument('--datasetname', dest='datasetname', help='dataset for training.', default='', type=str, required=True)
	parser.add_argument('--backbonename', dest='backbonename', help='backbone network for training.', default='', type=str, required=True)
	parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to use.', default='', type=str)
	args = parser.parse_args()
	return args


'''train model'''
def train():
	# prepare base things
	args = parseArgs()
	cfg, cfg_file_path = getCfgByDatasetAndBackbone(datasetname=args.datasetname, backbonename=args.backbonename)
	checkDir(cfg.TRAIN_BACKUPDIR)
	logger_handle = Logger(cfg.TRAIN_LOGFILE)
	use_cuda = torch.cuda.is_available()
	is_multi_gpus = cfg.IS_MULTI_GPUS
	# prepare dataset
	if args.datasetname == 'coco':
		dataset = COCODataset(rootdir=cfg.DATASET_ROOT_DIR, image_size_dict=cfg.IMAGESIZE_DICT, max_num_gt_boxes=cfg.MAX_NUM_GT_BOXES, use_color_jitter=cfg.USE_COLOR_JITTER, img_norm_info=cfg.IMAGE_NORMALIZE_INFO, use_caffe_pretrained_model=cfg.USE_CAFFE_PRETRAINED_MODEL, mode='TRAIN', datasettype='train2017')
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCHSIZE, sampler=NearestRatioRandomSampler(dataset.img_ratios, cfg.BATCHSIZE), num_workers=cfg.NUM_WORKERS, collate_fn=COCODataset.paddingCollateFn, pin_memory=cfg.PIN_MEMORY)
	else:
		raise ValueError('Unsupport datasetname <%s> now...' % args.datasetname)
	# prepare model
	if args.backbonename.find('resnet') != -1:
		model = FasterRCNNResNets(mode='TRAIN', cfg=cfg, logger_handle=logger_handle)
	else:
		raise ValueError('Unsupport backbonename <%s> now...' % args.backbonename)
	start_epoch = 1
	end_epoch = cfg.MAX_EPOCHS
	if use_cuda:
		model = model.cuda()
	# prepare optimizer
	learning_rate_idx = 0
	if cfg.IS_USE_WARMUP:
		learning_rate = cfg.LEARNING_RATES[learning_rate_idx] / 3
	else:
		learning_rate = cfg.LEARNING_RATES[learning_rate_idx]
	optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
	# check checkpoints path
	if args.checkpointspath:
		checkpoints = loadCheckpoints(args.checkpointspath, logger_handle)
		model.load_state_dict(checkpoints['model'])
		optimizer.load_state_dict(checkpoints['optimizer'])
		start_epoch = checkpoints['epoch'] + 1
		for epoch in range(1, start_epoch):
			if epoch in cfg.LR_ADJUST_EPOCHS:
				learning_rate_idx += 1
	# data parallel
	if is_multi_gpus:
		model = nn.DataParallel(model)
	# print config
	logger_handle.info('Dataset used: %s, Number of images: %s' % (args.datasetname, len(dataset)))
	logger_handle.info('Backbone used: %s' % args.backbonename)
	logger_handle.info('Checkpoints used: %s' % args.checkpointspath)
	logger_handle.info('Config file used: %s' % cfg_file_path)
	# train
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	for epoch in range(start_epoch, end_epoch+1):
		# --set train mode
		if is_multi_gpus:
			model.module.setTrain()
		else:
			model.setTrain()
		# --adjust learning rate
		if epoch in cfg.LR_ADJUST_EPOCHS:
			learning_rate_idx += 1
			adjustLearningRate(optimizer=optimizer, target_lr=cfg.LEARNING_RATES[learning_rate_idx], logger_handle=logger_handle)
		# --log info
		logger_handle.info('Start epoch %s, learning rate is %s...' % (epoch, cfg.LEARNING_RATES[learning_rate_idx]))
		# --train epoch
		for batch_idx, samples in enumerate(dataloader):
			if (epoch == 1) and (cfg.IS_USE_WARMUP) and (batch_idx == cfg.NUM_WARMUP_STEPS):
				assert learning_rate_idx == 0, 'BUGS may exist...'
				adjustLearningRate(optimizer=optimizer, target_lr=cfg.LEARNING_RATES[learning_rate_idx], logger_handle=logger_handle)
			optimizer.zero_grad()
			img_ids, imgs, gt_boxes, img_info, num_gt_boxes = samples
			output = model(x=imgs.type(FloatTensor), gt_boxes=gt_boxes.type(FloatTensor), img_info=img_info.type(FloatTensor), num_gt_boxes=num_gt_boxes.type(FloatTensor))
			rois, cls_probs, bbox_preds, rpn_cls_loss, rpn_loc_loss, loss_cls, loss_loc = output
			loss = rpn_cls_loss.mean() + rpn_loc_loss.mean() + loss_cls.mean() + loss_loc.mean()
			logger_handle.info('[EPOCH]: %s/%s, [BTACH]: %s/%s, [LEARNING_RATE]: %s, [DATASET]: %s \n\t [LOSS]: rpn_cls_loss %.4f, rpn_loc_loss %.4f, loss_cls %.4f, loss_loc %.4f, total %.4f' % \
								(epoch, end_epoch, (batch_idx+1), len(dataloader), cfg.LEARNING_RATES[learning_rate_idx], args.datasetname, rpn_cls_loss.mean().item(), rpn_loc_loss.mean().item(), loss_cls.mean().item(), loss_loc.mean().item(), loss.item()))
			loss.backward()
			optimizer.step()
		# --save model
		if (epoch % cfg.SAVE_INTERVAL == 0) or (epoch == end_epoch):
			state_dict = {'epoch': epoch,
						  'model': model.module.state_dict() if is_multi_gpus else model.state_dict(),
						  'optimizer': optimizer.state_dict()}
			savepath = os.path.join(cfg.TRAIN_BACKUPDIR, 'epoch_%s.pth' % epoch)
			saveCheckpoints(state_dict, savepath, logger_handle)


'''run'''
if __name__ == '__main__':
	train()