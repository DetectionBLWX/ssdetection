'''
Function:
	detect objects in one image
Author:
	Charles
'''
import torch
import argparse
import numpy as np
from modules.utils.utils import *
from modules.utils.datasets import *
from libs.nms.nms_wrapper import nms
from PIL import Image, ImageDraw, ImageFont
from modules.fasterRCNN import FasterRCNNResNets
from cfgs.getcfg import getCfgByDatasetAndBackbone


'''parse arguments for training'''
def parseArgs():
	parser = argparse.ArgumentParser(description='Faster R-CNN')
	parser.add_argument('--imagepath', dest='imagepath', help='image you want to detect.', default='', type=str, required=True)
	parser.add_argument('--backbonename', dest='backbonename', help='backbone network for demo.', default='', type=str, required=True)
	parser.add_argument('--datasetname', dest='datasetname', help='dataset used to train.', default='', type=str, required=True)
	parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to use.', default='', type=str, required=True)
	parser.add_argument('--nmsthresh', dest='nmsthresh', help='thresh used in nms.', default=0.5, type=float)
	parser.add_argument('--confthresh', dest='confthresh', help='thresh used in showing bounding box.', default=0.5, type=float)
	args = parser.parse_args()
	return args


'''detect objects in one image'''
def demo():
	# prepare base things
	args = parseArgs()
	cfg, cfg_file_path = getCfgByDatasetAndBackbone(datasetname=args.datasetname, backbonename=args.backbonename)
	checkDir(cfg.TEST_BACKUPDIR)
	logger_handle = Logger(cfg.TEST_LOGFILE)
	use_cuda = torch.cuda.is_available()
	clsnames = loadclsnames(cfg.CLSNAMESPATH)
	# prepare model
	if args.backbonename.find('resnet') != -1:
		model = FasterRCNNResNets(mode='TEST', cfg=cfg, logger_handle=logger_handle)
	else:
		raise ValueError('Unsupport backbonename <%s> now...' % args.backbonename)
	if use_cuda:
		model = model.cuda()
	# load checkpoints
	checkpoints = loadCheckpoints(args.checkpointspath, logger_handle)
	model.load_state_dict(checkpoints['model'])
	model.eval()
	# do detect
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	img = Image.open(args.imagepath)
	if args.datasetname == 'coco':
		input_img, scale_factor, target_size = COCODataset.preprocessImage(img, use_color_jitter=False, image_size_dict=cfg.IMAGESIZE_DICT, img_norm_info=cfg.IMAGE_NORMALIZE_INFO, use_caffe_pretrained_model=cfg.USE_CAFFE_PRETRAINED_MODEL)
	else:
		raise ValueError('Unsupport datasetname <%s> now...' % args.datasetname)
	input_img = input_img.unsqueeze(0).type(FloatTensor)
	gt_boxes = torch.FloatTensor([1, 1, 1, 1, 0]).unsqueeze(0).type(FloatTensor)
	img_info = torch.from_numpy(np.array([target_size[0], target_size[1], scale_factor])).unsqueeze(0).type(FloatTensor)
	num_gt_boxes = torch.FloatTensor([0]).unsqueeze(0).type(FloatTensor)
	with torch.no_grad():
		output = model(x=input_img, gt_boxes=gt_boxes, img_info=img_info, num_gt_boxes=num_gt_boxes)
	rois = output[0].data[..., 1:5]
	cls_probs = output[1].data
	bbox_preds = output[2].data
	# parse the results
	if cfg.IS_CLASS_AGNOSTIC:
		box_deltas = bbox_preds.view(-1, 4) * torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_STDS).type(FloatTensor) + torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_MEANS).type(FloatTensor)
		box_deltas = box_deltas.view(1, -1, 4)
	else:
		box_deltas = bbox_preds.view(-1, 4) * torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_STDS).type(FloatTensor) + torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_MEANS).type(FloatTensor)
		box_deltas = box_deltas.view(1, -1, 4*cfg.NUM_CLASSES)
	boxes_pred = BBoxFunctions.decodeBboxes(rois, box_deltas)
	boxes_pred = BBoxFunctions.clipBoxes(boxes_pred, img_info.data)
	boxes_pred = boxes_pred.squeeze()
	scores = cls_probs.squeeze()
	thresh = 0.05
	for j in range(1, cfg.NUM_CLASSES):
		idxs = torch.nonzero(scores[:, j] > thresh).view(-1)
		if idxs.numel() > 0:
			cls_scores = scores[:, j][idxs]
			_, order = torch.sort(cls_scores, 0, True)
			if cfg.IS_CLASS_AGNOSTIC:
				cls_boxes = boxes_pred[idxs, :]
			else:
				cls_boxes = boxes_pred[idxs][:, j*4: (j+1)*4]
			cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
			cls_dets = cls_dets[order]
			keep_idxs = nms(cls_dets, args.nmsthresh, force_cpu=False)
			cls_dets = cls_dets[keep_idxs.view(-1).long()]
			for cls_det in cls_dets:
				if cls_det[-1] > args.confthresh:
					x1, y1, x2, y2 = cls_det[:4]
					x1 = x1.item() / scale_factor
					x2 = x2.item() / scale_factor
					y1 = y1.item() / scale_factor
					y2 = y2.item() / scale_factor
					label = clsnames[j-1]
					logger_handle.info('Detect a %s in confidence %.4f...' % (label, cls_det[-1].item()))
					color = (0, 255, 0)
					draw = ImageDraw.Draw(img)
					draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill=color)
					font = ImageFont.truetype('libs/font.TTF', 25)
					draw.text((x1+5, y1), label, fill=color, font=font)
	img.save(os.path.join(cfg.TEST_BACKUPDIR, 'demo_output.jpg'))


'''run'''
if __name__ == '__main__':
	demo()