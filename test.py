'''
Function:
	test mAP
Author:
	Charles
'''
import json
import torch
import argparse
import numpy as np
from modules.utils.utils import *
from modules.utils.datasets import *
from libs.nms.nms_wrapper import nms
from modules.fasterRCNN import FasterRCNNResNets
from cfgs.getcfg import getCfgByDatasetAndBackbone


'''parse arguments for training'''
def parseArgs():
	parser = argparse.ArgumentParser(description='Faster R-CNN')
	parser.add_argument('--datasetname', dest='datasetname', help='dataset for testing.', default='', type=str, required=True)
	parser.add_argument('--annfilepath', dest='annfilepath', help='used to specify annfilepath.', default='', type=str)
	parser.add_argument('--datasettype', dest='datasettype', help='used to specify datasettype.', default='val2017', type=str)
	parser.add_argument('--backbonename', dest='backbonename', help='backbone network for testing.', default='', type=str, required=True)
	parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to use.', default='', type=str, required=True)
	parser.add_argument('--nmsthresh', dest='nmsthresh', help='thresh used in nms.', default=0.5, type=float)
	args = parser.parse_args()
	return args


'''test mAP'''
def test():
	# prepare base things
	args = parseArgs()
	cfg, cfg_file_path = getCfgByDatasetAndBackbone(datasetname=args.datasetname, backbonename=args.backbonename)
	checkDir(cfg.TEST_BACKUPDIR)
	logger_handle = Logger(cfg.TEST_LOGFILE)
	use_cuda = torch.cuda.is_available()
	clsnames = loadclsnames(cfg.CLSNAMESPATH)
	# prepare dataset
	if args.datasetname == 'coco':
		dataset = COCODataset(rootdir=cfg.DATASET_ROOT_DIR, image_size_dict=cfg.IMAGESIZE_DICT, max_num_gt_boxes=-1, use_color_jitter=False, img_norm_info=cfg.IMAGE_NORMALIZE_INFO, use_caffe_pretrained_model=cfg.USE_CAFFE_PRETRAINED_MODEL, mode='TEST', datasettype=args.datasettype, annfilepath=args.annfilepath)
	else:
		raise ValueError('Unsupport datasetname <%s> now...' % args.datasetname)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
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
	# test mAP
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	results = []
	img_ids = []
	for batch_idx, samples in enumerate(dataloader):
		logger_handle.info('detect %s/%s...' % (batch_idx+1, len(dataloader)))
		# --do detect
		img_id, img, w_ori, h_ori, gt_boxes, img_info, num_gt_boxes = samples
		img_id, w_ori, h_ori, scale_factor = int(img_id.item()), w_ori.item(), h_ori.item(), img_info[0][-1].item()
		img_ids.append(img_id)
		with torch.no_grad():
			output = model(x=img.type(FloatTensor), gt_boxes=gt_boxes.type(FloatTensor), img_info=img_info.type(FloatTensor), num_gt_boxes=num_gt_boxes.type(FloatTensor))
		rois = output[0].data[..., 1:5]
		cls_probs = output[1].data
		bbox_preds = output[2].data
		# --parse the results
		if cfg.IS_CLASS_AGNOSTIC:
			box_deltas = bbox_preds.view(-1, 4) * torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_STDS).type(FloatTensor) + torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_MEANS).type(FloatTensor)
			box_deltas = box_deltas.view(1, -1, 4)
		else:
			box_deltas = bbox_preds.view(-1, 4) * torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_STDS).type(FloatTensor) + torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_MEANS).type(FloatTensor)
			box_deltas = box_deltas.view(1, -1, 4*cfg.NUM_CLASSES)
		boxes_pred = BBoxFunctions.decodeBboxes(rois, box_deltas)
		boxes_pred = BBoxFunctions.clipBoxes(boxes_pred, torch.from_numpy(np.array([h_ori*scale_factor, w_ori*scale_factor, scale_factor])).unsqueeze(0).type(FloatTensor).data)
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
					category_id = dataset.clsids2cococlsids_dict.get(j)
					x1, y1, x2, y2, score = cls_det
					x1 = x1.item() / scale_factor
					x2 = x2.item() / scale_factor
					y1 = y1.item() / scale_factor
					y2 = y2.item() / scale_factor
					bbox = [x1, y1, x2, y2]
					bbox[2] = bbox[2] - bbox[0]
					bbox[3] = bbox[3] - bbox[1]
					image_result = {
									'image_id': img_id,
									'category_id': int(category_id),
									'score': float(score.item()),
									'bbox': bbox
								}
					results.append(image_result)
	json.dump(results, open(cfg.TEST_BBOXES_SAVE_PATH, 'w'), indent=4)
	if args.datasettype in ['val2017']:
		dataset.doDetectionEval(img_ids, cfg.TEST_BBOXES_SAVE_PATH)


'''run'''
if __name__ == '__main__':
	test()