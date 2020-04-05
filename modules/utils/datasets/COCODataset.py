'''
Function:
	for loading COCO2017 dataset
Author:
	Charles
'''
import os
import sys
import torch
import random
import numpy as np
from modules.utils import *
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.transforms import transforms
sys.path.append(os.path.join(os.getcwd(), 'libs/cocoapi/PythonAPI'))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


'''
Function:
	load coco2017 dataset
Return:
	--img: img data
	--gt_boxes: (x1, y1, x2, y2, label)
	--img_info: (height, width, scale_factor)
	--num_gt_boxes: number of gt_boxes
'''
class COCODataset(Dataset):
	def __init__(self, rootdir, image_size_dict, max_num_gt_boxes, use_color_jitter, img_norm_info, use_caffe_pretrained_model, mode, **kwargs):
		assert mode in ['TRAIN', 'TEST'], 'Unsupport COCODataset.mode <%s>...' % mode
		self.rootdir = rootdir
		self.image_size_dict = image_size_dict
		self.max_num_gt_boxes = max_num_gt_boxes
		self.use_color_jitter = use_color_jitter
		self.mode = mode
		self.datasettype = kwargs.get('datasettype')
		# prepare annotation file
		annfilepath = kwargs.get('annfilepath')
		if not annfilepath:
			annfilepath = COCODataset.getAnnFilePath(self.rootdir, self.datasettype)
		self.coco_api = COCO(annfilepath)
		# get categories and the maps
		categories = self.coco_api.loadCats(self.coco_api.getCatIds())
		self.clsnames = tuple(['__background__'] + [c['name'] for c in categories])
		self.num_classes = len(self.clsnames)
		self.clsnames2clsids_dict = dict(zip(self.clsnames, list(range(self.num_classes))))
		self.clsnames2cococlsids_dict = dict(zip([c['name'] for c in categories], self.coco_api.getCatIds()))
		self.clsids2cococlsids_dict = dict(zip(list(range(1, self.num_classes)), self.coco_api.getCatIds()))
		self.cococlsids2clsids_dict = dict(zip(self.coco_api.getCatIds(), list(range(1, self.num_classes))))
		# get all image ids
		self.img_ids = list(self.coco_api.imgToAnns.keys()) if self.datasettype in ['train2017', 'val2017'] else self.coco_api.getImgIds()
		# used to normalize image
		self.use_caffe_pretrained_model = use_caffe_pretrained_model
		self.img_norm_info = img_norm_info
		# used to make sure that all the img ids have ground-truths while training model.
		if mode == 'TRAIN':
			self.filtered_img_ids, self.img_ratios = self.preprocessImageIds()
		else:
			self.filtered_img_ids = self.img_ids
			self.img_ratios = list()
	'''get item'''
	def __getitem__(self, index):
		if self.mode == 'TRAIN':
			while True:
				# get image id
				img_id = self.filtered_img_ids[index]
				# get annotations, format is (x1, y1, x2, y2, label)
				width, height, bboxes = self.id2annotation(img_id)
				gt_boxes = np.array(bboxes)
				# calculate num_gt_boxes
				num_gt_boxes = torch.from_numpy(np.array([len(gt_boxes)]))
				# get img
				img_path = self.imgid2imgpath(img_id, self.datasettype)
				img = Image.open(img_path).convert('RGB')
				assert img.width == width, 'something may be wrong while reading image and annotation'
				assert img.height == height, 'something may be wrong while reading image and annotation'
				# preprocess img
				if random.random() > 0.5:
					img = ImageOps.mirror(img)
					gt_boxes[:, [0, 2]] = img.width - gt_boxes[:, [2, 0]] - 1
				img, scale_factor, target_size = COCODataset.preprocessImage(img, use_color_jitter=self.use_color_jitter, image_size_dict=self.image_size_dict, img_norm_info=self.img_norm_info, use_caffe_pretrained_model=self.use_caffe_pretrained_model)
				# img info: (h, w, scale_factor)
				img_info = torch.from_numpy(np.array([target_size[0], target_size[1], scale_factor]))
				# correct gt_boxes
				gt_boxes[..., :4] = gt_boxes[..., :4] * scale_factor
				# padding gt_boxes
				gt_boxes_padding = np.zeros((self.max_num_gt_boxes, 5), dtype=np.float32)
				gt_boxes_padding[range(len(gt_boxes))[:self.max_num_gt_boxes]] = gt_boxes[:self.max_num_gt_boxes]
				gt_boxes_padding = torch.from_numpy(gt_boxes_padding)
				# return necessary data
				return img_id, img, gt_boxes_padding, img_info, num_gt_boxes
		else:
			# get image id
			img_id = self.filtered_img_ids[index]
			# read img
			img_path = self.imgid2imgpath(img_id, self.datasettype)
			img = Image.open(img_path).convert('RGB')
			w_ori, h_ori = img.width, img.height
			img, scale_factor, target_size = COCODataset.preprocessImage(img, use_color_jitter=False, image_size_dict=self.image_size_dict, img_norm_info=self.img_norm_info, use_caffe_pretrained_model=self.use_caffe_pretrained_model)
			# img info: (h, w, scale_factor)
			img_info = torch.from_numpy(np.array([target_size[0], target_size[1], scale_factor]))
			# placeholder for gt_boxes_padding
			gt_boxes_padding = torch.from_numpy(np.array([1, 1, 1, 1, 0]))
			# placeholder for gt_boxes_padding
			num_gt_boxes = torch.from_numpy(np.array([0]))
			# return necessary data
			return img_id, img, w_ori, h_ori, gt_boxes_padding, img_info, num_gt_boxes
	'''calculate mmAP by using coco api'''
	def doDetectionEval(self, img_ids, resultsfilepath):
		coco_pred = self.coco_api.loadRes(resultsfilepath)
		coco_eval = COCOeval(self.coco_api, coco_pred, 'bbox')
		coco_eval.params.imgIds = img_ids
		coco_eval.evaluate()
		coco_eval.accumulate()
		coco_eval.summarize()
	'''convert image id to image path'''
	def imgid2imgpath(self, img_id, datasettype):
		filename = self.coco_api.loadImgs(img_id)[0]['file_name']
		if datasettype == 'train2017':
			imgpath = os.path.join(self.rootdir, 'train2017', filename)
		elif datasettype == 'val2017':
			imgpath = os.path.join(self.rootdir, 'val2017', filename)
		elif datasettype == 'testdev':
			imgpath = os.path.join(self.rootdir, 'test2017', filename)
		else:
			raise ValueError('COCODataset.imgid2imgpath unsupport datasettype <%s>...' % str(datasettype))
		assert os.path.exists(imgpath), 'Image path does not exist: {}...'.format(imgpath)
		return imgpath
	'''convert image id to annotation'''
	def id2annotation(self, index):
		img_annotation = self.coco_api.loadImgs(index)[0]
		width = img_annotation['width']
		height = img_annotation['height']
		annotation_ids = self.coco_api.getAnnIds(imgIds=index)
		targets = self.coco_api.loadAnns(annotation_ids)
		bboxes = []
		for target in targets:
			if 'bbox' in target:
				x1 = np.max((0, target['bbox'][0]))
				y1 = np.max((0, target['bbox'][1]))
				x2 = np.min((width-1, x1+np.max((0, target['bbox'][2]-1))))
				y2 = np.min((height-1, y1+np.max((0, target['bbox'][3]-1))))
				bbox = [x1, y1, x2, y2]
				if (bbox[2] <= bbox[0]) or (bbox[3] <= bbox[1]) or (target['area'] <= 0):
					continue
				label_idx = self.cococlsids2clsids_dict[target['category_id']]
				bbox.append(label_idx)
				bboxes += [bbox]
		return width, height, bboxes
	'''get image ids when using <TRAIN> mode'''
	def preprocessImageIds(self):
		filtered_img_ids = []
		img_ratios = []
		for img_id in self.img_ids:
			width, height, bboxes = self.id2annotation(img_id)
			if len(bboxes) > 0:
				filtered_img_ids.append(img_id)
				img_ratios.append(float(width/height))
		return filtered_img_ids, img_ratios
	'''calculate length'''
	def __len__(self):
		return len(self.filtered_img_ids)
	'''get annotation filepath'''
	@staticmethod
	def getAnnFilePath(rootdir, datasettype='train2017'):
		if datasettype == 'train2017':
			return os.path.join(rootdir, 'annotations/instances_train2017.json')
		elif datasettype == 'val2017':
			return os.path.join(rootdir, 'annotations/instances_val2017.json')
		elif datasettype == 'testdev':
			return os.path.join(rootdir, 'annotations/image_info_test-dev2017.json')
		else:
			raise ValueError('COCODataset.getAnnFilePath unsupport datasettype <%s>...' % str(datasettype))
	'''preprocess image, PIL.Image ——> torch.tensor'''
	@staticmethod
	def preprocessImage(img, use_color_jitter, image_size_dict, img_norm_info, use_caffe_pretrained_model):
		# calculate target_size and scale_factor, target_size's format is (h, w)
		w_ori, h_ori = img.width, img.height
		if w_ori > h_ori:
			target_size = (image_size_dict.get('SHORT_SIDE'), image_size_dict.get('LONG_SIDE'))
		else:
			target_size = (image_size_dict.get('LONG_SIDE'), image_size_dict.get('SHORT_SIDE'))
		h_t, w_t = target_size
		scale_factor = min(w_t/w_ori, h_t/h_ori)
		target_size = (round(scale_factor*h_ori), round(scale_factor*w_ori))
		# define and do transform
		if use_caffe_pretrained_model:
			means_norm = img_norm_info['caffe'].get('mean_rgb')
			stds_norm = img_norm_info['caffe'].get('std_rgb')
			if use_color_jitter:
				transform = transforms.Compose([transforms.Resize(target_size),
												transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
												transforms.ToTensor(),
												transforms.Normalize(mean=means_norm, std=stds_norm)])
			else:
				transform = transforms.Compose([transforms.Resize(target_size),
												transforms.ToTensor(),
												transforms.Normalize(mean=means_norm, std=stds_norm)])
			img = transform(img) * 255
			img = img[(2, 1, 0), :, :]
		else:
			means_norm = img_norm_info['pytorch'].get('mean_rgb')
			stds_norm = img_norm_info['pytorch'].get('std_rgb')
			if use_color_jitter:
				transform = transforms.Compose([transforms.Resize(target_size),
												transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
												transforms.ToTensor(),
												transforms.Normalize(mean=means_norm, std=stds_norm)])
			else:
				transform = transforms.Compose([transforms.Resize(target_size),
												transforms.ToTensor(),
												transforms.Normalize(mean=means_norm, std=stds_norm)])
			img = transform(img)
		# return necessary data
		return img, scale_factor, target_size
	'''padding collate fn'''
	@staticmethod
	def paddingCollateFn(data_batch):
		# data_batch: [[img_id, img(channel, height, width), gt_boxes, img_info, num_gt_boxes], ...]
		max_height = max([data[1].shape[1] for data in data_batch])
		max_width = max([data[1].shape[2] for data in data_batch])
		# get new data_batch
		img_id_batch = []
		img_batch = []
		gt_boxes_batch = []
		img_info_batch = []
		num_gt_boxes_batch = []
		for data in data_batch:
			img_id, img, gt_boxes, img_info, num_gt_boxes = data
			# (left, right, top, bottom)
			img_padding = F.pad(input=img, pad=(0, max_width-img.shape[2], 0, max_height-img.shape[1]))
			img_info[0] = max_height
			img_info[1] = max_width
			img_id_batch.append(torch.from_numpy(np.array([img_id])))
			img_batch.append(img_padding)
			gt_boxes_batch.append(gt_boxes)
			img_info_batch.append(img_info)
			num_gt_boxes_batch.append(num_gt_boxes)
		img_id_batch = torch.stack(img_id_batch, dim=0)
		img_batch = torch.stack(img_batch, dim=0)
		gt_boxes_batch = torch.stack(gt_boxes_batch, dim=0)
		img_info_batch = torch.stack(img_info_batch, dim=0)
		num_gt_boxes_batch = torch.stack(num_gt_boxes_batch, dim=0)
		return img_id_batch, img_batch, gt_boxes_batch, img_info_batch, num_gt_boxes_batch