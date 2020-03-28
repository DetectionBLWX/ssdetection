# FasterRCNN
```
Pytorch Implementation of FasterRCNN.
You can star this repository to keep track of the project if it's helpful for you, thank you for your support.
```


# Environment
```
OS: Ubuntu 16.04
Python: python3.x with torch==1.2.0, torchvision==0.4.0
```


# Performance
|  Backbone   | Train       |  Test         |  Pretrained Model  |  Epochs  |	Learning Rate		|   RoI per image   |   AP                                                    |
|  :----:     | :----:      |  :----:       |  :----:    	     |	:----:  |	:----:				|   :----:  	    |	:----	                                              |
| ResNet-18   | trainval35k |  minival5k    |  Pytorch		     |	12 	    |	2e-2/2e-3/2e-4	    |	512			    |   [27.1](PerformanceDetails/Res18_pytorch_epoch12.MD)   |
| ResNet-34   | trainval35k |  minival5k    |  Pytorch		     |	12 	    |	2e-2/2e-3/2e-4	    |	512			    |                                                         |
| ResNet-50   | trainval35k |  minival5k    |  Pytorch		     |	12	    |	2e-2/2e-3/2e-4   	|	512			    |                                                         |
| ResNet-101  | trainval35k |  minival5k    |  Pytorch   	     |	12	    |	2e-2/2e-3/2e-4		|	512			    |                                                         |


# Trained models
```
You could get the trained models reported above at 
https://drive.google.com/open?id=1JYs4r1M6doRlMgKCxSWmue2iKAcMkJxe
```


# Usage
#### Setup
```
cd libs
sh make.sh
```
#### Train
```
usage: train.py [-h] --datasetname DATASETNAME --backbonename BACKBONENAME
                [--checkpointspath CHECKPOINTSPATH]
optional arguments:
  -h, --help            show this help message and exit
  --datasetname DATASETNAME
                        dataset for training.
  --backbonename BACKBONENAME
                        backbone network for training.
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to use.
cmd example:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --datasetname coco --backbonename resnet50
```
#### Test
```
usage: test.py [-h] --datasetname DATASETNAME [--annfilepath ANNFILEPATH]
               [--datasettype DATASETTYPE] --backbonename BACKBONENAME
               --checkpointspath CHECKPOINTSPATH [--nmsthresh NMSTHRESH]
optional arguments:
  -h, --help            show this help message and exit
  --datasetname DATASETNAME
                        dataset for testing.
  --annfilepath ANNFILEPATH
                        used to specify annfilepath.
  --datasettype DATASETTYPE
                        used to specify datasettype.
  --backbonename BACKBONENAME
                        backbone network for testing.
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to use.
  --nmsthresh NMSTHRESH
                        thresh used in nms.
cmd example:
CUDA_VISIBLE_DEVICES=0 python test.py --checkpointspath faster_res50_trainbackup_coco/epoch_12.pth --datasetname coco --backbonename resnet50
```
#### Demo
```
usage: demo.py [-h] --imagepath IMAGEPATH --backbonename BACKBONENAME
               --datasetname DATASETNAME --checkpointspath CHECKPOINTSPATH
               [--nmsthresh NMSTHRESH] [--confthresh CONFTHRESH]
optional arguments:
  -h, --help            show this help message and exit
  --imagepath IMAGEPATH
                        image you want to detect.
  --backbonename BACKBONENAME
                        backbone network for demo.
  --datasetname DATASETNAME
                        dataset used to train.
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to use.
  --nmsthresh NMSTHRESH
                        thresh used in nms.
  --confthresh CONFTHRESH
                        thresh used in showing bounding box.
cmd example:
CUDA_VISIBLE_DEVICES=0 python demo.py --checkpointspath faster_res50_trainbackup_coco/epoch_12.pth --datasetname coco --backbonename resnet50 --imagepath 000001.jpg
```


# Reference
```
[1]. https://github.com/jwyang/faster-rcnn.pytorch
```