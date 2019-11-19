# FasterRCNN
```
Pytorch Implementation of FasterRCNN.
```


# Environment
```
OS: Ubuntu 16.04
Python: python3.x with torch==0.4.1, torchvision==0.2.2
```


# Performance
|  Backbone   | Train       |  Test         |  Pretrained Model  |  Epochs  |   AP      											|
|  :----:     | :----:      |  :----:       |  :----:    	     |	:----:  |   :----:  											|
| ResNet-50   | trainval35k |  minival5k    |  Caffe		     |	12 	    |	[34.9](PerformanceDetails/Res50_caffe_epoch12.MD)	|
| ResNet-50   | trainval35k |  minival5k    |  Pytorch		     |	12	    |	[34.3](PerformanceDetails/Res50_pytorch_epoch12.MD)	|
| ResNet-101  | trainval35k |  minival5k    |  Caffe		     |	12	    |	-													|
| ResNet-101  | trainval35k |  minival5k    |  Pytorch   	     |	12	    |	-  													|


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
```


# Reference
```
[1]. https://github.com/jwyang/faster-rcnn.pytorch
```