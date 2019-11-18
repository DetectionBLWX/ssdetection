#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

cd cocoapi/PythonAPI
make
cd ..
cd ..

cd dcn
sh make.sh
cd ..

cd nms
sh make.sh
cd ..

cd roi_align
sh make.sh
cd ..

cd roi_crop
sh make.sh
cd ..

cd roi_pooling
sh make.sh
cd ..