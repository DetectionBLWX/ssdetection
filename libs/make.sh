#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

cd cocoapi/PythonAPI
make
cd ..
cd ..

python setup.py develop
rm -rf build
rm -rf FPN.egg-info