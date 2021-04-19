#!/bin/bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python setup.py install
cd ..
git clone https://github.com/timesler/facenet-pytorch.git
cd facenet-pytorch
python3 setup.py install

