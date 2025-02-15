<h1 align="center" id="title">PATHFinder: A Foundation Model for Road Mapping in Support of United Nations Humanitarian Affairs</h1>


This repository contains the official Pytorch implementation of training & evaluation code and the pretrained model for PATHFinder (Paper to be released soon).


![Python 3.10.5](https://img.shields.io/badge/python-3.10.5-green.svg)


PATHFinder  is an efficient road semantic segmentation model, as shown in Figure 1, built on the simple yet powerful [SegFormer](https://arxiv.org/abs/2105.15203).

## Installation

For install and data preparation, please follow the steps mentioned below.


An example (works for me): ```CUDA 10.1``` and  ```pytorch 1.7.1``` 

```
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
cd SegFormer && pip install -e . --user
```
