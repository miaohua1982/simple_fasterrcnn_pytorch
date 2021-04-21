# simple_fasterrcnn_pytorch
This is a simplest implementation of fasterrcnn by pytorch when I learn the paper Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (https://arxiv.org/abs/1506.01497)
I give the key operation iou/nms/roi_pool in details by python and c++ version, not just call the torchvision library, so you are able to see the implementation of details. By the way, you can
compare the different implementation.I use The PASCAL Visual Object Classes(VOC2007) to train & test the model, the highest score is almost 0,695.

## Table of Contents

- [Background](#background)
- [Requirements](#requirements)
- [Install](#install)
- [Usage](#usage)
- [Scores](#scores)


## Background
This is a simplest implementation of fasterrcnn by pytorch when I learn the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497).
There are lots of implementation in github.But many of them is too complicated, or just calling the torchvision's module.
I implementation the key operation iou/nms/roi_pool in python & c++(not cuda), so you can see what the operation do on data.
What's more, the c++ version nms & roi_pool are written independently, and you can install them as a python package.

## Requirements
 ipdb>=0.13.5  
 matplotlib>=3.1.1  
 numpy>=1.17.5  
 Pillow>=8.0.1  
 scikit-image>=0.18.1  
 torch>=1.5.1  
 torchvision>=0.6.1  
 tqdm>=4.51.0  
 visdom>=0.1.8.9  
 pybind11>=2.6.2  

 The whole project is test on python3.6
## Install
To install nms & roi_pool you should have [pybind11](https://github.com/pybind/pybind11/tree/stable) [installed](https://pybind11.readthedocs.io/en/stable/installing.html)

What you have to do is going into the util folder and run the following code:

```sh
pip install ./nms_mh
```
and  
```sh
pip install ./roi_pool_mh
```

then you can use the library in other project not only in current one.  

The example for nms. And the nms_mh package also contains the iou & giou & ciou & diou functions
```sh
import numpy as np
import nms_mh as m

rois = np.random.rand(12000,4).astype(np.float32)
scores = np.random.rand(12000).astype(np.float32)

keep_list = m.nms(rois, scores, 0.7)
```