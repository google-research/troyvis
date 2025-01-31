# YouTubeVIS data loading and evaluation
## UPDATE
FIXED bug using numpy.linspace in **coco_eval.py** & **ytvos_eval.py** \
FIXED error caused by 'unicode' type running on python3 in **coco.py** & **ytvos.py**

## Introduction

This package provides data loading and evaluation functionalities for video instance segmentation on [YouTubeVIS](https://youtube-vos.org/dataset/vis/). It is built based on [COCO API](https://github.com/cocodataset/cocoapi) designed for the MSCOCO dataset (http://cocodataset.org/). For evaluation metrics, please refer to the [descriptions](https://youtube-vos.org/dataset/vis/) for details.
We have only implemented Python API for YouTubeVIS. API in other languages are not available for now.

## Installation
To install:
```
cd PythonAPI
# To compile and install locally 
python setup.py build_ext --inplace
# To install library to Python site-packages 
python setup.py build_ext install
```

## Contact
If you have any questions regarding the repo, please create an issue.
