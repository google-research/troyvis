# Install
## Requirements
We test the codes in the following environments, other versions may also be compatible but Pytorch vision should be >= 1.7

- CUDA 11.8
- Python 3.10.14
- Pytorch 2.0.0
- Torchvison 0.15.1

## Install environment for EVA-Perceiver

```
# install dependency of Detectron2
pip3 install -r requirements.txt

# install other packages
pip3 install einops==0.6.1 \
torch==2.0.0 \
torchvision \
shapely \
lvis \
scipy \
fairscale \
einops \
xformers \
tensorboard \
opencv-python-headless \
timm \
ftfy \
transformers==4.36.0 \
ipdb \
apex \
onnx \
onnxruntime \
onnxruntime-gpu \
setuptools==69.5.1

# deal with pycocotools (building locally)
pip uninstall -y pycocotools
cd third_party/cocoapi/PythonAPI
pip3 install -e .
cd ../../..

# install detectron2 (building locally)
cd third_party/d2
pip3 install -e .
cd ../..


# compile deformable attention (building locally)
cd third_party/ops/
python3 setup.py build install --user
cd ../../../../../..


# Download pretrained Language Model
# CLIP-Base
wget -P projects/EVAP/clip_vit_base_patch32/  https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE/clip_vit_base_patch32/pytorch_model.bin
# EVA-02 CLIP-L
mkdir -p weights
wget -c https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt -O weights/EVA02_CLIP_L_336_psz14_s6B.pt


```
