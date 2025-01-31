# here we use the base image with cuda11.8
FROM gcr.io/deeplearning-platform-release/base-gpu.py310:m122
RUN if ! id 1000; then useradd -m -u 1000 clouduser; fi
RUN echo "Installing packages"
WORKDIR /home/jupyter
RUN echo "In this dockerfile, we don't install CUDNN"
COPY requirements.txt /home/jupyter/requirements.txt
RUN python3 -m pip --no-cache-dir install -r /home/jupyter/requirements.txt
ENV FORCE_CUDA="1"
RUN pip --no-cache-dir install \
einops==0.6.1 \
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
RUN pip uninstall -y pycocotools