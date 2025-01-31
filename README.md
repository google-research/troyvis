
# EVA-Perceiver: Efficient Open-Vocabulary Foundation Models for Object Perception

*This is not an officially supported Google product.*

## Highlight:

- EVA-Perceiver is the first **efficient foundation model** family for **open-vocabulary** object perception. It can detect and segment objects of any class in images and track objects of any class in videos.
- EVA-Perceiver can do open-vocabulary video instance segmentation of more than 1K object categories in **real-time** on A100 GPUs. 
- EVA-Perceiver is trained on huge amount of images and videos from different domains, showing strong **zero-shot** perception ability.


## Getting started

1. **Installation**: Please refer to [INSTALL.md](assets/INSTALL.md) for more details.
2. **Data preparation**: Please refer to [DATA.md](assets/DATA.md) for more details.
3. **Training**: Please refer to [TRAIN.md](assets/TRAIN.md) for more details.
4. **Testing**: Please refer to [TEST.md](assets/TEST.md) for more details. 
5. **Model zoo**: Please refer to [MODEL_ZOO.md](assets/MODEL_ZOO.md) for more details.

## Acknowledgments

- Thanks [GLEE](https://github.com/FoundationVision/GLEE) for providing strong object-level foundation model as our baseline.

  
## Third Party
- **ops** from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
- **pycocotools** from [cocoapi](https://github.com/wjf5203/cocoapi)
- **d2** from [detectron2](https://github.com/facebookresearch/detectron2)