#!/usr/bin/env python3

# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import os
import sys
import itertools
import time
from typing import Any, Dict, List, Set
import yaml
import torch
import warnings
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators, LVISEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
ROOT = os.path.dirname(os.path.abspath(__file__))  # current project root
sys.path.insert(0, ROOT)
from glee import add_glee_config, build_detection_train_loader, build_detection_test_loader
from glee.backbone.eva01 import get_vit_lr_decay_rate
from glee.data import (
    get_detection_dataset_dicts, RefCOCODatasetMapper, YTVISDatasetMapper,  OMNILABEL_Evaluator, Joint_Image_LSJDatasetMapper,Joint_Image_Video_LSJDatasetMapper,  COCO_CLIP_DatasetMapper,UnivideoimageDatasetMapper , UnivideopseudoDatasetMapper
)
from glee.data import build_custom_train_loader,YTVISEvaluator

Trainer = DefaultTrainer

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_glee_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()  
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # download latest ckpt automatically
    if comm.get_local_rank() == 0:
        dir_name = cfg.OUTPUT_DIR.split("/")[-1]
        server_folder = f'gs://xcloud-shared/masterbin/exp/{dir_name}'
        local_folder = f'/home/jupyter/code/eva-perceiver/exp/{dir_name}'
        last_ckpt_server = server_folder + "/last_checkpoint"
        last_ckpt_local = local_folder + "/last_checkpoint"
        os.system(f"gsutil -m cp {last_ckpt_server} {last_ckpt_local}")
        if os.path.exists(last_ckpt_local):
            with open(last_ckpt_local, "r") as f:
                ckpt_name = f.readlines()[0].strip()
            ckpt_path_server = server_folder + f"/{ckpt_name}"
            ckpt_path_local = local_folder + f"/{ckpt_name}"
            os.system(f"gsutil -m cp {ckpt_path_server} {ckpt_path_local}")
    comm.synchronize()
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        # deployment (exporting to ONNX)
        image_size = (480, 864) # (864, 480)
        output = "export/onnx/glee-r50-backbone.onnx"
        opset = 17

        dummy_input = {"x": torch.randn((1, 3, image_size[0], image_size[1]), dtype=torch.float).cuda()}
        dynamic_axes = {
            "x": {0: "batch_size", 2: "image_height", 3: "image_width"},
        }
        onnx_model = model.glee.backbone
        _ = onnx_model(**dummy_input)

        output_names = ["outputs"]

        if not os.path.exists(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            print(f"Exporting onnx model to {output}...")
            with open(output, "wb") as f:
                torch.onnx.export(
                    onnx_model,
                    tuple(dummy_input.values()),
                    f,
                    export_params=True,
                    verbose=False,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=list(dummy_input.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
