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

import math
import time
import copy
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import CLIPTokenizer,CLIPTextModel
from detectron2.modeling import build_backbone
from .pixel_decoder.maskdino_encoder import build_pixel_decoder
from .transformer_decoder.maskdino_decoder import build_transformer_decoder
from .eva_clip import create_model, get_tokenizer



def agg_lang_feat(features, mask, pool_type="average"):
    """average pooling of language features"""
    # feat: (bs, seq_len, C)
    # mask: (bs, seq_len)
    if pool_type == "average":
        embedded = features * mask.unsqueeze(-1).float() # use mask to zero out invalid token features
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())
    elif pool_type == "max":
        out = []
        for i in range(len(features)):
            pool_feat, _ = torch.max(features[i][mask[i]], 0) # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0) # (bs, C)
    else:
        raise ValueError("pool_type should be average or max")
    return aggregate


class EVAP_Model_DPL_ONNX(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    For deployment. Inference only. Simplify the code
    """
    def __init__(self, cfg, device, video_info, contras_mean):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
         
        self.text_encode_type = cfg.MODEL.TEXT.ARCH
        self.early_fusion = cfg.MODEL.EARLYFUSION
         
        if cfg.MODEL.TEXT.ARCH in ['clip_frozen', 'clip_unfrozen', 'clip_teacher']:
            self.tokenizer = CLIPTokenizer.from_pretrained(cfg.MODEL.TEXT.DIR)
            self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.text_encoder = CLIPTextModel.from_pretrained(cfg.MODEL.TEXT.DIR)
            self.lang_encoder = None
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))
            if cfg.MODEL.TEXT.ARCH == 'clip_unfrozen':
                self.text_encode_type = 'clip_frozen'
        elif cfg.MODEL.TEXT.ARCH in ['EVA02_L_CLIP_frozen', 'EVA02_L_CLIP_unfreeze', 'EVA02_L_CLIP_teacher']:
            model_name = "EVA02-CLIP-L-14-336"
            pretrained = "weights/EVA02_CLIP_L_336_psz14_s6B.pt"
            self.text_encoder = create_model(model_name, pretrained, force_custom_clip=True)
            self.tokenizer = get_tokenizer(model_name)
            # self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.lang_encoder = None
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))
 
        self.pixel_decoder = build_pixel_decoder(cfg, self.backbone.output_shape())
        transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.predictor = build_transformer_decoder(cfg, transformer_predictor_in_channels, lang_encoder = self.lang_encoder, mask_classification=True,)
        self.to(device)
        
        self.video_info = video_info
        self.contras_mean = contras_mean

        self.track_loss_version = cfg.MODEL.TRACK_VERSION

        self.no_mask_tasks = ['obj365', 'obj365_clip','openimage', 'openimage_clip', 'vg', 'grit', 'bdd_det', 'bdd_track_box'] 

        self.use_offline_cls_embed = cfg.MODEL.USE_OFFLINE_CLS_EMBED
        if self.use_offline_cls_embed:
            if len(cfg.DATASETS.TEST) == 1 and cfg.DATASETS.TEST[0] in ["lvvis_val", "BURST_video_val"]:
                cls_embed = torch.load("cls_embed_lvvis.pth", map_location="cuda")
                self.extra_class_embeddings = cls_embed["extra"]["class_embeddings"] # (1196, 256)
                self.early_semantic_hidden = cls_embed["early_semantic"]["hidden"] # (1, 4983, 256)
                self.early_semantic_masks = cls_embed["early_semantic"]["masks"] # (1, 4983)

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, images, task, mask_kernels=None):
        assert self.use_offline_cls_embed  # use cached language embeddings
        # extra = {}
        # early_semantic = None
        # torch.cuda.synchronize()
        # t0 = time.time()
        # import ipdb; ipdb.set_trace()
        if self.text_encode_type in ["clip_frozen", "clip_teacher", 'EVA02_L_CLIP_frozen', 'EVA02_L_CLIP_unfreeze', 'EVA02_L_CLIP_teacher']: # True during inference
            # if task not in ['grounding','rvos']:
            # get extra from cache
            # import ipdb; ipdb.set_trace()
            # extra = copy.deepcopy(self.cls_embed["extra"])
            # extra = {"class_embeddings": copy.deepcopy(self.cls_embed["extra"]["class_embeddings"])}
            extra_class_embeddings = self.extra_class_embeddings.clone()
            if self.early_fusion:  # early_fusion
                # get early_semantic from cache
                # early_semantic = copy.deepcopy(self.cls_embed["early_semantic"])
                early_semantic_hidden = self.early_semantic_hidden.clone()
                early_semantic_masks = self.early_semantic_masks.clone()
                # import ipdb; ipdb.set_trace()
                # saved_obj = {"extra": extra, "early_semantic": early_semantic}
                # torch.save(saved_obj, "cls_embed_lvvis.pth") 
        else:
            raise NotImplementedError
        # torch.cuda.synchronize()
        # t1 = time.time()

        if isinstance(images, torch.Tensor):
            features = self.backbone(images)
        else:
            features = self.backbone(images.tensor)
        # torch.cuda.synchronize()
        # t2 = time.time()

    
        # bz = len(images)//2
        # print([(k, v.shape) for (k, v) in features.items()])
        mask_features, _, multi_scale_features, _, _ = self.pixel_decoder.forward_features(features, masks=None, early_fusion={"hidden": early_semantic_hidden, "masks": early_semantic_masks})
        # print(mask_features.shape)
        # print([x.shape for x in multi_scale_features])
        # torch.cuda.synchronize()
        # t3 = time.time()
        
        targets = None

        outputs = self.predictor(multi_scale_features, mask_features, extra={"class_embeddings": extra_class_embeddings}, task=task, masks=None, targets=targets, mask_kernels=mask_kernels)
        # torch.cuda.synchronize()
        # t4 = time.time()
        # print("text:%.4f, backbone:%.4f, encoder:%.4f, decoder:%.4f"%(t1-t0, t2-t1, t3-t2, t4-t3))
        return outputs[0]
