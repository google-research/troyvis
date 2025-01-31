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
"""
Training Script.
"""


import math
from os import DirEntry
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES
from .data.datasets.objects365_v2 import categories as OBJ365_CATEGORIESV2
from .data.datasets.open_image import categories as OPENIMAGE_CATEGORIES
from .data.datasets.burst_video import BURST_CATEGORIES
from .data.datasets.vis import YTVIS_CATEGORIES_2019, OVIS_CATEGORIES, YTVIS_CATEGORIES_2021, LVVIS_CATEGORIES
from .data.datasets.uvo_video import UVO_CATEGORIES
from .data.datasets.bdd100k import BDD_DET_CATEGORIES,BDD_INST_CATEGORIES,BDD_TRACK_CATEGORIES
from .data.datasets.VisualGenome import VG_name_list
from .data.datasets.tao import TAO_CATEGORIES
from .data.datasets.odinw import odinw_category_dict
import torchvision.ops as ops
import random
from PIL import Image
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks
from detectron2.utils.logger import log_first_n

from .models.evap_model_dpl_ort import EVAP_Model_DPL_ORT
from .models.matcher import HungarianMatcher
from .models.criterion import SetCriterion

from detectron2.modeling.postprocessing import sem_seg_postprocess
from .utils import box_ops
from scipy.optimize import linear_sum_assignment
import os
import copy
import time

__all__ = ["EVAP_DPL_ORT"]



@META_ARCH_REGISTRY.register()
class EVAP_DPL_ORT(nn.Module):
    """
    Implement EVAP_DPL_ORT (for deployment only)
    """

    def __init__(self, cfg):
        super().__init__()
        # kernel reuse
        self.reuse_kernel = cfg.MODEL.REUSE_KERNEL
        self.reuse_kernel_clip_length = cfg.MODEL.REUSE_KERNEL_CLIP_LENGTH

        self.pseudo_video = cfg.MODEL.PSEUDO_VIDEO

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.LVIS_class_names = [cat['name'] for cat in LVIS_CATEGORIES]
        self.OBJ365_class_names = [cat['name'] for cat in OBJ365_CATEGORIESV2]
        self.OPENIMAGE_class_names = [cat['name'] for cat in OPENIMAGE_CATEGORIES]
        self.VG_name_list = VG_name_list

        self.category_set = {
            'obj365': set(list(range(365))),
            'openimage': set(list(range(601))),
            'lvis': set(list(range(1203))),
            'obj365_clip': set(list(range(365))),
            'openimage_clip': set(list(range(601))),
            'lvis_clip': set(list(range(1203))),
        }
        # self.OBJ365_set = set(list(range(365)))
        # self.OPENIMAGE_set = set(list(range(601)))
        self.brust_class_names= [cat['name'] for cat in BURST_CATEGORIES] 
        uvo_calss_name = [cat['name']+', object'  for cat in UVO_CATEGORIES[:-1] ]  + [UVO_CATEGORIES[-1]['name']]
        # print(uvo_calss_name)
        coco_class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        self.dataset_name_dicts = {
            'coco': coco_class_name,
            'coco_clip': coco_class_name,
            'lvis': self.LVIS_class_names,
            'obj365': self.OBJ365_class_names,
            'openimage': self.OPENIMAGE_class_names,
            'lvis_clip': self.LVIS_class_names,
            'obj365_clip': self.OBJ365_class_names,
            'openimage_clip': self.OPENIMAGE_class_names, 
            'bdd_det': [cat['name']  for cat in BDD_DET_CATEGORIES],
            'bdd_inst': [cat['name']  for cat in BDD_INST_CATEGORIES],
            'ytvis19':[cat['name'] for cat in YTVIS_CATEGORIES_2019],
            'image_yt19': [cat['name'] for cat in YTVIS_CATEGORIES_2019],
            'ytvis21': [cat['name'] for cat in YTVIS_CATEGORIES_2021],
            'image_yt21': [cat['name'] for cat in YTVIS_CATEGORIES_2021],
            'ovis': [cat['name'] for cat in OVIS_CATEGORIES],
            'image_o': [cat['name'] for cat in OVIS_CATEGORIES],
            'lvvis':  [cat['name'] for cat in LVVIS_CATEGORIES],
            'image_lv': [cat['name'] for cat in LVVIS_CATEGORIES],
            'uvo_video':uvo_calss_name,
            'burst':self.brust_class_names,
            'image_bur': self.brust_class_names,
            'image_tao': [cat['name'] for cat in TAO_CATEGORIES],
            'tao_video': [cat['name'] for cat in TAO_CATEGORIES],
            'sa1b': ['object'],
            'sa1b_clip': ['object'],
            'grounding': ['object'],
            'rvos': ['object'],
            'bdd_track_box':  [cat['name'] for cat in BDD_TRACK_CATEGORIES],
            'bdd_track_seg':  [cat['name'] for cat in BDD_TRACK_CATEGORIES],
            'ytbvos': ['object'],
        }
        self.num_class = {
            'lvis': len(self.LVIS_class_names),
            'obj365': 365,
            'openimage': 601,
            'coco': 80,
            'bdd_det': 10,
            'bdd_inst': 8,
            'ytvis19': 40,
            'image_yt19': 40,
            'ovis': 25,
            'image_o': 25,
            'ytvis21': 40,
            'image_yt21': 40,
            'lvvis': len(LVVIS_CATEGORIES),
            'image_lv': len(LVVIS_CATEGORIES),
            'uvo_video': 81,
            'burst':len(self.brust_class_names),
            'image_bur': len(self.brust_class_names),
            'image_tao': len(TAO_CATEGORIES),
            'tao_video': len(TAO_CATEGORIES),
        }
        for k, v in odinw_category_dict.items():
            if k == 'odinw13_Rabbits':
                self.dataset_name_dicts.update({k: ['rabbits' for cat in v]})
            elif k == 'odinw13_EgoHands':
                self.dataset_name_dicts.update({k: ['hand hands' for cat in v]})
            elif k == 'odinw13_Mushrooms':
                self.dataset_name_dicts.update({k: ['mushroom ' + cat['name'] for cat in v]})
            elif k=='odinw13_Packages':
                self.dataset_name_dicts.update({k: ['packages' for cat in v]})
            else:
                self.dataset_name_dicts.update({k: [cat['name'] for cat in v]})
            self.num_class.update({k:len(v)})

        self.video_info = {'bz':cfg.DATALOADER.DATASET_BS[-1], 'len': cfg.INPUT.SAMPLING_FRAME_NUM}

        self.glee = EVAP_Model_DPL_ORT(cfg, self.device, self.video_info, cfg.MODEL.CONTRAS_MEAN).eval() # to eval mode
      
        self.cate_sampled = False
        if cfg.MODEL.MAX_CATEGORY_LEN is not None:
            self.cate_sampled = True
            self.max_category_len = cfg.MODEL.MAX_CATEGORY_LEN 

        size_divisibility = cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.glee.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.test_topk_per_image = 100
        self.num_queries = cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES
        self.instance_on = True
        self.visaul_prompt = cfg.MODEL.VISUAL_PROMPT

        self.is_lsj = cfg.INPUT.DATASET_MAPPER_NAME == 'coco_instance_lsj'

        # for VIS  
        self.is_multi_cls = True
        self.apply_cls_thres  = 0.01
        self.save_path_prefix = os.path.join(cfg.OUTPUT_DIR, "Annotations")

        #for debug ensure all loss exist for each batch
        self.all_loss_nams = ['dist_loss', 'loss_bbox', 'loss_bbox_0', 'loss_bbox_1', 'loss_bbox_2', 'loss_bbox_3', 'loss_bbox_4', 'loss_bbox_5', 'loss_bbox_6', 'loss_bbox_7', 'loss_bbox_8', 'loss_bbox_dn', 'loss_bbox_dn_0', 'loss_bbox_dn_1', 'loss_bbox_dn_2', 'loss_bbox_dn_3', 'loss_bbox_dn_4', 'loss_bbox_dn_5', 'loss_bbox_dn_6', 'loss_bbox_dn_7', 'loss_bbox_dn_8', 'loss_bbox_interm', 'loss_ce', 'loss_ce_0', 'loss_ce_1', 'loss_ce_2', 'loss_ce_3', 'loss_ce_4', 'loss_ce_5', 'loss_ce_6', 'loss_ce_7', 'loss_ce_8', 'loss_ce_dn', 'loss_ce_dn_0', 'loss_ce_dn_1', 'loss_ce_dn_2', 'loss_ce_dn_3', 'loss_ce_dn_4', 'loss_ce_dn_5', 'loss_ce_dn_6', 'loss_ce_dn_7', 'loss_ce_dn_8', 'loss_ce_interm', 'loss_conf', 'loss_conf_0', 'loss_conf_1', 'loss_conf_2', 'loss_conf_3', 'loss_conf_4', 'loss_conf_5', 'loss_conf_6', 'loss_conf_7', 'loss_conf_8', 'loss_conf_interm', 'loss_dice', 'loss_dice_0', 'loss_dice_1', 'loss_dice_2', 'loss_dice_3', 'loss_dice_4', 'loss_dice_5', 'loss_dice_6', 'loss_dice_7', 'loss_dice_8', 'loss_dice_interm', 'loss_giou', 'loss_giou_0', 'loss_giou_1', 'loss_giou_2', 'loss_giou_3', 'loss_giou_4', 'loss_giou_5', 'loss_giou_6', 'loss_giou_7', 'loss_giou_8', 'loss_giou_dn', 'loss_giou_dn_0', 'loss_giou_dn_1', 'loss_giou_dn_2', 'loss_giou_dn_3', 'loss_giou_dn_4', 'loss_giou_dn_5', 'loss_giou_dn_6', 'loss_giou_dn_7', 'loss_giou_dn_8', 'loss_giou_interm', 'loss_mask', 'loss_mask_0', 'loss_mask_1', 'loss_mask_2', 'loss_mask_3', 'loss_mask_4', 'loss_mask_5', 'loss_mask_6', 'loss_mask_7', 'loss_mask_8', 'loss_mask_interm', 'track_loss']
        self.video_task_list = ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'lvvis', 'rvos','ytbvos', 'uvo_video','burst', 'coco_clip','obj365_clip' ,'sa1b_clip','lvis_clip','openimage_clip','bdd_track_box','bdd_track_seg']


    def forward(self, batched_inputs):
        
        task, prompt_list = self.get_task_name(batched_inputs)
        
        batch_name_list = None

        if self.training:
            raise ValueError("EVAP_DPL_ONNX is only used for inference rather than training")
        else:  # evaluation
            if task in ['vis', 'ovis', 'ytvis19','ytvis21', 'lvvis', 'uvo_video','burst', 'tao_video']:
                # return self.IDOL_inference(batched_inputs, task)
                return self.MinVIS_inference(batched_inputs, task)
            elif task in ['rvos']:
                self.inference_rvos(batched_inputs, prompt_list, task)
                return 
            elif task in ['ytbvos']:
                self.inference_ytbvos(batched_inputs, prompt_list, task)
            elif task in ['omnilabel']:
                return self.omnilabel_inference(batched_inputs, task)
            else:
                # img = batched_inputs[0]['image']
                # zero_pad = torch.zeros(3,1536,1536).to(img)
                # _,H,W = img.shape
                # zero_pad[:,:H,:W] = img
                # crop_size = (H,W)
                # batched_inputs[0]['image'] = zero_pad

                images = self.preprocess_image(batched_inputs, task)
                batch_name_list = self.dataset_name_dicts[task]

                (outputs,_),_,_ = self.glee(images, prompt_list, task, batch_name_list=batch_name_list, is_train=False)

                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                mask_box_results = outputs["pred_boxes"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                del outputs
                processed_results = []
                for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})
                    new_size = mask_pred_result.shape[-2:]

                    if self.is_lsj:
                        resize_ratio = image_size[0]/max(height, width)
                        crop_size =  (int(height*resize_ratio), int(width*resize_ratio))
                    else:
                        crop_size = image_size
                    # mask_pred_result = sem_seg_postprocess(
                    #     mask_pred_result, crop_size, height, width
                    # )
                    mask_pred_result = mask_pred_result[None,][:,:,:crop_size[0],:crop_size[1]]
                    mask_pred_result = F.interpolate( mask_pred_result,   size=(height,width),  mode="bilinear",  align_corners=False,      )[0]
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                    # instance segmentation inference
                    if self.instance_on:
                        mask_box_result = mask_box_result.to(mask_pred_result)
                        # height = new_size[0]/crop_size[0]*height
                        # width = new_size[1]/crop_size[1]*width
                        if self.is_lsj:
                            mask_box_result = self.LSJ_box_postprocess(mask_box_result, new_size, crop_size, height, width)
                        else:
                            height = new_size[0]/crop_size[0]*height
                            width = new_size[1]/crop_size[1]*width
                            mask_box_result = self.box_postprocess(mask_box_result, height, width)
                        instance_r = self.instance_inference(mask_cls_result, mask_pred_result, mask_box_result, task)
                        processed_results[-1]["instances"] = instance_r
                return processed_results

    def get_task_name(self, batched_inputs):
        prompt_list = {}
        if 'dataset_name' in batched_inputs[0]:
            # print([x['dataset_name'] for x in batched_inputs])
            if 'rvos' in  batched_inputs[0]['dataset_name']:
                task = 'rvos'
                prompt_list["grounding"] = []
                for x in batched_inputs:
                    prompt_list["grounding"] += x["expressions"]
            elif 'obj365' in batched_inputs[0]['dataset_name']:
                task = 'obj365'
                if self.pseudo_video and self.training:
                    task = 'obj365_clip'
            else:
                task = batched_inputs[0]['dataset_name'] # [ovis, ytvis19, ytvis21, uvo_video, bdd_det, bdd_inst]
                if task == 'UVO_image':
                    task = 'sa1b'
                    if self.pseudo_video and self.training:
                        task = 'sa1b_clip'

        elif "expressions" in batched_inputs[0]:
            task = 'grounding'
            prompt_list["grounding"] = [x["expressions"] for x in batched_inputs]
        elif 'task' in batched_inputs[0]:
            if batched_inputs[0]['task'] == 'sa1b':
                task = 'sa1b'
                if self.pseudo_video and self.training:
                    task = 'sa1b_clip'
            else:
                task = batched_inputs[0]['task']
        elif 'not_exhaustive_category_ids' in batched_inputs[0]:
            task = 'lvis'
        else:
            task = 'undefined'
         
        if task == 'coco' and self.pseudo_video and self.training:
            task = 'coco_clip'

        if 'pseudo_video' in batched_inputs[0] and  batched_inputs[0]['pseudo_video']  == True  and self.pseudo_video and self.training:
            if task in ['lvis', 'openimage']:
                task = task+'_clip'

        return task, prompt_list
         
    def preprocess_video(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))
        images = ImageList.from_tensors(images,size_divisibility=self.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs, task):
        """
        Normalize, pad and batch the input images.
        """
        if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21','lvvis', 'uvo_video', 'burst', 'rvos','ytbvos',\
            'coco_clip','obj365_clip','sa1b_clip','lvis_clip','openimage_clip','bdd_track_box','bdd_track_seg']:
            return self.preprocess_video(batched_inputs)  #[bz [frame]]
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images,size_divisibility=self.size_divisibility)
        return images

    def instance_inference(self, mask_cls, mask_pred, mask_box_result, task):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        if task == 'grounding':
            max_inst = 1
            prob = mask_cls.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(-1), max_inst, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, mask_cls.shape[1], rounding_mode='floor')
            labels = topk_indexes % mask_cls.shape[1]
            scores_per_image = scores
            labels_per_image = labels
            mask_pred = mask_pred[topk_boxes]
            mask_box_result = mask_box_result[topk_boxes]
        elif task == 'sa1b':
            max_inst = 100
            prob = mask_cls.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(-1), max_inst, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, mask_cls.shape[1], rounding_mode='floor')
            labels = topk_indexes % mask_cls.shape[1]
            scores_per_image = scores
            labels_per_image = labels
            mask_pred = mask_pred[topk_boxes]
            mask_box_result = mask_box_result[topk_boxes]
        
        else: 
            # [Q, K]
            if task in ['lvis', 'image_tao', 'image_bur']:
                self.test_topk_per_image = 300

            scores = mask_cls.sigmoid()  # [100, 80]
            labels = torch.arange(self.num_class[task], device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.num_class[task]
            mask_pred = mask_pred[topk_indices]
            mask_box_result = mask_box_result[topk_indices]

        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        result.pred_boxes = Boxes(mask_box_result)
        

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    
    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes
    
    def LSJ_box_postprocess(self, out_bbox,  lsj_size, crop_size, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        lsj_sclae = torch.tensor([lsj_size[1], lsj_size[0], lsj_size[1], lsj_size[0]]).to(out_bbox)
        crop_scale = torch.tensor([crop_size[1], crop_size[0], crop_size[1], crop_size[0]]).to(out_bbox)
        boxes = boxes * lsj_sclae
        boxes = boxes / crop_scale
        boxes = torch.clamp(boxes,0,1)

        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices
    
    def MinVIS_inference(self, batched_inputs, task):
        video_len = len(batched_inputs[0]['file_names'])
        clip_length = 3 # self.batch_infer_len
        batch_name_list = self.dataset_name_dicts[task]
        # torch.cuda.synchronize()
        # t0 = time.time()
        if self.reuse_kernel:
            pred_logits, pred_boxes, pred_embds, pred_masks = [], [], [], []
            for idx in range(video_len):
                start_idx = idx
                end_idx = idx+1
                clip_inputs = [{'image':batched_inputs[0]['image'][start_idx:end_idx]}]
                clip_images = self.preprocess_video(clip_inputs)
                # torch.cuda.synchronize()
                # t0 = time.time()
                if idx % self.reuse_kernel_clip_length == 0: # run decoder only on key frames
                    clip_output = self.glee(clip_images, task, mask_kernels=None)
                    mask_kernels = clip_output["pred_kernels"]
                    pred_logits.append(clip_output['pred_logits'].squeeze(0))
                    pred_boxes.append(clip_output['pred_boxes'].squeeze(0))
                    pred_embds.append(clip_output['pred_track_embed'].squeeze(0))
                else:
                    clip_output = self.glee(clip_images, task, mask_kernels=mask_kernels)
                    pred_logits.append(pred_logits[-1])
                    pred_boxes.append(pred_boxes[-1])
                    pred_embds.append(pred_embds[-1])
                # torch.cuda.synchronize()
                # t1 = time.time()
                # print(1000*(t1-t0))
                pred_masks.append(clip_output['pred_masks'].squeeze(0)) #.to(self.merge_device)
        else:
            #split long video into clips to form a batch input 
            # if video_len > clip_length:
            num_clips = math.ceil(video_len/clip_length)
            logits_list, boxes_list, embed_list, points_list, masks_list = [], [], [], [], []
            for c in range(num_clips):
                start_idx = c*clip_length
                end_idx = (c+1)*clip_length
                clip_inputs = [{'image':batched_inputs[0]['image'][start_idx:end_idx]}]
                clip_images = self.preprocess_video(clip_inputs)
                # import ipdb;ipdb.set_trace()
                (clip_output,_),dist,loss = self.glee(clip_images, {}, task,  batch_name_list = batch_name_list, is_train= False)
                logits_list.append(clip_output['pred_logits'])
                boxes_list.append(clip_output['pred_boxes'])
                embed_list.append(clip_output['pred_track_embed'])
                masks_list.append(clip_output['pred_masks']) #.to(self.merge_device)
            outputs = {
                'pred_logits':torch.cat(logits_list,dim=0).detach(),
                'pred_track_embed':torch.cat(embed_list,dim=0).detach(),
                'pred_masks':torch.cat(masks_list,dim=0).detach(),
                'pred_boxes': torch.cat(boxes_list,dim=0).detach(),
            }    

            # torch.cuda.synchronize()
            # t1 = time.time()
            # batch_name_list  = self.dataset_name_dicts[task]
            # import pdb;pdb.set_trace()

            pred_logits = list(torch.unbind(outputs['pred_logits']))
            pred_masks = list(torch.unbind(outputs['pred_masks']))
            pred_embds = list(torch.unbind(outputs['pred_track_embed']))
            pred_boxes = list(torch.unbind(outputs['pred_boxes']))
            del outputs
        # import pdb;pdb.set_trace()
        out_logits = []
        out_masks = []
        out_embds = []
        out_boxes = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds.append(pred_embds[0])
        out_boxes.append(pred_boxes[0])
        memory_embedding = out_embds[-1]

        for i in range(1, len(pred_logits)):
            # indices = self.match_from_embds(memory_embedding, pred_embds[i])
            MA_embedding = torch.stack(out_embds[-3:]).mean(0)
            indices = self.match_from_embds(MA_embedding, pred_embds[i])
            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])
            out_boxes.append(pred_boxes[i][indices, :])
            score_weights = pred_logits[i][indices, :].sigmoid().max(-1)[0][:,None]
            memory_embedding = (memory_embedding+pred_embds[i][indices, :]*score_weights )/(1+score_weights)

        # for i in range(1, len(pred_logits)):
        #     # indices = self.match_from_embds(out_embds[-1], pred_embds[i])
        #     MA_embedding = torch.stack(out_embds[-3:]).mean(0)
        #     indices = self.match_from_embds(MA_embedding, pred_embds[i])
        #     out_logits.append(pred_logits[i][indices, :])
        #     out_masks.append(pred_masks[i][indices, :, :])
        #     out_embds.append(pred_embds[i][indices, :])

        mask_cls_result = sum(out_logits)/len(out_logits)

        out_logits = torch.stack(out_logits, dim=1)  # q numc -> q t numc

        mask_pred_result = torch.stack(out_masks, dim=1) # q h w -> q t h w
        mask_box_result = torch.stack(out_boxes, dim=1) # q 4 -> q t 4
        first_resize_size = (clip_images.tensor.shape[-2], clip_images.tensor.shape[-1])

        input_per_image = batched_inputs[0]
        image_size = clip_images.image_sizes[0]  # image size without padding after data augmentation

        height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
        width = input_per_image.get("width", image_size[1])
        mask_box_result = self.box_postprocess(mask_box_result, height, width)
        # torch.cuda.synchronize()
        # t2 = time.time()
        output = self.minvis_inference_video(mask_cls_result, mask_pred_result, mask_box_result, image_size, height, width, first_resize_size, task, out_logits, batched_inputs)
        # torch.cuda.synchronize()
        # t3 = time.time()
        # print("forward:%.4f, associate:%.4f, post-process:%.4f"%(t1-t0, t2-t1, t3-t2))
        return output

    def minvis_inference_video(self, mask_cls, mask_pred, mask_box_result, img_size, output_height, output_width, first_resize_size, task, ori_logits, batched_inputs):
        if task != 'tao_video':
            if len(mask_cls) > 0:
                # import pdb;pdb.set_trace()            
                # keep top-20 predictions
                # torch.cuda.synchronize()
                # t0 = time.time()
                topk = 30
                scores = mask_cls.sigmoid()  # [300, 40]

                num_class = self.num_class[task]
                labels = torch.arange(num_class, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
                scores_per_image, topk_indices = scores.flatten(0, 1).topk(topk, sorted=False)  # select 20
                labels_per_image = labels[topk_indices]
                topk_indices = topk_indices // num_class
                # import ipdb; ipdb.set_trace()
                # torch.cuda.synchronize()
                # t1 = time.time()
                # mask_pred = mask_pred[topk_indices.cpu()].cuda() # (topk, L, H/4, W/4)
                mask_pred = mask_pred[topk_indices] # (topk, L, H/4, W/4)
                mask_box_result = mask_box_result[topk_indices]
                if self.is_lsj:
                    resize_ratio = img_size[0]/max(output_height, output_width)
                    crop_size =  (int(output_height*resize_ratio), int(output_width*resize_ratio))
                else:
                    crop_size = img_size
                # resize_ratio = image_size[0]/max(height, width)
                #             crop_size =  (int(height*resize_ratio), int(width*resize_ratio))
                pred_masks = F.interpolate(
                    mask_pred, size=first_resize_size, mode="bilinear", align_corners=False
                )
                pred_masks = pred_masks[:, :, : crop_size[0], : crop_size[1]]
                pred_masks = F.interpolate(
                    pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
                )

                masks = pred_masks > 0. # (topk, L, output_height, output_width)
                # torch.cuda.synchronize()
                # t2 = time.time()

                out_scores = scores_per_image.tolist()
                out_labels = labels_per_image.tolist()
                # import ipdb; ipdb.set_trace()
                # vid_len = masks.shape[1]
                # out_masks = []
                # out_masks_tmp = rle_encode_gpu_batch(masks.flatten(0, 1)) # a long list
                # for k in range(topk):
                #     start_idx = k * vid_len
                #     end_idx = (k+1) * vid_len
                #     out_masks.append(out_masks_tmp[start_idx: end_idx])
                # out_masks = []
                # for mask in masks:
                #     # loop objects
                #     out_masks.append([])
                #     for m in mask:
                #         # loop frames
                #         out_masks[-1].append(rle_encode_gpu(m))
                out_masks = [m for m in masks]

                mask_box_result[:,:,2]=mask_box_result[:,:,2]-mask_box_result[:,:,0]
                mask_box_result[:,:,3]=mask_box_result[:,:,3]-mask_box_result[:,:,1]

                # xyxy2 xywh
                mask_box_result = mask_box_result.cpu().long()
                out_boxes = [m for m in mask_box_result]
                # torch.cuda.synchronize()
                # t3 = time.time()
                # print("topk: %.4f, resize: %.4f, to-rle: %.4f"%(t1-t0, t2-t1, t3-t2))
            else:
                out_scores = []
                out_labels = []
                out_masks = []
                out_boxes = []

            video_output = {
                "image_size": (output_height, output_width),
                "pred_scores": out_scores,
                "pred_labels": out_labels,
                "pred_masks": out_masks,
                "pred_boxes":out_boxes,
            }
        # else:  # for TAO video  teta metric
        #     scores = mask_cls.sigmoid()  # [300, numcls]

        #     topk_num = 50

        #     num_class = self.num_class[task]

            
        #     scores_per_video, topk_indices = scores.max(-1)[0].topk(topk_num, sorted=False)  # select 20
        #     labels_per_video =  scores[topk_indices].max(-1)[1]  # [select_num]

        #     # import pdb;pdb.set_trace()

        #     mask_pred = mask_pred[topk_indices.cpu()]  #[select, len, H, W]
        #     mask_pred = mask_pred>0

        #     mask_box_result = mask_box_result[topk_indices]  #[slelct_num, len, 4]
        #      # xyxy2 xywh
        #     mask_box_result[:,:,2]=mask_box_result[:,:,2]-mask_box_result[:,:,0]
        #     mask_box_result[:,:,3]=mask_box_result[:,:,3]-mask_box_result[:,:,1]

        #     ori_logits = ori_logits[topk_indices].sigmoid()    #[slelct_num, len, num_class]
            
        #     image_ids = batched_inputs[0]['image_ids']
        #     video_id = batched_inputs[0]['video_id']
        #     video_len = len(image_ids)
        #     track_ids = torch.arange(topk_num).to(scores_per_video) + topk_num*video_id


        #     video_results = []
        #     for i,image_id in enumerate(image_ids):
                
        #         # frame_logits = ori_logits[:,i] # [topk_num,nun_cls]
        #         # scores_per_frame, labels_per_frames = frame_logits.max(-1)

        #         frame_boxes = mask_box_result[:,i]  
        #         frame_masks = mask_pred[:,i]  
        #         mask_valid = frame_masks.flatten(1,2).sum(-1)>5

        #         frame_boxes = frame_boxes[mask_valid]
        #         frame_scores = scores_per_video[mask_valid]
        #         frame_labels = labels_per_video[mask_valid]
        #         frame_trackids = track_ids[mask_valid]

        #         # box nms
        #         boxes_before_nms = box_ops.box_cxcywh_to_xyxy(frame_boxes)
        #         keep_indices = ops.nms(boxes_before_nms,frame_scores,0.5)#.tolist()

        #         frame_boxes = frame_boxes[keep_indices]
        #         frame_scores = frame_scores[keep_indices]
        #         frame_labels = frame_labels[keep_indices]
        #         frame_trackids = frame_trackids[keep_indices]

                
        #         for box,score,label,trackid in zip(frame_boxes,frame_scores,frame_labels,frame_trackids):
        #             video_results.append(
        #                 {
        #                     "image_id" : image_id,
        #                     "category_id" : label.item(),
        #                     "bbox" : box.tolist(),
        #                     "score" : score.item(),
        #                     "track_id": trackid.item(),
        #                     "video_id": video_id
        #                 }
        #             )


        #     video_output = video_results



        else:  # for TAO video  trackmAP
            scores = mask_cls.sigmoid()  # [300, numcls]

            topk_num = 50

            num_class = self.num_class[task]
            labels = torch.arange(num_class, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            scores_per_video, topk_indices = scores.flatten(0, 1).topk(topk_num, sorted=False)  # select 20
            labels_per_video = labels[topk_indices]  # [select_num]
            topk_indices = topk_indices // num_class


            # import pdb;pdb.set_trace()

            mask_pred = mask_pred[topk_indices.cpu()]  #[select, len, H, W]
            mask_pred = mask_pred>0

            mask_box_result = mask_box_result[topk_indices]  #[slelct_num, len, 4]
             # xyxy2 xywh
            mask_box_result[:,:,2]=mask_box_result[:,:,2]-mask_box_result[:,:,0]
            mask_box_result[:,:,3]=mask_box_result[:,:,3]-mask_box_result[:,:,1]

            ori_logits = ori_logits[topk_indices].sigmoid()    #[slelct_num, len, num_class]
            
            image_ids = batched_inputs[0]['image_ids']
            video_id = batched_inputs[0]['video_id']
            video_len = len(image_ids)

            video_results = []
            for i,image_id in enumerate(image_ids):
                
                frame_logits = ori_logits[:,i] # [topk_num,nun_cls]
                frame_boxes = mask_box_result[:,i]  

                scores_per_frame, labels_per_frames = frame_logits.max(-1)
                
                for idx in range(topk_num):
                    if mask_pred[idx,i].sum()<5:
                        continue
                    video_results.append(
                        {
                            "image_id" : image_id,
                            "category_id" : labels_per_video[idx].item(),
                            "bbox" : frame_boxes[idx].tolist(),
                            "score" : scores_per_video[idx].item(),
                            "track_id": topk_num*video_id + idx,
                            "video_id": video_id
                        }
                    )

            video_output = video_results

        return video_output
