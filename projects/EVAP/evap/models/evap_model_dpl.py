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


class EVAP_Model_DPL(nn.Module):
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
            if len(cfg.DATASETS.TEST) == 1 and cfg.DATASETS.TEST[0] == "lvvis_val":
                self.cls_embed = torch.load("cls_embed_lvvis.pth", map_location="cuda")

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, images, prompts, task, targets=None, batch_name_list=None, is_train = True, mask_kernels=None):
        extra = {}
        dist_loss = torch.tensor(0.0, device="cuda")
        early_semantic = None
        # torch.cuda.synchronize()
        # t0 = time.time()
        if self.text_encode_type in ["clip_frozen", "clip_teacher"]: # True during inference
            if task not in ['grounding','rvos']:
                assert batch_name_list
                calsses_name_list = batch_name_list
                if (not is_train) and self.use_offline_cls_embed:
                    extra = copy.deepcopy(self.cls_embed["extra"])
                else:
                    tokenized = self.tokenizer.batch_encode_plus(calsses_name_list,
                            max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                            padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                            return_special_tokens_mask=True,
                            return_tensors='pt',
                            truncation=True).to("cuda")

                    texts = (tokenized['input_ids'], tokenized['attention_mask'])
                    token_x = self.text_encoder(*texts)['last_hidden_state']

                    valid_mask = tokenized['attention_mask'].bool()
                    token_x = token_x @ self.lang_projection
                    lang_feat_pool = agg_lang_feat(token_x, tokenized['attention_mask'], pool_type="average")  # (bs,  768)
                    extra['class_embeddings'] = lang_feat_pool 
                if self.early_fusion:  # early_fusion
                    if (not is_train) and self.use_offline_cls_embed:
                        early_semantic = copy.deepcopy(self.cls_embed["early_semantic"])
                    else:
                        gather_all_classtoken = token_x.flatten(0, 1)[tokenized['attention_mask'].flatten(0, 1) >0]
                        gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images), 1, 1)  #[bs,L,C]
                        gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0]) > 0  #[bs,L]
                        early_semantic = {"hidden": gather_all_classtoken.float(), "masks": gather_all_classtoken_mask}
                # import ipdb; ipdb.set_trace()
                # saved_obj = {"extra": extra, "early_semantic": early_semantic}
                # torch.save(saved_obj, "cls_embed_lvvis.pth") 
        elif self.text_encode_type in ['EVA02_L_CLIP_frozen', 'EVA02_L_CLIP_unfreeze', 'EVA02_L_CLIP_teacher','EVA01_G_CLIP_frozen'  , 'EVA01_G_CLIP_unfreeze'  ,'EVA01_G_CLIP_teacher']:
            if task not in ['grounding','rvos']:
                assert batch_name_list
                text = self.tokenizer(batch_name_list).to('cuda')   
                lang_feat_final = self.text_encoder(text)
                lang_feat_final = lang_feat_final @ self.lang_projection
                extra['class_embeddings'] = lang_feat_final
                if self.early_fusion: # class early_fusion
                    gather_all_classtoken = lang_feat_final.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                    gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                    early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask} 
                # import ipdb; ipdb.set_trace()
                # saved_obj = {"extra": extra, "early_semantic": early_semantic}
                # torch.save(saved_obj, "cls_embed_lvvis.pth") 
        # torch.cuda.synchronize()
        # t1 = time.time()


        if 'grounding' in prompts:
 
            if self.text_encode_type == 'clip_frozen' or self.text_encode_type == 'clip_teacher':

                tokens = self.tokenizer(
                    prompts['grounding'], padding='max_length', truncation=True, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, return_tensors='pt'
                    )
                tokens = {key: value.to(images.device) for key, value in tokens.items()}

                texts = (tokens['input_ids'], tokens['attention_mask'])
                x = self.text_encoder(*texts)
                token_x = x['last_hidden_state']
                token_x = token_x @ self.lang_projection

                extra['grounding_tokens'] = token_x.permute(1,0,2) #[len,bz,C]

                non_zero_query_mask = tokens['attention_mask']
                lang_feat_pool = agg_lang_feat(token_x, non_zero_query_mask, pool_type="average").unsqueeze(1) # (bs, 1, 768)

                dist_loss =  (lang_feat_pool*0).sum()
                
                extra['grounding_nonzero_mask'] = ~non_zero_query_mask.bool()  # [bz,len]
                extra['grounding_class'] = lang_feat_pool.squeeze(1) #[bz,C
                # gather_all_classtoken = token_x.flatten(0,1)[tokenized['attention_mask'].flatten(0,1)>0]
                # gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                # gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                # early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask} 
                early_semantic = {"hidden":token_x.float(),"masks":tokens['attention_mask']>0} 
        

        if isinstance(images, torch.Tensor):
            features = self.backbone(images)
        else:
            features = self.backbone(images.tensor)
        # torch.cuda.synchronize()
        # t2 = time.time()

    
        # bz = len(images)//2
        # print([(k, v.shape) for (k, v) in features.items()])
        mask_features, _, multi_scale_features, zero_loss, _ = self.pixel_decoder.forward_features(features, masks=None, early_fusion = early_semantic)
        # print(mask_features.shape)
        # print([x.shape for x in multi_scale_features])
        # torch.cuda.synchronize()
        # t3 = time.time()
        

        if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'burst', 'rvos','coco_clip','obj365_clip','sa1b_clip',\
            'bdd_track_seg', 'bdd_track_box', 'uvof_clip','lvis_clip','openimage_clip'] and is_train:
            video_outputs = []
            outputs = self.predictor(multi_scale_features, mask_features, extra=extra, task=task, masks=None, targets=targets, mask_kernels=mask_kernels)
            track_loss = self.get_tracking_contrastive_lossv3(outputs[0], targets, task)
            return outputs, track_loss, dist_loss
        else:
            outputs = self.predictor(multi_scale_features, mask_features, extra=extra, task=task, masks=None, targets=targets, mask_kernels=mask_kernels)
            # torch.cuda.synchronize()
            # t4 = time.time()
            # print("text:%.4f, backbone:%.4f, encoder:%.4f, decoder:%.4f"%(t1-t0, t2-t1, t3-t2, t4-t3))
            fake_track_loss = torch.tensor(0.0, device="cuda")
            return  outputs, fake_track_loss, dist_loss
