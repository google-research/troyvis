# Tutorial for Training

EVA-Perceiver has three training stages: (1) Objects365 & OpenImages pretraining (2) image-level joint training respectively. 

By default, we train EVA-Perceiver using 64 A100 GPUs with the batchsize of 128. Users interested in specific datasets or aiming to further improve performance by training on individual datasets can adjust the `DATASETS` config within the YAML configuration file.

We provide configurations for Stage 1 and 2 training with two types of backbones (ResNet50, and EfficientViT-L2)  across different variants, under the [projects/EVAP/configs](../projects/EVAP/configs) folder.  For employing larger or novel backbones, it is advisable to initialize the components beyond the backbone with the pretrained weights from EVAP-Lite-joint to expedite convergence.



## Pretrained Weights

```bash
# CLIP-Base
wget -P projects/EVAP/clip_vit_base_patch32/  https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE/clip_vit_base_patch32/pytorch_model.bin
# EVA-02 CLIP-L
mkdir -p weights
wget -c https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt -O weights/EVA02_CLIP_L_336_psz14_s6B.pt

# R50 (GLEE_Lite) warmup initialized weight
# The randomly initialized Transformer Decoder is difficult to converge when combined with the large vocabulary of Objects365 and OpenImages. 
# It is recommended to use the Transformer weights of MaskDINO (with region proposal capability) to initialize and accelerate convergence.

cd weights/
wget https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/converted_maskdino_r50_withoutclip.pth

```


## Joint Training



To train from scratch, it is necessary to follow the sequence of stages 1 and 2, executing the training scripts in order, with each stage building upon the weights from the previous one. 

For training on a single machine, you can execute the following command:

```bash
python3 projects/EVAP/train_net.py --config-file projects/EVAP/configs/images/<config_stageX.yaml> --num-gpus 8
```

Replace `<config_stageX.yaml>` with the actual configuration file for each stage: 

```
${EVAP_ROOT} 
    -- projects
        -- EVAP
        		-- configs
            		-- images
            				-- Lite
            						-- Final_Stage1_baseline_new_enc3_dec3_evaclipl_64g.yaml
            						-- Final_Stage2_baseline_new_enc3_dec3_evaclipl_64g.yaml
            				-- Plus
            						-- Final_Stage1_baseline_new_enc3_dec3_evaclipl_64g_swint.yaml
            						-- Final_Stage2_baseline_new_enc3_dec3_evaclipl_64g_swint.yaml
            				-- Tiny
            						-- Final_Stage1_baseline_new_enc3_dec3_evaclipl_64g_L2.yaml
            						-- Final_Stage2_baseline_new_enc3_dec3_evaclipl_64g_L2.yaml
```



Our standard setup involves training on multiple machines (64 x A100), for which you can use the distributed training script:

```bash
python3 launch.py --nn <num_machines>  --port <PORT> --worker_rank <Global_Rank> --master_address $<MASTER_ADDRESS>  --config-file projects/EVAP/configs/<config_stageX.yaml>
```

Here, `<num_machines>` should be replaced with the number of machines you intend to use, `<MASTER_ADDRESS>` should be the IP address of node 0. `<PORT>` should be the same among multiple nodes. , and `<config.yaml>` with the configuration file for the specific stage of training.

