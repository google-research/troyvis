# COCO and LVIS
python3 projects/EVAP/train_net.py --config-file projects/EVAP/configs/images/Tiny/Final_Stage2_baseline_new_enc3_dec3_evaclipl_64g_L2.yaml  --num-gpus 8 --eval-only --resume

# LVVIS
# Step1: save language embeddings
python3 projects/EVAP/train_net.py --config-file projects/EVAP/configs/images/Tiny/Final_Stage2_baseline_new_enc3_dec3_evaclipl_64g_L2.yaml  --num-gpus 1 --eval-only --resume DATASETS.TEST '("lvvis_val", )' INPUT.MIN_SIZE_TEST 480 MODEL.REUSE_KERNEL True MODEL.USE_OFFLINE_CLS_EMBED False DATALOADER.NUM_WORKERS 0 MODEL.META_ARCHITECTURE EVAP_DPL
# Step2: test and evaluate
python3 projects/EVAP/train_net.py --config-file projects/EVAP/configs/images/Tiny/Final_Stage2_baseline_new_enc3_dec3_evaclipl_64g_L2.yaml  --num-gpus 8 --eval-only --resume DATASETS.TEST '("lvvis_val", )' INPUT.MIN_SIZE_TEST 480 MODEL.REUSE_KERNEL True MODEL.USE_OFFLINE_CLS_EMBED True MODEL.META_ARCHITECTURE EVAP_DPL_ONNX
python -m evaluate.mAP --dt_path results.json

# BURST
# Step1: save language embeddings
python3 projects/EVAP/train_net.py --config-file projects/EVAP/configs/images/Tiny/Final_Stage2_baseline_new_enc3_dec3_evaclipl_64g_L2.yaml  --num-gpus 1 --eval-only --resume DATASETS.TEST '("BURST_video_val",)' INPUT.MIN_SIZE_TEST 720 MODEL.REUSE_KERNEL True MODEL.USE_OFFLINE_CLS_EMBED False DATALOADER.NUM_WORKERS 0 MODEL.META_ARCHITECTURE EVAP_DPL
# Step2: test and evaluate
python3 projects/EVAP/train_net.py --config-file projects/EVAP/configs/images/Tiny/Final_Stage2_baseline_new_enc3_dec3_evaclipl_64g_L2.yaml  --num-gpus 8 --eval-only --resume DATASETS.TEST '("BURST_video_val",)' INPUT.MIN_SIZE_TEST 720 MODEL.REUSE_KERNEL True MODEL.USE_OFFLINE_CLS_EMBED True MODEL.META_ARCHITECTURE EVAP_DPL_ONNX MODEL.REUSE_KERNEL_CLIP_LENGTH 1
