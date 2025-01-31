# EVA-Perceiver: Efficient Open-Vocabulary Foundation Models for Object Perception

## Details of Our Final Model
**config file**: projects/EVAP/configs/images/Tiny/Final_Stage2_baseline_new_enc3_dec3_evaclipl_64g_L2.yaml

**model ckpt**: https://drive.google.com/drive/folders/18SqZT6X3XBIQ3Tiol7w38fh2eQ4pifl2?usp=drive_link


## How to launch a VScode for debugging on XCloud (GCP)

```bash xm_vscode.sh```

After the job is running, wait ~20 mintues for data downloading. Then input ```http://localhost:8080/?folder=/home/jupyter/code/eva-perceiver``` in browser on your cloudtop. Finally, you will have a vscode interface for debugging with GPUs.

## How to launch a training

```bash xm_dist_train_twostage.sh```

This script will do two-stage training sequentially. Please change pool_size (the number of nodes for distributed training, pool_size=8 meaning 64GPUs, pool_size=2 meaning 16GPUs), config1, config2, and dirname1

**How to test and evaluate our model on Xcloud**

On GCP VScode, run ```bash infer.sh```

