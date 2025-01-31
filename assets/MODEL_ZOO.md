# EVAP MODEL ZOO

## Introduction
EVAP maintains state-of-the-art (SOTA) performance across multiple tasks while preserving versatility and openness, demonstrating strong generalization capabilities. Here, we provide the model weights for all three stages of EVAP: '-pretrain', '-joint', and '-scaleup'. The '-pretrain' weights refer to those pretrained on Objects365 and OpenImages, yielding effective initializations from over three million detection data. The '-joint' weights are derived from joint training on 15 datasets, where the model achieves optimal performance. 

###  Stage 1: Pretraining 

|        Name        |                            Config                            |                            Weight                            |
| :----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| EVAP-Lite-pretrain |     Stage1_pretrain_openimage_obj365_CLIPfrozen_R50.yaml     | [Model]() |
| EVAP-Plus-pretrain |    Stage1_pretrain_openimage_obj365_CLIPfrozen_SwinL.yaml    | [Model]() |



### Stage 2: Image-level Joint Training 

|      Name       |                    Config                     |                            Weight                            |
| :-------------: | :-------------------------------------------: | :----------------------------------------------------------: |
| EVAP-Lite-joint |  Stage2_joint_training_CLIPteacher_R50.yaml   | [Model]() |
| EVAP-Plus-joint |    Stage2_joint_training_CLIPteacher_SwinL    | [Model]() |

