





# Data Preparation

**Here details how to prepare all the datasets used in the training and testing stages of EVA-Perceiver.**

EVA-Perceiver used the following datasets for joint training, and perform zero-shot evaluation on additional datasets. For users who only want to test or continue fine-tune on part of the datasets, there is no need of downloading all datasets. 

## For Training



### COCO

Please download [COCO](https://cocodataset.org/#home) from the offical website. We use [train2017.zip](http://images.cocodataset.org/zips/train2017.zip), [train2014.zip](http://images.cocodataset.org/zips/train2014.zip), [val2017.zip](http://images.cocodataset.org/zips/val2017.zip), [test2017.zip](http://images.cocodataset.org/zips/test2017.zip) & [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [image_info_test2017.zip](http://images.cocodataset.org/annotations/image_info_test2017.zip). We expect that the data is organized as below.

```
${EVAP_ROOT}
    -- datasets
        -- coco
            -- annotations
            -- train2017
            -- train2014
            -- val2017
            -- test2017
```

### LVIS

Please download [LVISv1](https://www.lvisdataset.org/dataset) from the offical website. LVIS uses the COCO 2017 train, validation, and test image sets, so only Annotation needs to be downloaded：[lvis_v1_train.json.zip](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip),  [lvis_v1_val.json.zip](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip), [lvis_v1_minival_inserted_image_name.json](https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json). We expect that the data is organized as below.

```
${EVAP_ROOT}
    -- datasets
        -- lvis
            -- lvis_v1_train.json
            -- lvis_v1_val.json
            -- lvis_v1_minival_inserted_image_name.json
```

### VisualGenome

Please download [VisualGenome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) images from the offical website:  [part 1 (9.2 GB)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part 2 (5.47 GB)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip), and download our preprocessed annotation file: [train.json](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/annotations/VisualGenome/train.json), [train_from_objects.json](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/annotations/VisualGenome/train_from_objects.json) . We expect that the data is organized as below.

```
${EVAP_ROOT}
    -- datasets
        -- visual_genome
            -- images
            	-- *.jpg
            			...
            -- annotations
              -- train_from_objects.json
              -- train.json
```



### OpenImages

Please download [OpenImages v6](https://storage.googleapis.com/openimages/web/download_v6.html) images from the offical website, all detection annotations need to be preprocessed into coco format. We expect that the data is organized as below. 

```
${EVAP_ROOT}
    -- datasets
        -- openimages
            -- detection
            -- openimages_v6_train_bbox.json
```

### VIS

Download YouTube-VIS [2019](https://codalab.lisn.upsaclay.fr/competitions/6064#participate-get_data), [2021](https://codalab.lisn.upsaclay.fr/competitions/7680#participate-get_data), [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763#participate) dataset for video instance segmentation task, and it is necessary to convert their video annotation into coco format in advance for image-level joint-training by run: ```python3 conversion/conver_vis2coco.py```  We expect that the data is organized as below. 

```
${EVAP_ROOT}
    -- datasets
        -- ytvis_2019
            -- train
            -- val
            -- annotations
            		-- instances_train_sub.json
            		-- instances_val_sub.json
            		-- ytvis19_cocofmt.json
        -- ytvis_2021
            -- train
            -- val
            -- annotations
            		-- instances_train_sub.json
            		-- instances_val_sub.json
            		-- ytvis21_cocofmt.json
        -- ovis
            -- train
            -- val		
            -- annotations_train.json
            -- annotations_valid.json
            -- ovis_cocofmt.json
```


### Objects365

Following [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT), we prepare **Objects365** data, and we expect that they are organized as below:

```
${EVAP_ROOT}
    -- datasets
        -- Objects365v2
            -- annotations
                -- zhiyuan_objv2_train_new.json
                -- zhiyuan_objv2_val_new.json
            -- images


```



## For Evaluation Only

The following datasets are only used for zero-shot evaluation, and are not used in joint-training. 

### OmniLabel

Please download [OmniLabel](https://www.omnilabel.org/dataset/download) from the offical website, and download our converted annotation in coco formation: [omnilabel](). we expect that the data is organized as below. 

```
${EVAP_ROOT}
    -- datasets
        -- omnilabel
            -- images
            		-- coco
            		-- object365
            		-- openimagesv5
            -- omnilabel_coco.json
            -- omnilabel_obj365.json
            -- omnilabel_openimages.json
            -- omnilabel_cocofmt.json
```

### ODinW

We follow [GLIP](https://github.com/microsoft/GLIP) to prepare the ODinW 35 dataset, and run ```python3 download.py ``` to download it and organized as below. 

```
${EVAP_ROOT}
    -- datasets
        -- odinw 
            -- dataset
            		-- coAerialMaritimeDroneco
            		-- CottontailRabbits
            		-- NorthAmericaMushrooms
            		-- ...
           
```

### TAO&BURST

TAO and BURST share the same video frames.

First, download the validation set zip files (2-TAO_VAL.zip, 2_AVA_HACS_VAL_e49d8f78098a8ffb3769617570a20903.zip) and unzip them from https://motchallenge.net/tao_download.php.

Then, download our preprocessed YTVIS format (COCO-like) annotation files from huggingface:

https://huggingface.co/spaces/Junfeng5/GLEE_demo/tree/main/annotations/TAO

And organize them as below:

```
${EVAP_ROOT}
    -- datasets
        -- TAO 
            --burst_annotations
            		-- TAO_val_withlabel_ytvisformat.json
            		-- val
            				-- all_classes.json
            				-- ...
            --TAO_annotations
            	 	-- validation_ytvisfmt.json
            	 	-- validation.json
            -- frames
            		-- val
            			-- ArgoVerse
            			-- ava
            			-- ...
           
```

### LV-VIS
[LV-VIS](https://github.com/haochenheheda/LVVIS) is a large-scale open-vocabulary video instance segmentation benchmark.

First, download the validation set videos [val.zip](https://drive.google.com/file/d/1vTYUz_XLOBnYb9e7upJsZM-nQz2S6wDn/view) and annotations [val_instances_.json](https://drive.google.com/file/d/1stPD818M3gv7zUV3UIZG1Suru7Tk54jo/view), putting them under datasets/lvvis folder.

Then, unzip val.zip and rename val_instances_.json to val_instances.json

We expect the directory structure as below:

```
${EVAP_ROOT}
    -- datasets
        -- lvvis 
            --val
            --val_instances.json
           
```