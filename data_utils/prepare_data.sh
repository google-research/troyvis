gcloud config set storage/parallel_composite_upload_enabled True
mkdir -p datasets
{
    # COCO
    gsutil -m cp -r gs://xcloud-shared/masterbin/datasets/COCO datasets/
    mv datasets/COCO datasets/coco
    cd datasets/coco
    unzip -qq annotations_trainval2017.zip; rm annotations_trainval2017.zip
    unzip -qq train2017.zip; rm train2017.zip
    unzip -qq val2017.zip; rm val2017.zip
    cd ../..; 
    echo "COCO is ready"
} &
{
    # LVIS
    gsutil -m cp -r gs://xcloud-shared/masterbin/datasets/LVIS datasets/
    mv datasets/LVIS datasets/lvis
    cd datasets/lvis
    unzip -qq lvis_v1_train.json.zip; rm lvis_v1_train.json.zip
    unzip -qq lvis_v1_val.json.zip; rm lvis_v1_val.json.zip
    cd ../..
    echo "LVIS is ready"
} &
{
    # VG
    gsutil -m cp -r gs://xcloud-shared/masterbin/datasets/visual_genome datasets/
    cd datasets/visual_genome
    mkdir -p annotations
    mv *.json annotations
    (unzip -qq images.zip; rm images.zip) & (unzip -qq images2.zip; rm images2.zip)
    wait
    mkdir -p images
    mv VG_100K/* images; rm -rf VG_100K
    mv VG_100K_2/* images; rm -rf VG_100K_2
    cd ../..
    echo "VG is ready"
} &
{
    # VIS19
    gsutil -m cp -r gs://xcloud-shared/masterbin/datasets/ytvis_2019 datasets/
    cd datasets/ytvis_2019
    unzip -qq train.zip; rm -rf train.zip
    mkdir annotations
    # mv train.json annotations
    mv ytvis19_cocofmt.json annotations
    cd ../..
    echo "ytvis_2019 is ready"
} &
{
    # VIS21
    gsutil -m cp -r gs://xcloud-shared/masterbin/datasets/ytvis_2021 datasets/
    cd datasets/ytvis_2021
    unzip -qq train.zip; rm -rf train.zip
    mkdir annotations
    # mv train/instances.json annotations
    mv ytvis21_cocofmt.json annotations
    cd ../..
    echo "ytvis_2021 is ready"
} &
{
    # OVIS
    gsutil -m cp -r gs://xcloud-shared/masterbin/datasets/ovis datasets/
    cd datasets/ovis
    unzip -qq train.zip; rm -rf train.zip
    cd ../..
    echo "ovis is ready"
} &
{
    # LV-VIS
    gsutil -m cp -r gs://xcloud-shared/masterbin/datasets/lvvis datasets/
    cd datasets/lvvis
    unzip -qq val.zip; rm val.zip
    mv val_instances_.json val_instances.json
    cd ../..
    echo "lvvis is ready"
} &
wait
echo "all datasets are ready"






