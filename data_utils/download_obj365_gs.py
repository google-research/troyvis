import os
from multiprocessing.pool import ThreadPool
import time
from tqdm import tqdm
from pathlib import Path

# This script is modified from https://github.com/ultralytics/yolov5/blob/master/data/Objects365.yaml
# Download Obj365 from GS

def download_unzip_tar(tar_url):
    tar_name = os.path.split(tar_url)[-1]
    tar_path = os.path.join(data_dir, tar_name)
    # download
    while True:
        if not os.path.exists(tar_path):
            os.system(f"gsutil -m cp {tar_url} {data_dir}")
        else:
            break
    # unzip
    os.system(f"tar xfz {tar_path} --directory {data_dir}")
    # rm zips
    os.system(f"rm {tar_path}")


if __name__ == "__main__":
    start_time = time.time()
    data_root = "datasets/Objects365V2/images"
    num_thread = 16
    for split, num_tar in [("train", 51), ("val", 44)]:
    # for split, num_tar in [("train", 10)]:
        url = f"gs://xcloud-shared/masterbin/datasets/Objects365V2/images/"
        tar_url_list = []
        if split == 'train':
            tar_url_list += [f'{url}patch{i}.tar.gz' for i in range(num_tar)]
        elif split == 'val':
            tar_url_list += [f'{url}val/patch{i}.tar.gz' for i in range(num_tar)]
        data_dir = os.path.join(data_root, split) if split == "val" else data_root
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        pool = ThreadPool(num_thread)
        pool.imap(lambda x: download_unzip_tar(*x), [(tar_url, ) for tar_url in tar_url_list])  # multi-threaded
        pool.close()
        pool.join()
        # Move
        images = Path(data_root)
        for f in tqdm(Path(data_dir).rglob('*.jpg'), desc=f'Moving {split} images'):
            f.rename(images / f.name)  # move to /images/{split}
    # download annotations
    anno_root = data_root.replace("images", "annotations")
    if not os.path.exists(anno_root):
        os.makedirs(anno_root)
    train_anno_url = f'gs://xcloud-shared/masterbin/datasets/Objects365V2/annotations/zhiyuan_objv2_train_new.json'
    # train_anno_url = f'gs://xcloud-shared/masterbin/datasets/Objects365V2/annotations/zhiyuan_objv2_train_new_sub10.json'
    # val_anno_url = f'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.tar.gz'
    os.system(f"gsutil -m cp {train_anno_url} {anno_root}")
    # os.system(f"wget -c {val_anno_url}")
    end_time = time.time()
    total_time_mins = (end_time - start_time) / 60
    print("Total Time: %01d mins" % total_time_mins)