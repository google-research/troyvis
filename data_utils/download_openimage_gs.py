import os
from multiprocessing.pool import ThreadPool
import time
from tqdm import tqdm
from pathlib import Path

# This script is modified from download_obj365_gs.py
# Download OpenImage from GS

def download_unzip_tar(tar_url):
    tar_name = os.path.split(tar_url)[-1]
    # download
    os.system(f"gsutil -m cp {tar_url} {data_root}")
    # unzip
    tar_path = os.path.join(data_root, tar_name)
    os.system(f"unzip -qq {tar_path} -d {data_dir}")
    # rm zips
    os.system(f"rm {tar_path}")


if __name__ == "__main__":
    start_time = time.time()
    data_root = "datasets/openimages"
    num_tar = 103
    num_thread = 16
    url = f"gs://xcloud-shared/masterbin/datasets/openimages/part/"
    tar_url_list = []
    tar_url_list += [f'{url}part_{i}.zip' for i in range(1, num_tar+1)]
    data_dir = os.path.join(data_root, "detection")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    pool = ThreadPool(num_thread)
    pool.imap(lambda x: download_unzip_tar(*x), [(tar_url, ) for tar_url in tar_url_list])  # multi-threaded
    pool.close()
    pool.join()
    # download annotations
    anno_root = "datasets/openimages"
    train_anno_url = f'gs://xcloud-shared/masterbin/datasets/openimages/openimages_v6_train_bbox.json'
    os.system(f"gsutil -m cp {train_anno_url} {anno_root}")
    end_time = time.time()
    total_time_mins = (end_time - start_time) / 60
    print("Total Time: %01d mins" % total_time_mins)