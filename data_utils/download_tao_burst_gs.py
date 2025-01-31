import os
from multiprocessing.pool import ThreadPool
import time
from pathlib import Path

# Download TAO and BURST from GS


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
    os.system(f"unzip -qq {tar_path} -d {data_dir}")
    # rm zips
    os.system(f"rm {tar_path}")


if __name__ == "__main__":
    start_time = time.time()
    data_root = "datasets/TAO/"
    num_thread = 23
    num_tar = 23
    url = f"gs://xcloud-shared/masterbin/datasets/TAO/part"
    tar_url_list = []
    tar_url_list += [f'{url}/part_{i}.zip' for i in range(1, num_tar+1)]
    data_dir = os.path.join(data_root, "frames")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    pool = ThreadPool(num_thread)
    pool.imap(lambda x: download_unzip_tar(*x), [(tar_url, ) for tar_url in tar_url_list])  # multi-threaded
    pool.close()
    pool.join()

    # download annotations
    train_anno_url = f'gs://xcloud-shared/masterbin/datasets/TAO/burst_annotations'
    os.system(f"gsutil -m cp -r {train_anno_url} {data_root}")
    end_time = time.time()
    total_time_mins = (end_time - start_time) / 60
    print("Total Time: %01d mins" % total_time_mins)