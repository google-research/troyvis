import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", type=str, required=True)
    parser.add_argument("--force_final", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dir_name = args.dir_name
    server_folder = f'gs://xcloud-shared/masterbin/exp/{dir_name}'
    local_folder = f'/usr/local/google/home/masterbin/code/eva-perceiver/exp/{dir_name}'
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    last_ckpt_server = server_folder + "/log.txt"
    last_ckpt_local = local_folder + "/log.txt"
    os.system(f"gsutil -m cp {last_ckpt_server} {last_ckpt_local}")
