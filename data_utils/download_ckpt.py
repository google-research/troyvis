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
    local_folder = f'/home/jupyter/code/eva-perceiver/exp/{dir_name}'
    last_ckpt_server = server_folder + "/last_checkpoint"
    last_ckpt_local = local_folder + "/last_checkpoint"
    os.system(f"gsutil -m cp {last_ckpt_server} {last_ckpt_local}")
    if os.path.exists(last_ckpt_local):
        if args.force_final:
            with open(last_ckpt_local, "w") as f:
                f.write("model_final.pth")
            ckpt_name = "model_final.pth"
        else:
            with open(last_ckpt_local, "r") as f:
                ckpt_name = f.readlines()[0].strip()
        ckpt_path_server = server_folder + f"/{ckpt_name}"
        ckpt_path_local = local_folder + f"/{ckpt_name}"
        os.system(f"gsutil -m cp {ckpt_path_server} {ckpt_path_local}")
