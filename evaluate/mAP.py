from .lvvis import LVVIS
from .lvviseval import LVVISeval
import sys
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt_path", default=None, type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    gt_path = 'datasets/lvvis/val_instances.json'
    dt_path = args.dt_path

    ytvosGT = LVVIS(gt_path)
    ytvosDT = ytvosGT.loadRes(dt_path)
    ytvosEval = LVVISeval(ytvosGT, ytvosDT, "segm")
    ytvosEval.evaluate()
    ytvosEval.accumulate()
    ytvosEval.summarize()