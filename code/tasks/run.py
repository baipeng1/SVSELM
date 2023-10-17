import importlib
import sys
sys.path.append('/home2/baipeng/project/conformerlocal/DiffSinger/')
from utils.hparams import set_hparams, hparams
import os
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
# 线程为1方便调试
import torch
torch.set_num_threads(1)
os.environ ['OMP_NUM_THREADS'] ="1"
os.environ ['MKL_NUM_THREADS'] ="1"

def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()


if __name__ == '__main__':
    set_hparams()
    run_task()
