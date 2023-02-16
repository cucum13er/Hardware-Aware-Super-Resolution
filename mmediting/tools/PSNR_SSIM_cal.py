#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:45:33 2022

@author: rui

evaluate PSNR and SSIM for given images and folders
"""
import argparse
# import os
from mmedit.core import psnr, ssim
import mmcv
import numpy as np
# import torch
# from mmcv.parallel import MMDataParallel
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint

# from mmedit.apis import multi_gpu_test, set_random_seed, single_gpu_test
# from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets import build_dataloader, build_dataset
# from mmedit.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='PSNR and SSIM calculation based on given datasets')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')

    parser.add_argument('--out', help='output result pickle file')

    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    distributed = True
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    cfg = mmcv.Config.fromfile(args.config)
    
    dataset = build_dataset(cfg.data.test)
    
    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **loader_cfg)
    crop_border = cfg.test_cfg.crop_border
    metrics = cfg.test_cfg.metrics
    res = dict()
    for metric in metrics:
        res[metric] = []
    
    for imgs in data_loader:
        sr = imgs['lq'].squeeze().numpy()
        gt = imgs['gt'].squeeze().numpy()
        for metric in metrics:
            res[metric].append( allowed_metrics[metric](sr, gt, crop_border=crop_border) )
    # print(res)
    avgPSNR = np.mean( np.array(res['PSNR']) )
    avgSSIM = np.mean( np.array(res['SSIM']) )
    print(f'The average PSNR is {avgPSNR}')
    print(f'The average SSIM is {avgSSIM}')
    
if __name__ == '__main__':
    main()