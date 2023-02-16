# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import numbers
import os.path as osp

import cv2
import mmcv
import numpy as np

from ..registry import PIPELINES

@PIPELINES.register_module()
class CatLayers:
    '''
    change GrayScale imgs into RGB 3 channels by copying the channel by 3 times
    '''
    def __init__(self, keys, save_original=False):
        self.keys = keys
        self.save_original = save_original

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        
        for key in self.keys:
            if isinstance(results[key], list):
                if self.save_original:
                    results[key + '_grayscale'] = [
                        v.copy() for v in results[key]
                    ]
                
                results[key] = [ np.stack([v,v,v], axis=2)
                    for v in results[key]
                ]
            else:
                if self.save_original:
                    results[key + '_grayscale'] = results[key].copy()
                results[key] = np.stack( [results[key],results[key],results[key]], axis=2)

        # results['img_norm_cfg'] = dict(
        #     mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        # print(results.keys(), '1111111')
        # print(results['lq'].shape, '2222222')
        # print(results['gt'].shape, '3333333')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys} '
                     f'save_original={self.save_original})')

        return repr_str    
