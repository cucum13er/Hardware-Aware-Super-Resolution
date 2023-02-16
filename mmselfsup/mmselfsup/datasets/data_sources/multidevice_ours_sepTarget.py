# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import mmcv
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from ..builder import DATASOURCES
from .base import BaseDataSource


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [d for d in os.listdir(root) if osp.isdir(osp.join(root, d))]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = osp.expanduser(root)
    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = osp.join(root, folder_name)
        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = osp.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples
############################## get_samples for tiny-imagenet ##################
def get_samples_all(root, folder_to_idx, extensions):
    # root = /train, folder_to_idx = {'n01443537': 0, 'n01629819': 1, } 
    # extensions = JPEG
    
    samples = []
    root = osp.expanduser(root)
    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = osp.join(root, folder_name, 'lq_gray/X4')
        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = osp.join(folder_name, 'lq_gray/X4', fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples
###############################################################################
############################## get_samples for tiny-imagenet ##################
def get_samples_deg(root, folder_to_idx, extensions):
    # root = /train, folder_to_idx = {'n01443537': 0, 'n01629819': 1, } 
    # extensions = JPEG
    
    samples = []
    root = osp.expanduser(root)
    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = osp.join(root, folder_name)
        for path, subdirs, fns in sorted(os.walk(_dir)):
            if "target_S1" not in path: # target_F1, target_F2, target_J1, target_S1
                for fn in fns:
                    if has_file_allowed_extension(fn, extensions):
                        if "LR" in fn:
                            ptmp = osp.join(path, fn)
                            item = (ptmp, folder_to_idx[folder_name])
                            samples.append(item)
    # breakpoint()
    return samples
###############################################################################
@DATASOURCES.register_module()
class MultiDevice_ours_sepTarget(BaseDataSource):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py
    """  # noqa: E501

    IMG_EXTENSIONS = ('.mat')

    # def __init__(self):
        # super(MultiDevice, self).__init__()
        # self.color_type = 'grayscale'

    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            # change it to get_samples_tiny for tiny-imagenet ##############
            samples = get_samples_deg(
            # samples = get_samples_all(
            # samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)

            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif self.ann_file:
            folder_to_idx = find_folders(self.data_prefix)
            # change it to get_samples_tiny for tiny-imagenet ##############
            samples = get_samples_deg(
            # samples = get_samples_all(
            # samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)

            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))            
        else:
            
            raise TypeError('ann_file must be None')
        self.samples = samples
        # print(self.samples,'1111111111111111111111111111\n')################################
        data_infos = []
        for i, (filename, gt_label) in enumerate(self.samples):
            ############
            # print(filename, gt_label, '1111111111111111\n' )
            ##############
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['idx'] = int(i)
            data_infos.append(info)
            # print(info['img_info'])
        # breakpoint()
        return data_infos
    def get_img(self, idx):
        """Get image by index.

        Args:
            idx (int): Index of data.

        Returns:
            Image: PIL Image format.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # if self.data_infos[idx]['img_prefix'] is not None:
        #     filename = osp.join(self.data_infos[idx]['img_prefix'],
        #                         self.data_infos[idx]['img_info']['filename'])
        # else:
        filename = self.data_infos[idx]['img_info']['filename']
        mat = sio.loadmat(filename)
        lastkey = list(mat)[-1]
        img = mat[lastkey]
        img = np.stack([img,img,img],axis=0)
        # print(filename,'1111111')
        # print(img.shape)
        # breakpoint()
        # img_bytes = self.file_client.get(filename)
        # img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        # img = img.astype(np.uint8)
        return torch.from_numpy(img).float()
