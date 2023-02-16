# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
# from pathlib import Path
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS
import numpy as np
import copy
# import torch
# from mmcv import scandir


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

def get_samples_deg(root, folder_to_idx, extensions):
    # root = /train, folder_to_idx = {'n01443537': 0, 'n01629819': 1, } 
    # extensions = JPEG
    
    samples = []
    root = osp.expanduser(root)
    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = osp.join(root, folder_name)
        for path, subdirs, fns in sorted(os.walk(_dir)):
            # if "target_S1" in path:
                for fn in fns:
                    if has_file_allowed_extension(fn, extensions):
                        if "LR" in fn:
                            ptmp = osp.join(path, fn)
                            item = (ptmp, folder_to_idx[folder_name])
                            samples.append(item)
    # breakpoint()
    return samples
@DATASETS.register_module()
class SROurDataset_val(BaseSRDataset):
    """General paired image folder dataset for image restoration.

    The dataset loads lq (Low Quality) and gt (Ground-Truth) image pairs,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "folder mode", which needs to specify the lq folder path and gt
    folder path, each folder containing the corresponding images.
    Image lists will be generated automatically. You can also specify the
    filename template to match the lq and gt pairs.

    For example, we have two folders with the following structures:

    ::

        root
        ├── C640
        │   ├── target_F1
        │   ├── target_F2
        │   ├── target_J1
        │   ├── ...
        ├── C1300
        │   ├── target_F1
        │   ├── target_F2
        │   ├── target_J1
        │   ├── ...
        ├── C4112
        │   ├── target_F1
        │   ├── target_F2
        │   ├── target_J1
        ...

    then, you need to set:

    .. code-block:: python

        filename_tmpl = '{}_x4'

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (List[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
    """
    
    IMG_EXTENSIONS = ('.mat')
    def __init__(self,
                 # lq_root,
                 gt_folder,
                 pipeline,                 
                 scale,
                 num_views=2,
                 test_mode=False,
                 filename_tmpl='{}'):
        super().__init__(pipeline, scale, test_mode)
        # assert isinstance(lq_folders, list)
        # for lq_folder in lq_folders:
        # assert len(num_views) == len(pipeline)
        assert isinstance(num_views, int)
        self.num_views = num_views
        # self.lq_root = str(lq_root)
        self.gt_folder = str(gt_folder)
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        
        """Load annoations for SR dataset.

        It loads the LQ and GT image path from folders.
        It also gives each LQ a label of which device it belongs to

        Returns:
            dict: Returned dict for LQ and (GT_image, GT_label) pairs.
        """
        # breakpoint()
        data_infos = []
        # lq_paths = []
        folder_to_idx = find_folders(self.gt_folder)
        # num_lqs = len(self.lq_folders)
        # gt_paths = self.scan_folder(self.gt_folder)
        # breakpoint()
        samples = get_samples_deg(
            self.gt_folder,
            folder_to_idx,
            extensions=self.IMG_EXTENSIONS)
        if len(samples) == 0:
            raise (RuntimeError('Found 0 files in subfolders of: '
                                f'{self.data_prefix}. '
                                'Supported extensions are: '
                                f'{",".join(self.IMG_EXTENSIONS)}'))
            
        data_infos = []
        # breakpoint()
        for i, (filename, gt_label) in enumerate(samples):
            ############            
            # print(filename, gt_label, '1111111111111111\n' )
            # breakpoint()
            ##############
            info = {'root': self.gt_folder}
            info['lq_path'] = filename
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['idx'] = int(i)
            gt_name = filename.replace("LR", "HR")
            info['gt_path'] = gt_name
            data_infos.append(info)   
            
            
        # for i, lq_folder in enumerate(self.lq_folders):
        #     lq_paths.append(self.scan_folder(lq_folder))
        # # lq_paths = self.scan_folder(self.lq_folder)
        #     assert len(lq_paths[i]) == len(gt_paths), (
        #     f'gt and lq datasets have different number of images: '
        #     f'{len(lq_paths)}, {len(gt_paths)}.')
        
        # for gt_path in gt_paths:
        #     basename, ext = osp.splitext(osp.basename(gt_path))
        #     for i in range(len(lq_paths)):
                
        #         lq_path = osp.join(self.lq_folders[i],
        #                        (f'{self.filename_tmpl.format(basename)}'
        #                         f'{ext}'))
        #         assert lq_path in lq_paths[i], f'{lq_path} is not in lq_paths.'
        #         data_infos.append(dict(lq_path=lq_path, gt_path=gt_path))
        #         # data_infos.append(dict(lq_path=lq_path, (gt_path=gt_path, gt_label=gt_label) ) )
        # breakpoint()
        '''
        data_infos format:
            a list of dict like:
            {'lq_root': 'data/MultiDegrade/DIV2K_tiny/X4/train', 
             'lq_path': 'data/MultiDegrade/DIV2K_tiny/X4/train/sig_0.2_4.0theta_0.63/0001.png', 
             'gt_label': array(0), 
             'idx': 0, 
             'gt_path': 'data/MultiDegrade/DIV2K_tiny/gt/train/0001.png'}

        '''
        return data_infos
    
    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        # breakpoint()
        # print(idx)
        results = copy.deepcopy(self.data_infos[idx])
        results['scale'] = self.scale
        # breakpoint()
        
        final_res = list()
        for _ in range(self.num_views):
            final_res.append(self.pipeline(results))
            
        
        # final_res = self.pipeline(results)
        # for i in range(self.num_views):
        #     if i != 0:
        #         tmp = self.pipeline(results)
        #         for key in final_res.keys():
        #             if key == ('lq' or 'gt'):
        #                 # breakpoint()
        #                 final_res[key] = np.concatenate((final_res[key], tmp[key]) )
        #             elif key == ('gt_label'):
        #                 final_res[key] = np.append(final_res[key], tmp[key] )
            
            # print(i, batch_res['lq'])
        # print(final_res['gt_label'])
        return final_res

# if __name__ == "__main__":
#     folder_root = "/home/rui/Rui_SR/Datasets/Ours/X2"
#     test_data = SROurDataset()
