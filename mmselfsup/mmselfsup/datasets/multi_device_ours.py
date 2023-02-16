# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.utils import build_from_cfg, print_log
from torchvision.transforms import Compose

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy
import os.path as osp
@DATASETS.register_module()
class MultiDeviceDataset_ours(BaseDataset):
    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        assert len(num_views) == len(pipelines)
        self.data_source = build_datasource(data_source)
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)
        self.prefetch = prefetch

        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipelines[i]] * num_views[i])
        self.trans = trans
    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        # if img.shape == (3,1024,1280):
            # breakpoint()
        # print(img.shape)
        # print(img.dtype)
        multi_views = list(map(lambda trans: trans(img), self.trans))
        # print(multi_views,'111222')
        # breakpoint()
        if self.prefetch:
            multi_views = [
                torch.from_numpy(to_numpy(img)) for img in multi_views
            ]
        label = self.data_source.data_infos[idx]['gt_label']
        # breakpoint()
        return dict(img=multi_views, label=label) #########################
    
    def evaluate(self, results, logger=None, save_path=None ):
        import numpy as np
        eval_res = list()
        for score, label in results.items():
            # print(score)
            # print(label)
            pred = torch.argmax(score).numpy()
            tmp_res = np.array([pred,label])
            eval_res.append(tmp_res)
        
        eval_res = np.stack(eval_res,0)
        acc = np.sum(eval_res[:,0]==eval_res[:,1]) / len(eval_res)
            # breakpoint()
        
            # val = torch.from_numpy(val)
            # target = torch.LongTensor(self.data_source.get_gt_labels())
            # assert val.size(0) == target.size(0), (
            #     f'Inconsistent length for results and labels, '
            #     f'{val.size(0)} vs {target.size(0)}')

            # num = val.size(0)
            # _, pred = val.topk(max(topk), dim=1, largest=True, sorted=True)
            # pred = pred.t()
            # correct = pred.eq(target.view(1, -1).expand_as(pred))  # [K, N]
            # for k in topk:
            #     correct_k = correct[:k].contiguous().view(-1).float().sum(
            #         0).item()
            #     acc = correct_k * 100.0 / num
            #     eval_res[f'{name}_top{k}'] = acc
        if logger is not None and logger != 'silent':
            print_log(f'Accuracy: {acc:.03f}', logger=logger)
        return eval_res        
        """The evaluation function to output accuracy.

        Args:
            results (dict): The key-value pair is the output features and
                corresponding labels.
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        """        