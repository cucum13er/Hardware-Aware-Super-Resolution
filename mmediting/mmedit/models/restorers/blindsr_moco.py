# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16
from ..builder import build_backbone, build_component, build_loss
from ..registry import MODELS
import torch
import numbers
import os.path as osp
import mmcv
from mmedit.core import psnr, ssim, tensor2img
from ..base import BaseModel
import torch.distributed as dist
import time

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [
            torch.zeros_like(input) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
    
@MODELS.register_module()
class BlindSR_MoCo(BaseModel):
    """Basic model for image restoration.

    It must contain a generator that takes an image as inputs and outputs a
    restored image. It also has a pixel-wise loss for training.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 pixel_loss,
                 contrastive_part=None,
                 # deg_head=None,
                 # neck = None,
                 # contrastive_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 train_contrastive=False,
                 contrastive_loss_factor=1,
                 ):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.train_contrastive = train_contrastive
        # support fp16
        self.fp16_enabled = False
        self.contrastive_loss_factor = contrastive_loss_factor
        # generator
        self.generator = build_component(generator)
        self.contrastive_part = build_component(contrastive_part)
        # self.deg_head = build_component(deg_head)
        self.init_weights(pretrained)
        # self.neck = build_component(neck)
        # loss
        self.pixel_loss = build_loss(pixel_loss)
        # self.contrastive_loss = build_loss(contrastive_loss) if contrastive_loss else None
    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt, **kwargs)



    def forward_train(self, lq, gt, gt_label, **kwargs):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt:
                (Tensor): GT Tensor with shape (n, c, h, w).
                (int): GT label with class 
        Returns:
            Tensor: Output tensor.
        """
        # breakpoint()
        losses = dict()
        # breakpoint()
        # contrastive_feat = self.deg_head(lq)
        # breakpoint()
        loss_contrastive, contrastive_feat = self.contrastive_part(lq, label=gt_label)###############
        if not self.train_contrastive:
        # pixel loss part
            sr_image = self.generator(lq, contrastive_feat)
            loss_pix = self.pixel_loss(sr_image, gt)
            losses['loss_pix'] = loss_pix########################
        
        # contrastive loss part
        losses.update((x, y*self.contrastive_loss_factor) for x, y in loss_contrastive.items())
        
        # losses['loss_contrastive'] = loss_contrastive * self.contrastive_loss_factor
        if self.train_contrastive:
            outputs = dict(
                losses=losses,
                num_samples=len(gt.data),
                # results=dict(lq=lq.cpu(), 
                #              gt=gt.cpu(), 
                #              # output=sr_image.cpu(),
                #              )
                )
        else:
            outputs = dict(
                losses=losses,
                num_samples=len(gt.data),
                results=dict(lq=lq.cpu(), 
                             gt=gt.cpu(), 
                             output=sr_image.cpu(),
                             )
                )
        # breakpoint()
        return outputs
    def evaluate(self, output, gt):
        """Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border

        output = tensor2img(output)
        gt = tensor2img(gt)
        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](output, gt,
                                                               crop_border)
        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        # breakpoint()
        # contrastive_feat = self.deg_head(lq)
        contrastive_feat = self.contrastive_part.module.extract_feat(lq) ############
        output = self.generator(lq, contrastive_feat)
        # output = self.generator(lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)
            # breakpoint()
        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out, _ = self.generator(img)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        
        if isinstance(data_batch, list):
            data_all = data_batch[0]
            for i in range(len(data_batch)):
                if i != 0:
                    for key in data_batch[i].keys():
                        if key != 'meta':
                            data_all[key] = torch.concat([data_all[key], data_batch[i][key]])
            
        # t = time.time()    
        # t = time.time() - t
        # print(t)
        # breakpoint()            
            
        outputs = self(**data_all, test_mode=False)
        # breakpoint()
        loss, log_vars = self.parse_losses(outputs.pop('losses'))
        # breakpoint()
        # print(log_vars)
        if not self.train_contrastive:
            optimizer['generator'].zero_grad()
        
        optimizer['contrastive_part'].zero_grad()
        # optimizer['neck'].zero_grad()
        # breakpoint()
        # optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        if not self.train_contrastive:
            optimizer['generator'].step()
            # breakpoint()
        optimizer['contrastive_part'].step()
        # optimizer['neck'].step()
        
        outputs.update({'log_vars': log_vars})
        
        # breakpoint()
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.
        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output






