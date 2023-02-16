exp_name = 'edsr_x4c64b16_g1_300k_div2k'

scale = 4
# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='HAEDSR',
        deg_head=dict(
			type='ResNet',
			depth=18,
			in_channels=3,
			frozen_stages=4,
			out_indices=[4],  # 0: conv-1, x: stage-x
			init_cfg=dict(type='Pretrained', checkpoint='/home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_aniso/weights_2000.pth')
		      ),        
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale,
        res_scale=1,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        deg_rand = True,
        ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
#train_dataset_type = 'SRFolderDataset'
train_dataset_type = 'SRMultiFolderDataset'
#val_dataset_type = 'SRFolderDataset'
val_dataset_type = 'SRMultiFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    # dict(
    #     type='Normalize',
    #     keys=['lq', 'gt'],
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=196),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    # dict(
    #     type='Normalize',
    #     keys=['lq', 'gt'],
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     to_rgb=True),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folders=['data/MultiDegrade/DIV2K_aniso/X4/train/sig_0.2_4.0theta_00',
            		 'data/MultiDegrade/DIV2K_aniso/X4/train/sig_0.2_4.0theta_0.63',
            		 'data/MultiDegrade/DIV2K_aniso/X4/train/sig_0.2_4.0theta_1.26',
            		 'data/MultiDegrade/DIV2K_aniso/X4/train/sig_0.2_4.0theta_1.88',
            		 'data/MultiDegrade/DIV2K_aniso/X4/train/sig_0.2_4.0theta_2.51',
            		 #'data/MultiDegrade/DIV2K/X4/train/sig_0.5',
                    	 #'data/MultiDegrade/DIV2K/X4/train/sig_01',
                    	 #'data/MultiDegrade/DIV2K/X4/train/sig_02',
                   	 #'data/MultiDegrade/DIV2K/X4/train/sig_03',
                     	 #'data/MultiDegrade/DIV2K/X4/train/sig_04',
            		],
            gt_folder='data/MultiDegrade/DIV2K/gt/train',
            pipeline=train_pipeline,
            scale=scale,
            filename_tmpl = '{}'
            )),
    val=dict(
        type=val_dataset_type,
        lq_folders=[#'data/MultiDegrade/DIV2K/X4/val/sig_0.5',
                    #'data/MultiDegrade/DIV2K/X4/val/sig_01',
                    'data/MultiDegrade/DIV2K/X4/val/sig_02',
                    #'data/MultiDegrade/DIV2K/X4/val/sig_03',
                    #'data/MultiDegrade/DIV2K/X4/val/sig_04',
                    #'data/MultiDegrade/DIV2K_aniso/X4/val/sig_0.2_4.0theta_00',
                    #'data/MultiDegrade/DIV2K_aniso/X4/val/sig_0.2_4.0theta_0.63',
                    #'data/MultiDegrade/DIV2K_aniso/X4/val/sig_0.2_4.0theta_1.26',
                    #'data/MultiDegrade/DIV2K_aniso/X4/val/sig_0.2_4.0theta_1.88',
                    #'data/MultiDegrade/DIV2K_aniso/X4/val/sig_0.2_4.0theta_2.51',
                   ],
        gt_folder='data/MultiDegrade/DIV2K/gt/val',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folders=[#'data/MultiDegrade/DIV2K/X4/test/sig_0.5',
                    #'data/MultiDegrade/DIV2K/X4/test/sig_01',
                    #'data/MultiDegrade/DIV2K/X4/test/sig_02',
                    #'data/MultiDegrade/DIV2K/X4/test/sig_03',
                    #'data/MultiDegrade/DIV2K/X4/test/sig_04',
                    'data/MultiDegrade/DIV2K_aniso/X4/test/sig_0.2_4.0theta_00',            		 
                    'data/MultiDegrade/DIV2K_aniso/X4/test/sig_0.2_4.0theta_0.63',            		 
                    'data/MultiDegrade/DIV2K_aniso/X4/test/sig_0.2_4.0theta_1.26',            		 
                    'data/MultiDegrade/DIV2K_aniso/X4/test/sig_0.2_4.0theta_1.88',            		 
                    'data/MultiDegrade/DIV2K_aniso/X4/test/sig_0.2_4.0theta_2.51',
                   ],
        gt_folder='data/MultiDegrade/DIV2K/gt/test',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 100000
lr_config = dict(policy='Step', by_epoch=False,step=[20000, 40000, 60000, 80000], gamma=0.5)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=500000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None#'work_dirs/edsr_x2c64b16_g1_300k_div2k/iter_300000.pth'
resume_from = None
workflow = [('train', 1)]
