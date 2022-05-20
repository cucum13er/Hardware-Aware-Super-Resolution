exp_name = 'dasr_x4c64b16_g1_100k_div2kflickr'

scale = 4
# model settings
model = dict(
    	type='BlindSR',
        train_contrastive=True,
    	generator=dict(
            		type='DASR',  
                    in_channels=3,
                    out_channels=3,
                    mid_channels=64,
                    num_blocks=5,
                    num_groups=5,
                    upscale_factor=scale,
                    reduction=8,
                    ),
        deg_head = dict(type='EasyRes', in_channels=3),
  		# deg_head=dict(
  		# 		type='ResNet',
  		# 		depth=18,
  		# 		in_channels=3,
  		# 		frozen_stages=0,
  		# 		out_indices=[4],  # 0: conv-1, x: stage-x
  		# 		init_cfg=dict(type='Pretrained', checkpoint='/home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_aniso/weights_2000.pth')
  		#       ),  
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        neck = dict(
        			type='NonLinearNeck',  # SimCLR non-linear neck
        			in_channels=512,
        			hid_channels=2048,
        			out_channels=128,
        			num_layers=2,
        			with_avg_pool=True
        	    ),
        contrastive_loss = dict(type='ContrastiveLoss', temperature=0.1),
        )
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
# dataset settings
train_dataset_type = 'SRMultiFolderLabeledDataset'
#train_dataset_type = 'SRMultiFolderDataset'
# val_dataset_type = 'SRMultiFolderLabeledDataset'
val_dataset_type = 'SRMultiFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[1, 25],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 3],
            poisson_gray_noise_prob=0.4),
        keys=['lq'],
    ),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=192),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt','gt_label'], meta_keys=['lq_path', 'gt_path', 'gt_label']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=32, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_root = 'data/DIV2K_Flickr2K/lq_aniso/X4',
            # lq_folders=['data/MultiDegrade/DIV2K/X4/train/sig_0.5',
            # 		 'data/MultiDegrade/DIV2K/X4/train/sig_01',
            # 		 'data/MultiDegrade/DIV2K/X4/train/sig_02',
            # 		 'data/MultiDegrade/DIV2K/X4/train/sig_03',
            # 		 'data/MultiDegrade/DIV2K/X4/train/sig_04',
            		# ],
            gt_folder='data/DIV2K_Flickr2K/gt/',
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
                   ],
        gt_folder='data/MultiDegrade/DIV2K/gt/val',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    # Test dataset is to be revised
    test=dict(
        type=val_dataset_type,
        lq_folders=['data/MultiDegrade/DIV2K/X4/test/sig_0.5',
                    'data/MultiDegrade/DIV2K/X4/test/sig_01',
                    'data/MultiDegrade/DIV2K/X4/test/sig_02',
                    'data/MultiDegrade/DIV2K/X4/test/sig_03',
                    'data/MultiDegrade/DIV2K/X4/test/sig_04',
                   ],
        gt_folder='data/MultiDegrade/DIV2K/gt/test',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(
                # type='Adam', lr=1e-3, betas=(0.9, 0.999),
		generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999) ),
        neck=dict(type='Adam', lr=1e-2, betas=(0.9, 0.999) ),
        deg_head=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999) ),

        )

# learning policy
total_iters = 100000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[20000, 40000, 60000, 80000],
    gamma=0.5)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=500000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None#'work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k/iter_90000.pth'#None
workflow = [('train', 1)]
