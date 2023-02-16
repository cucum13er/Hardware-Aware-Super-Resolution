exp_name = 'rdn_x2c64b16_g1_1000k_div2k'

scale = 2
# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='RDN',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale, save_image=True)

# dataset settings
train_dataset_type = 'SRMultiFolderLabeledDataset'
# val_dataset_type = 'SRMultiFolderDataset'
#val_dataset_type = 'SROurDataset_val'
val_dataset_type = 'SRDRealSR'
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
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=64),
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
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=1146 ), # crop the new lq and gt
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/DIV2K/DIV2K_train_LR_bicubic/X2_sub',
            gt_folder='data/DIV2K/DIV2K_train_HR_sub',
            ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',
            pipeline=train_pipeline,
            scale=scale)),
    # val=dict(
    #     type=val_dataset_type,
    #     lq_folder='data/val_set5/Set5_bicLRx2',
    #     gt_folder='data/val_set5/Set5_mod12',
    #     pipeline=test_pipeline,
    #     scale=scale,
    #     filename_tmpl='{}'),
    # test=dict(
    #     type=val_dataset_type,
    #     lq_folders=[
    #                 # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_0.5',
    #                 # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_01',
    #                 # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_02',
    #                 # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_03',
    #                 # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_04',
    #                 # 'data/Set5/X4/lq/sig_0.5',            		 
    #                 # 'data/Set5/X4/lq/sig_1.0',            		 
    #                 # 'data/Set5/X4/lq/sig_2.0',
    #                 # 'data/Set5/X4/lq/sig_3.0',
    #                  # 'data/Set5/X2/lq/sig_1.0',
    #                 # 'data/Set14/X2/lq/sig_4.0',
    #                 # 'data/BSD100/X2/lq/sig_4.0',
    #                 'data/Urban100/X2/lq/sig_4.0',                    
    #                ],
    #     # gt_folder= 'data/Set5/X2/gt/',
    #     # gt_folder= 'data/Set14/X2/gt/',
    #     gt_folder= 'data/Urban100/X2/gt/',   
    #     # gt_folder= 'data/BSD100/X2/gt/',           
    #     pipeline=test_pipeline,
    #     scale=scale,
    #     filename_tmpl='{}'),
    # for ours
    # val=dict(
    #     type=val_dataset_type,
    #     gt_folder= 'data/Ours_noNew/X2_test',
    #     pipeline=test_pipeline,
    #     scale=scale,
    #     filename_tmpl='{}'),
    # test=dict(
    #     type=val_dataset_type,
    #     gt_folder= 'data/Ours_noNew/X2_test',
    #     pipeline=test_pipeline,
    #     scale=scale,
    #     filename_tmpl='{}'),   
    # for DRealSR
    val=dict(
        type=val_dataset_type,
        gt_folder= '/media/rui/Samsung4TB/DRealSRplusImagePairs/Test_x2/test_LR',
        pipeline=test_pipeline,
        scale=scale,
        num_views=1,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        gt_folder= '/media/rui/Samsung4TB/DRealSRplusImagePairs/Test_x2/test_LR',
        pipeline=test_pipeline,
        scale=scale,
        num_views=1,
        filename_tmpl='{}')
    )

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 1000000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[200000, 400000, 600000, 800000],
    gamma=0.5)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
