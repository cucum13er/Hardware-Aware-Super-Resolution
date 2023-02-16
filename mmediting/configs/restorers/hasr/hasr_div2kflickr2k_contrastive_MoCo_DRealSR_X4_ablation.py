exp_name = 'hasr_DRealSR'

scale = 4
# model settings
model = dict(
    	type='BlindSR_MoCo',
        train_contrastive=False,
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    	generator=dict(
            		type='HASR',  
                    in_channels=3,
                    out_channels=3,
                    mid_channels=64,
                    num_blocks=5,
                    num_groups=5,
                    upscale_factor=scale,
                    reduction=8,
                    scale = scale,
                    frozen_groups = 0,
                    ),
        contrastive_part = dict(
                    type='MoCo_label',
                    queue_len=8192,
                    feat_dim=64,
                    momentum=0.999,
                    backbone=dict(
                        type='EasyRes',
                        in_channels=3,
                        pretrained = None, #'/home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/moco/moco_easyres_DRealSR/weights_2000.pth',
                        ),
                    neck=dict(
                        type='MoCoV2Neck',
                        in_channels=512,
                        hid_channels=2048,
                        out_channels=64,
                        with_avg_pool=True),
                    head=dict(type='SNNLossHead', temperature=0.07)
                    ),
        contrastive_loss_factor = 0.00000001,
        )
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
# dataset settings
train_dataset_type = 'SRDRealSR'
val_dataset_type = 'SRDRealSR'
#val_dataset_type = 'SRMultiFolderDataset'
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
    #### add another view ########################
    dict(type='CopyValues', src_keys=['lq','gt'], dst_keys=['lq_tmp','gt_tmp']),
    dict(type='PairedRandomCrop', gt_patch_size=192), # only crop lq and gt
    dict(type='CopyValues', src_keys=['lq','gt'], dst_keys=['lq_view','gt_view']), # another view
    dict(type='CopyValues', src_keys=['lq_tmp','gt_tmp'], dst_keys=['lq','gt']), # the whole image back
    dict(type='PairedRandomCrop', gt_patch_size=192), # crop the new lq and gt
    # random flip and transpose for both views
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['lq_view', 'gt_view'], flip_ratio=0.5,
        direction='horizontal'),    
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='Flip', keys=['lq_view', 'gt_view'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='RandomTransposeHW', keys=['lq_view', 'gt_view'], transpose_ratio=0.5),
    # collect all views and label
    dict(type='Collect', keys=['lq', 'gt' ,'lq_view', 'gt_view', 'gt_label'], meta_keys=['lq_path', 'gt_path', 'gt_label']),
    dict(type='ImageToTensor', keys=['lq', 'gt', 'lq_view', 'gt_view'])
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
    dict(type='PairedRandomCrop', gt_patch_size=512), # crop the new lq and gt
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=32, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            gt_folder='/media/rui/Samsung4TB/DRealSR/Train_x4/train_LR',
            #gt_folder='/media/rui/Rui_T7/DRealSRplusImagePairs/Train_x2/train_LR',            
            pipeline=train_pipeline,
            scale=scale,
            num_views=2,
            filename_tmpl = '{}'
            )),

    
    val=dict(
        type=val_dataset_type,
        gt_folder= '/media/rui/Samsung4TB/DRealSR/Test_x4/test_LR',
        pipeline=test_pipeline,
        scale=scale,
        num_views=1,
        filename_tmpl='{}'),

    test=dict(
        type=val_dataset_type,
        gt_folder= '/media/rui/Samsung4TB/DRealSR/Test_x4/test_LR',
        pipeline=test_pipeline,
        scale=scale,
        num_views=1,
        filename_tmpl='{}')


    # val=dict(
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
    #                 # 'data/Set5/X4/lq/sig_4.0',
    #                 'data/Set5/X2/lq/sig_4.0',
    #                 # 'data/BSD100/X4/lq/sig_1.0',
    #                 # 'data/Urban100/X2/lq/sig_0.5',
    #                ],
    #     gt_folder= 'data/Set5/X2/gt/',
    #     # gt_folder= 'data/Urban100/X2/gt/',    
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
    #                 # 'data/Set5/X4/lq/sig_4.0',
    #                 # 'data/Set5/X2/lq/sig_4.0',
    #                 # 'data/Set14/X4/lq/sig_4.0',
    #                 # 'data/BSD100/X2/lq/sig_4.0',
    #                 'data/Urban100/X4/lq/sig_4.0',
    #                 # 'data/Urban100/X2/lq_aniso/sig_0.2_4.0theta_0.0',
    #                ],
    #     # gt_folder= 'data/Set5/X2/gt/',
    #     # gt_folder= 'data/Set14/X4/gt/',
    #     gt_folder= 'data/Urban100/X4/gt/',   
    #     # gt_folder= 'data/BSD100/X2/gt/',
    #     pipeline=test_pipeline,
    #     scale=scale,
    #     filename_tmpl='{}')        
        )

# optimizer
optimizers = dict(
                 # type='Adam', lr=1e-2, betas=(0.9, 0.999),
                 # contrastive_part=dict(type='LARS', lr=4.8, weight_decay=1e-6, momentum=0.9),
                 contrastive_part=dict(type='Adam', lr=1e-19, betas=(0.9, 0.999) ),
                 generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999) ),
        )

# learning policy
total_iters = 200000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[40000, 80000, 120000, 160000],
    gamma=0.5)

checkpoint_config = dict(interval=20000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=400000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None #'/home/rui/Rui_SR/mmediting/work_dirs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both_6-layer-pretrain/iter_200000.pth'
resume_from = None # '/home/rui/Rui_SR/mmediting/work_dirs/restorers/hasr/X4/hasr_1616/iter_40000.pth'
workflow = [('train', 1)]
