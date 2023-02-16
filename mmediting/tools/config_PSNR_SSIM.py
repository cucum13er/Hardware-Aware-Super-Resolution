exp_name = 'PSNR&SSIM calculation'

scale = 2
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)


# train_dataset_type = 'SRMultiFolderLabeledDataset'
#train_dataset_type = 'SRMultiFolderDataset'
# val_dataset_type = 'SRMultiFolderLabeledDataset'
val_dataset_type = 'SRMultiFolderDataset'

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
    # dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    # dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    val=dict(
        type=val_dataset_type,
        lq_folders=[
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_0.5',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_01',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_02',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_03',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_04',
                    # 'data/Set5/X4/lq/sig_0.5',            		 
                    # 'data/Set5/X4/lq/sig_1.0',            		 
                    # 'data/Set5/X4/lq/sig_2.0',
                    # 'data/Set5/X4/lq/sig_3.0',
                    # 'data/Set5/X4/lq/sig_4.0',
                    'data/Set5/X2/lq/sig_4.0',
                    # 'data/BSD100/X4/lq/sig_1.0',
                    # 'data/Urban100/X2/lq/sig_0.5',
                   ],
        gt_folder= 'data/Set5/X2/gt/',
        # gt_folder= 'data/Urban100/X2/gt/',    
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folders=[
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_0.5',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_01',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_02',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_03',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_04',
                    # 'data/Set5/X4/lq/sig_0.5',            		 
                    # 'data/Set5/X4/lq/sig_1.0',            		 
                    # 'data/Set5/X4/lq/sig_2.0',
                    # 'data/Set5/X4/lq/sig_3.0',
                    # 'data/Set5/X2/lq/sig_1.0',
                    # '/home/rui/Rui_SR/mmediting/work_dirs/restorers/real-esrgan/Set14/X2/sig_4.0',
                    # '/home/rui/Rui_SR/mmediting/work_dirs/restorers/real-esrgan/BSD100/X2/sig_4.0',
                    '/home/rui/Rui_SR/mmediting/work_dirs/restorers/real-esrgan/Urban100/X2/sig_4.0',
                   ],
        # gt_folder= '/home/rui/Rui_SR/mmediting/data/Set14/X2/gt/',
        # gt_folder= '/home/rui/Rui_SR/mmediting/data/BSD100/X2/gt/',
        gt_folder= '/home/rui/Rui_SR/mmediting/data/Urban100/X2/gt/',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))


