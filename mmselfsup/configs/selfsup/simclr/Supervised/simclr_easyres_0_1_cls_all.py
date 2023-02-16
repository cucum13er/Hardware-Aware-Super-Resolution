_base_ = [
    '../../_base_/models/simclr_multidevice_easy_cls.py',
#    '../../_base_/datasets/multidevice_simclr_TestAll.py',
    '../../_base_/datasets/multidevice_simclr_DIV2K.py',
    '../../_base_/schedules/lars_coslr-200e_in1k.py',
    '../../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(
    type='LARS',
    lr=0.3,
    momentum=0.9,
    weight_decay=1e-6,
    paramwise_options={
        '(bn|gn)(\\d+)?.(weight|bias)':
        dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True)
    })

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-4,
    warmup_by_epoch=True)

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
runner = dict(type='EpochBasedRunner', max_epochs=2000) 
#load_from = 'work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_randomcrop/epoch_1000.pth'  # Runner to load ckpt
checkpoint_config = dict(interval=200, max_keep_ckpts=5)
