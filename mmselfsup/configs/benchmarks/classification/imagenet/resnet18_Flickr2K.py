_base_ = [
    '../_base_/models/resnet18_cls5.py',
    '../_base_/datasets/multidevice_simclr_Flickr2K.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    '../_base_/default_runtime.py',
]

model = dict(backbone=dict(frozen_stages=4))

# swav setting
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3

# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=1e-6)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10, max_keep_ckpts=10)
#load_from = 'work_dirs/benchmarks/classification/imagenet/resnet18/weights_1000.pth/epoch_100.pth'  # Runner to load ckpt
