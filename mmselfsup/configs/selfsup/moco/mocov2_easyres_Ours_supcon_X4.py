_base_ = [
    '../_base_/models/mocov2_multidevice_easy.py',
    '../_base_/datasets/multidevice_mocov2_OursX4.py',
    '../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
runner = dict(type='EpochBasedRunner', max_epochs=1000) ####################
checkpoint_config = dict(interval=200, max_keep_ckpts=5)
