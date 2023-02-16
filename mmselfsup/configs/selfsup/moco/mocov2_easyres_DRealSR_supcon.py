_base_ = [
    '../_base_/models/mocov2_multidevice_easy.py',
    '../_base_/datasets/multidevice_mocov2_DRealSR.py',
    '../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
log_config = dict(
    interval=200,
    interval_exp_name=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
runner = dict(type='EpochBasedRunner', max_epochs=5) ####################
checkpoint_config = dict(interval=1, max_keep_ckpts=50)
