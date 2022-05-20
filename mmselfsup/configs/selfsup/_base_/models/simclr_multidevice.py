# model settings
model = dict(
    type='SimCLR_Multidevice',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        #frozen_stages= 4
        ),
    neck=dict(
        type='NonLinearNeck',  # SimCLR non-linear neck
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='SNNLossHead', temperature=0.1))
