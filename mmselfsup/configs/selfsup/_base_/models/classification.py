# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=1,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),

    head=dict(type='ContrastiveHead', temperature=0.1))
