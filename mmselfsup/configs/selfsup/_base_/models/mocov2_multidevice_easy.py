# model settings
model = dict(
    type='MoCo_label',
    queue_len=8192,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='EasyRes',
        in_channels=3,
        init_cfg=None,
        ),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='SNNLossHead', temperature=0.07))
#    head=dict(type='ContrastiveHead', temperature=0.07))
