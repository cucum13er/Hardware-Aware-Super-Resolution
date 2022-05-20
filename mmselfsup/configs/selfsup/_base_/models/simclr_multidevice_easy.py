# model settings
model = dict(
    type='SimCLR_Multidevice',
    backbone=dict(
        type='EasyRes',
        in_channels=3,
        ),
    neck=dict(
        type='NonLinearNeck',  # SimCLR non-linear neck
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=False),
    head=dict(type='SNNLossHead', temperature=0.07))
