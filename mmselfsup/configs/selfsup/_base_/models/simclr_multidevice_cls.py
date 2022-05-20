# model settings
model = dict(
    type='SimCLR_Multidevice_cls',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),

    head=dict(  type='ClsHead_Twolayers', 
                with_avg_pool=True,
                in_channels=512,
                hid_channels=128,
                num_classes=5,
             ),
    
             )
