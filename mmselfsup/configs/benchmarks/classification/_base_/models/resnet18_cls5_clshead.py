# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        frozen_stages= 4),
    head=dict(  
        type='ClsHead', 
        with_avg_pool=True,
        in_channels=512,
        num_classes=5,
             ),
        )
        
        
