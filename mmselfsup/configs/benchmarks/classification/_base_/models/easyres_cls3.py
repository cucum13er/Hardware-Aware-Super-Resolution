# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='EasyRes',
        in_channels=3,
        frozen_stages= 1),
    head=dict(  
        type='ClsHead_Twolayers', 
        with_avg_pool=True,
        in_channels=512,
        hid_channels=128,
        num_classes=3,
             ),
        )
        
        
