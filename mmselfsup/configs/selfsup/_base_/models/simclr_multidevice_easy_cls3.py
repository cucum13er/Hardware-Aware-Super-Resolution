# model settings
model = dict(
    type='SimCLR_Multidevice_cls',
    backbone=dict(
        type='EasyRes',
        in_channels=3,
        ),

    head=dict(  type='ClsHead_Twolayers', 
                with_avg_pool=True,
                in_channels=512,
                hid_channels=128,
                num_classes=3,
             ),
    
             )
