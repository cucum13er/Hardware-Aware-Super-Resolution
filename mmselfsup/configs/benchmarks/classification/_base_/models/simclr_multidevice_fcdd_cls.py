# model settings
model = dict(
    type='SimCLR_Multidevice_cls',
    backbone=dict(
        type='FCDD_Rui',
        in_shape=(3,160,160),

        ),

    head=dict(  type='ClsHead_Twolayers', 
                with_avg_pool=True,
                in_channels=512,
                hid_channels=128,
                num_classes=3,
             ),
    
             )
