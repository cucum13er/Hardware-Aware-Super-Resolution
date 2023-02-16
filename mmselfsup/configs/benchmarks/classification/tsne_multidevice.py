data_source = 'MultiDevice'
dataset_type = 'MultiDeviceDataset'
name = 'MultiDevice_val'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_pipeline = [
    dict(type='RandomCrop', size=160),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='ToTensor'),
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=4,
    extract=dict(
        type='MultiDeviceDataset',
        data_source=dict(
            type=data_source,
            #data_prefix='data/MultiDegrade/DIV2K/X4/test',
            data_prefix='/media/rui/Samsung4TB/DRealSR/Test_x4/test_LR',
            ann_file=None,
        ),
        num_views=[1],
        pipelines=[test_pipeline],
        )
    )
