data_source = 'MultiDevice_ours'
dataset_type = 'MultiDeviceDataset_ours'
name = 'MultiDevice_val'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_pipeline = [
    dict(type='RandomCrop', size=160),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
    # dict(type='ToTensor'),
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=4,
    extract=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/Ours_noNew/X2_test/',
            ann_file=None,
        ),
        num_views=[1],
        pipelines=[test_pipeline],
        )
    )
