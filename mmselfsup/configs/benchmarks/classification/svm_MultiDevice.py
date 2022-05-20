data_source = 'MultiDevice'
dataset_type = 'MultiDeviceDataset'
split_at = [51]
split_name = ['multiD_trainval', 'multiD_test']
img_norm_cfg = dict(mean=[0.485,], std=[0.229,])

data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=4,
    extract=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/DIV2K_Flickr2K/lq/X4',
            
        ),
        pipeline=[
            dict(type='RandomCrop', size=160),
            dict(type='ToTensor'),
            
        ]))
