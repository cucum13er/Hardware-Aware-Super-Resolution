# dataset settings
data_source = 'MultiDevice_ours_val' # for testing
#data_source = 'MultiDevice_ours' # for training
dataset_type = 'MultiDeviceDataset_ours'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# The difference between mocov2 and mocov1 is the transforms in the pipeline
train_pipeline = [
    dict(type='RandomCrop', size=96),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='RandomCrop', size=96),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='ToTensor'),
]
# prefetch
prefetch = False
# if not prefetch:
#     train_pipeline.extend(
#         [dict(type='ToTensor'), # scale to [0,1]
#          #dict(type='Normalize', **img_norm_cfg)
#          ])

# dataset summary
data = dict(
    #imgs_per_gpu=256,  # for training
    imgs_per_gpu=1,  # for testing
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            ######################### changed to tiny-imagenet ################
            # data_prefix='data/ThreeDevices', #########################
            data_prefix='data/Ours/X4/', #########################
            #data_prefix='data/DIV2K_Flickr2K/lq/X4', #########################
            ann_file= None, #######################
            # data_prefix='data/imagenet/train', #########################
            # ann_file='data/imagenet/meta/train.txt', #######################
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            ######################### changed to tiny-imagenet ################
            # data_prefix='data/ThreeDevices', #########################
            #data_prefix='data/MultiDegrade/SupER1/X4/val', #########################
            data_prefix='data/Ours/X4/', #########################
            ann_file= None, #######################            
            # data_prefix='data/imagenet/train', #########################
            # ann_file='data/imagenet/meta/train.txt', #######################
        ),
        num_views=[1],
        pipelines=[test_pipeline],
        prefetch=prefetch,        
        ),
    test=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            ######################### changed to tiny-imagenet ################
            # data_prefix='data/ThreeDevices', #########################
            #data_prefix='data/MultiDegrade/SupER1/X4/test', #########################
            data_prefix='data/Ours/X4/', #########################
            ann_file= None, #######################            
            # data_prefix='data/imagenet/train', #########################
            # ann_file='data/imagenet/meta/train.txt', #######################
        ),
        num_views=[1],
        pipelines=[test_pipeline],
        prefetch=prefetch,
        ),
    
    )


