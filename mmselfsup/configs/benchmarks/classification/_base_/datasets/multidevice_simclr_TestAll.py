# dataset settings
data_source = 'MultiDevice'
dataset_type = 'MultiDeviceDataset'
img_norm_cfg = dict(mean=[0.485, ], std=[0.229,])
# The difference between mocov2 and mocov1 is the transforms in the pipeline
train_pipeline = [
    dict(type='RandomCrop', size=160),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
]
test_pipeline = [
    #dict(type='RandomCrop', size=200),
    dict(type='RandomCrop', size=160),    
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='ToTensor'),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'), # scale to [0,1]
         #dict(type='Normalize', **img_norm_cfg)
         ])

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
            ######################### changed to diff datasets ################
            data_prefix='data/test_contrastive/BSD100/lq', #########################
            #data_prefix='data/test_contrastive/Set5/lq', #########################
            #data_prefix='data/test_contrastive/Set14/lq', #########################            
            #data_prefix='data/test_contrastive/Urban100/lq', #########################   
            #data_prefix='data/test_contrastive/BSD100/lq_aniso', #########################
            #data_prefix='data/test_contrastive/Set5/lq_aniso', #########################
            #data_prefix='data/test_contrastive/Set14/lq_aniso', #########################            
            #data_prefix='data/test_contrastive/Urban100/lq_aniso', #########################                                      
            ann_file= None, #######################            

        ),
        num_views=[4],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ),
    val=dict(
        
        type=dataset_type,
        data_source=dict(
            type=data_source,
            ######################### changed to diff datasets ################
            data_prefix='data/test_contrastive/BSD100/lq', #########################
            #data_prefix='data/test_contrastive/Set5/lq', #########################
            #data_prefix='data/test_contrastive/Set14/lq', #########################            
            #data_prefix='data/test_contrastive/Urban100/lq', #########################   
            #data_prefix='data/test_contrastive/BSD100/lq_aniso', #########################
            #data_prefix='data/test_contrastive/Set5/lq_aniso', #########################
            #data_prefix='data/test_contrastive/Set14/lq_aniso', #########################            
            #data_prefix='data/test_contrastive/Urban100/lq_aniso', #########################                                      
            ann_file= None, #######################            

        ),
        num_views=[1],
        pipelines=[test_pipeline],
        prefetch=prefetch,        
        ),
    test=dict(
           
        type=dataset_type,
        data_source=dict(
            type=data_source,
            ######################### changed to diff datasets ################
            #data_prefix='data/MultiDegrade/Flickr2K/X4/test',
            #data_prefix='data/MultiDegrade/DIV2K/X4/test',            
            #data_prefix='data/MultiDegrade/DIV2K/X4/test_more',            
            data_prefix='data/MultiDegrade/DIV2K/X4/test_aniso',            
            #data_prefix='data/test_contrastive/BSD100/lq', #########################
            #data_prefix='data/test_contrastive/Set5/lq', #########################
            #data_prefix='data/test_contrastive/Set14/lq', #########################            
            #data_prefix='data/test_contrastive/Urban100/lq', #########################   
            #data_prefix='data/test_contrastive/BSD100/lq_aniso', #########################
            #data_prefix='data/test_contrastive/Set5/lq_aniso', #########################
            #data_prefix='data/test_contrastive/Set14/lq_aniso', #########################            
            #data_prefix='data/test_contrastive/Urban100/lq_aniso', #########################                                      
            ann_file= None, #######################            

        ),
        num_views=[1],
        pipelines=[test_pipeline],
        prefetch=prefetch,
        ),
    
    )


