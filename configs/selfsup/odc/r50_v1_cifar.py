_base_ = '../../base.py'
# Model settings
num_classes = 10
# num_classes = 10000
model = dict(
    type='ODC',
    pretrained=None,
    with_sobel=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeckV0',
        in_channels=2048,
        hid_channels=512,
        out_channels=256,
        with_avg_pool=True),
    head=dict(
        type='ClsHead',
        with_avg_pool=False,
        in_channels=256,
        num_classes=num_classes),
    memory_bank=dict(
        type='ODCMemory',
        length=1281167,
        feat_dim=256,
        momentum=0.5,
        num_classes=num_classes,
        min_cluster=20,
        debug=False))
# Dataset settings
data_source_cfg = dict(type='Cifar10', root='data')
# data_source_cfg = dict(
#     type='ImageNet',
#     memcached=True,
#     mclient_path='/mnt/lustre/share/memcached_client')
# data_train_list = 'data/imagenet/meta/train.txt'
# data_train_root = 'data/imagenet/train'
dataset_type = 'ClassificationDataset'
# dataset_type = 'DeepClusterDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
# img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
# train_pipeline = [
#     dict(type='RandomResizedCrop', size=224),
#     dict(type='RandomHorizontalFlip'),
#     dict(type='RandomRotation', degrees=2),
#     dict(
#         type='ColorJitter',
#         brightness=0.4,
#         contrast=0.4,
#         saturation=1.0,
#         hue=0.5),
#     dict(type='RandomGrayscale', p=0.2),
#     dict(type='ToTensor'),
#     dict(type='Normalize', **img_norm_cfg),
# ]
# extract_pipeline = [
#     dict(type='Resize', size=256),
#     dict(type='CenterCrop', size=224),
#     dict(type='ToTensor'),
#     dict(type='Normalize', **img_norm_cfg),
# ]

test_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline))
# data = dict(
#     imgs_per_gpu=64,  # 64*8
#     sampling_replace=True,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         data_source=dict(
#             list_file=data_train_list, root=data_train_root,
#             **data_source_cfg),
#         pipeline=train_pipeline))
# Additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=10,
        imgs_per_gpu=128,
        workers_per_gpu=8,
        eval_param=dict(topk=(1, 5)))
]
# custom_hooks = [
#     dict(
#         type='DeepClusterHook',
#         extractor=dict(
#             imgs_per_gpu=128,
#             workers_per_gpu=8,
#             dataset=dict(
#                 type=dataset_type,
#                 data_source=dict(
#                     list_file=data_train_list,
#                     root=data_train_root,
#                     **data_source_cfg),
#                 pipeline=extract_pipeline)),
#         clustering=dict(type='Kmeans', k=num_classes, pca_dim=-1),  # no pca
#         unif_sampling=False,
#         reweight=True,
#         reweight_pow=0.5,
#         init_memory=True,
#         initial=True,  # call initially
#         interval=9999999999),  # initial only
#     dict(
#         type='ODCHook',
#         centroids_update_interval=10,  # iter
#         deal_with_small_clusters_interval=1,
#         evaluate_interval=50,
#         reweight=True,
#         reweight_pow=0.5)
# ]
# Optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
# optimizer = dict(
#     type='SGD', lr=0.06, momentum=0.9, weight_decay=0.00001,
#     nesterov=False,
#     paramwise_options={'\Ahead.': dict(momentum=0.)})
# Learning policy
lr_config = dict(policy='step', step=[150, 250])
# lr_config = dict(policy='step', step=[400], gamma=0.4)
checkpoint_config = dict(interval=50)
# checkpoint_config = dict(interval=10)
# Runtime settings
total_epochs = 350
# total_epochs = 480
