_base_ = '../../base.py'
# Model settings
model = dict(
    type='MOCO',
    pretrained=None,
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='LinearNeck',
        in_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.07))
# Dataset settings
data_source_cfg = dict(type='Cifar10', root='data')
# data_source_cfg = dict(
#     type='ImageNet',
#     memcached=True,
#     mclient_path='/mnt/lustre/share/memcached_client')
# data_train_list = 'data/imagenet/meta/train.txt'
# data_train_root = 'data/imagenet/train'
dataset_type = 'ClassificationDataset'
# dataset_type = 'ContrastiveDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
# img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
# train_pipeline = [
#     dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
#     dict(type='RandomGrayscale', p=0.2),
#     dict(
#         type='ColorJitter',
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.4,
#         hue=0.4),
#     dict(type='RandomHorizontalFlip'),
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
#     imgs_per_gpu=32,  # total 32*8=256
#     workers_per_gpu=4,
#     drop_last=True,
#     train=dict(
#         type=dataset_type,
#         data_source=dict(
#             list_file=data_train_list, root=data_train_root,
#             **data_source_cfg),
#         pipeline=train_pipeline))
# Optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
# optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
# Learning policy
lr_config = dict(policy='step', step=[150, 250])
# lr_config = dict(policy='step', step=[120, 160])
checkpoint_config = dict(interval=50)
# checkpoint_config = dict(interval=10)
# Runtime settings
total_epochs = 350
# total_epochs = 200
