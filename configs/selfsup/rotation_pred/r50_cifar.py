_base_ = '../../base.py'
# model settings
model = dict(
    type='RotationPred',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048, num_classes=4))
# dataset settings
data_source_cfg = dict(type='Cifar10', root='data') ## classification data_source_cfg
dataset_type = 'RotationPredDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=16,  # (16*4) x 8 = 512
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            split='train',
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            split='test',
            **data_source_cfg),
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001, nesterov=False)
# learning policy
lr_config = dict(
    policy='step',
    step=[30, 50],
    warmup='linear',
    warmup_iters=5,  # 5 ep
    warmup_ratio=0.1,
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 70
