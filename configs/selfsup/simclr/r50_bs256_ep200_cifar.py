_base_ = '../../base.py'
# model settings
model = dict( ## totally use SimCLR model
    type='SimCLR',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeckSimCLR', # SimCLR non-linear neck
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.1))
# dataset settings
data_source_cfg = dict(type='Cifar10', root='data') ## classification data_source_cfg
dataset_type = 'ContrastiveDataset' ## SimCLR dataset_type
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ## SimCLR img_norm_cfg
train_pipeline = [ ## SimCLR train_pipeline
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.5),
]
test_pipeline = [ ## classification test_pipeline
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict( ## classification data with val & test
    imgs_per_gpu=128,
    workers_per_gpu=2,
    train=dict( ## SimCLR style training with prefetch
        type=dataset_type,
        data_source=dict(
            split='train', **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [ ## classification custom_hooks
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=10,
        imgs_per_gpu=128,
        workers_per_gpu=8,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
optimizer = dict( ## SimCLR optimizer
    type='LARS',
    lr=0.3,
    weight_decay=0.000001,
    momentum=0.9,
    paramwise_options={
        r'(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True)})
# learning policy
lr_config = dict( ## SimCLR lr_config
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10) ## SimCLR checkpoint_config
# runtime settings
total_epochs = 200 ## SimCLR total_epochs
