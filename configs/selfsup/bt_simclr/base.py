_base_ = "../../base.py"
# model settings
model = dict(
    type="BtSimClr",
    pretrained=None,
    backbone=dict(
        type="ResNet",
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type="SyncBN"),
    ),
    neck=dict(type="AvgPoolNeck"),
    head=dict(type="BtSimClrHead"),
)
# dataset settings
data_source_cfg = dict(type="Cifar10", root="data")
dataset_type = "ContrastiveDataset"
img_norm_cfg = dict(
    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201]
)  # distribution of cifar10
train_pipeline = [
    dict(type="RandomResizedCrop", size=224),
    dict(type="RandomHorizontalFlip"),
    dict(
        type="RandomAppliedTrans",
        transforms=[
            dict(
                type="ColorJitter",
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2,
            )
        ],
        p=0.8,
    ),
    dict(type="RandomGrayscale", p=0.2),
    dict(
        type="RandomAppliedTrans",
        transforms=[dict(type="GaussianBlur", sigma_min=0.1, sigma_max=2.0)],
        p=0.5,
    ),
]

# prefetch
prefetch = True  # speeding up IO, disabled by default
if not prefetch:
    train_pipeline.extend(
        [dict(type="ToTensor"), dict(type="Normalize", **img_norm_cfg)]
    )

data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(split="train", **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
)
# optimizer
optimizer = dict(
    type="SGD",  # use SGD if the batch size is small
    lr=0.1,  # new_lr = old_lr * new_ngpus / 8
    weight_decay=1e-6,
    momentum=0.9,
    paramwise_options={
        r"(bn|gn)(\d+)?.(weight|bias)": dict(weight_decay=0.0, lars_exclude=True),
        "bias": dict(weight_decay=0.0, lars_exclude=True),
    },
)
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=0.0,
    warmup="linear",
    warmup_iters=10,
    warmup_ratio=0.0001,
    warmup_by_epoch=True,
)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
