_base_ = [
    '../../configs_mmseg/setr/setr_vit-l_pup_8xb2-160k_ade20k-512x512.py'
]
train_dataloader = dict(batch_size=4)

custom_imports = dict(imports=['eval.mysegmentation.backbones.my_vit'], allow_failed_imports=False)

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='MyViT',
        model_name='vit_base',
        img_size=224,
        patch_size=16,
        out_indices=[2, 5, 8, 11],
        drop_path_rate=0.,
        block='nested',
        block_chunks=4,
        layerscale=0.1,
        drop_path_uniform=True,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        init_cfg=dict(type='Pretrained', checkpoint='path/to/checkpoint.pth')
    ),
    decode_head=dict(
        in_channels=768,
        in_index=3,
        num_classes=150
    ),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=256,
            in_index=0,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=256,
            in_index=1,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=256,
            in_index=2,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ]
)

# Enable find_unused_parameters for DDP
find_unused_parameters = True