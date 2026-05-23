_base_ = [
    '../../configs_mmseg/setr/setr_vit-l_pup_8xb2-160k_ade20k-512x512.py'
]
train_dataloader = dict(batch_size=4)

custom_imports = dict(imports=['eval.mysegmentation.backbones.my_vit_kope'], allow_failed_imports=False)

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='MyViTKoPE',
        model_name='vit_kope_base',
        img_size=224,
        patch_size=16,
        out_indices=[2, 5, 8, 11],
        drop_path_rate=0.,
        # KoPE specific parameters aligned with pretraining command
        block_chunks=4,
        block_type='nested',
        layerscale=0.1,
        drop_path_uniform=True,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        kope_gamma=0.05,
        learn_phase_gamma=True,
        coupling_qknorm=True,
        coupling_qknorm_learn=True,
        share_kope_coupling=True,
        update_ext_token_phase=True,
        use_learnable_pos_embed=True,
        coupling_use_phase_bias_rotation=False,
        coupling_use_vo_proj=False,
        use_bias_for_phase_update=False,
        fixed_layout=True,
        kope_vo_rotation=True,
        kope_mix=True,
        kope_mix_init_gain=0.1,
        kope_mix_phase_norm=True,
        base=20,
        phase_mode='per_frequency',
        no_phase_norm=True,
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