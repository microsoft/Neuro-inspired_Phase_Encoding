_base_ = [
    '../../configs_mmdet/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic.py'
]
train_dataloader = dict(batch_size=4)

custom_imports = dict(imports=['eval.mysegmentation.backbones.my_vit_kope', 'eval.mysegmentation.backbones.feature_pyramid'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='MyViTKoPE',
        model_name='vit_kope_base',
        img_size=224,
        patch_size=16,
        out_indices=[11],
        drop_path_rate=0.1,
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
        checkpoint_ffn_func=False,
        init_cfg=dict(type='Pretrained', checkpoint='path/to/checkpoint.pth')
    ),
    neck=dict(
        type='MyFeature2Pyramid',
        embed_dim=768,
        rescales=[4, 2, 1, 0.5],
    ),
    panoptic_head=dict(
        in_channels=[768, 768, 768, 768]
    )
)

# Optimizer config with custom_keys for ViT
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.pos_embed': dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.cls_token': dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.mask_token': dict(lr_mult=0.1, decay_mult=0.0),
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

# Enable find_unused_parameters for DDP
find_unused_parameters = True
