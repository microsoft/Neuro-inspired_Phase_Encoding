_base_ = [
    '../../configs_mmdet/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic.py'
]
train_dataloader = dict(batch_size=4)

custom_imports = dict(imports=['eval.mysegmentation.backbones.my_vit', 'eval.mysegmentation.backbones.feature_pyramid'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='MyViT',
        model_name='vit_base',
        img_size=224, # Aligned with pretraining
        patch_size=16, # Aligned with pretraining
        out_indices=[11],
        drop_path_rate=0.1, # Aligned with pretraining: student.drop_path_rate=0.1
        block='nested',
        block_chunks=4,
        layerscale=0.1,
        drop_path_uniform=True,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        gradient_checkpointing=True, # new
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
