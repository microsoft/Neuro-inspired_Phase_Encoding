# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits
from . import vision_transformer_rope as rope_vits
from . import vision_transformer_kope as kope_vits


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224, patch_size=16):
    args.arch = args.arch.removesuffix("_memeff")
    if args.arch in rope_vits.__dict__:
        vit_kwargs = dict(img_size=img_size, **args)
        vit_kwargs["patch_size"] = patch_size
        for i in ["arch", "gradient_checkpointing",
                  "drop_path_rate", "attn_drop", "ffn_drop",
                  'pretrained_weights', 'pretrained_patch_size', 'pretrained_img_size', 'freeze_backbone_epochs']:
            vit_kwargs.pop(i, None)
        teacher = rope_vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = rope_vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            attn_drop=args.attn_drop,
            ffn_drop=args.ffn_drop,
            gradient_checkpointing=args.gradient_checkpointing
        )
        embed_dim = student.embed_dim
        logger.info(f"Model {student.__class__.__name__} {img_size}p{patch_size} built. Total params: {sum(p.numel() for p in student.parameters())}")
    elif args.arch in kope_vits.__dict__:
        vit_kwargs = dict(img_size=img_size, **args)
        vit_kwargs["patch_size"] = patch_size
        args_dict = getattr(args, "__dict__", {})
        for key in args_dict:
            if key.startswith("kope_") and args_dict[key] is not None:
                vit_kwargs[key] = args_dict[key]
        for i in ["arch", "gradient_checkpointing",
                  "drop_path_rate", "attn_drop", "ffn_drop",
                  'pretrained_weights', 'pretrained_patch_size', 'pretrained_img_size', 'freeze_backbone_epochs']:
            vit_kwargs.pop(i, None)
        teacher = kope_vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = kope_vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            attn_drop=args.attn_drop,
            ffn_drop=args.ffn_drop,
            gradient_checkpointing=args.gradient_checkpointing
        )
        embed_dim = student.embed_dim
        logger.info(f"Model {student.__class__.__name__} {img_size}p{patch_size} built. Total params: {sum(p.numel() for p in student.parameters())}")
    elif args.arch in vits.__dict__ or "vit" in args.arch:
        vit_kwargs = dict(img_size=img_size, **args)
        vit_kwargs["patch_size"] = patch_size
        for i in ["arch", "gradient_checkpointing",
                  "drop_path_rate", "attn_drop", "ffn_drop",
                  'pretrained_weights', 'pretrained_patch_size', 'pretrained_img_size', 'freeze_backbone_epochs']:
            vit_kwargs.pop(i, None)
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            attn_drop=args.attn_drop,
            ffn_drop=args.ffn_drop,
            gradient_checkpointing=args.gradient_checkpointing
        )
        embed_dim = student.embed_dim
        logger.info(f"Model {student.__class__.__name__} {img_size}p{patch_size} built. Total params: {sum(p.numel() for p in student.parameters())}")
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    if cfg.student.pretrained_weights:
        return build_model(cfg.student, only_teacher, cfg.student.pretrained_img_size, cfg.student.pretrained_patch_size)
    return build_model(cfg.student, only_teacher, cfg.crops.global_crops_size, cfg.student.patch_size)
