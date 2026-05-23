import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS as MMSEG_MODELS
from mmdet.registry import MODELS as MMDET_MODELS

from simdinov2.models.vision_transformer import DinoVisionTransformer, vit_base, vit_large, vit_small, vit_giant2

def register_model(cls):
    MMSEG_MODELS.register_module(module=cls)
    MMDET_MODELS.register_module(module=cls)
    return cls

@register_model
class MyViT(BaseModule):
    def __init__(self,
                 model_name='vit_base',
                 img_size=518,
                 patch_size=14,
                 init_cfg=None,
                 out_indices=[2, 5, 8, 11],
                 drop_path_rate=0.1,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.model_name = model_name
        self.out_indices = out_indices

        # Build the model using simdinov2 factory functions
        if model_name == 'vit_base':
            self.model = vit_base(img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate, **kwargs)
        elif model_name == 'vit_large':
            self.model = vit_large(img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate, **kwargs)
        elif model_name == 'vit_small':
            self.model = vit_small(img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate, **kwargs)
        elif model_name == 'vit_giant2':
            self.model = vit_giant2(img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate, **kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.patch_size = patch_size

    def init_weights(self):
        if self.init_cfg is None:
            return

        if self.init_cfg.get('type') == 'Pretrained':
            checkpoint_path = self.init_cfg.get('checkpoint')
            print(f"Loading MyViT checkpoint from {checkpoint_path}")

            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Handle SimDINO/DINOv2 checkpoint structure
            # User might provide a full checkpoint (with 'teacher', 'student', etc.)
            # or a specific extracted checkpoint (just the state_dict)
            if isinstance(checkpoint, dict):
                if 'teacher' in checkpoint and isinstance(checkpoint['teacher'], dict):
                    print("Found 'teacher' key in checkpoint, loading teacher weights.")
                    state_dict = checkpoint['teacher']
                elif 'student' in checkpoint and isinstance(checkpoint['student'], dict):
                    print("Found 'student' key in checkpoint, loading student weights.")
                    state_dict = checkpoint['student']
                elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                    print("Found 'model' key in checkpoint, loading model weights.")
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                    print("Found 'state_dict' key in checkpoint, loading state_dict.")
                    state_dict = checkpoint['state_dict']
                else:
                    print("No standard wrapper keys found (teacher/student/model/state_dict). Assuming checkpoint is the state_dict itself.")
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Remove prefix if necessary (e.g. "backbone.", "module.")
            new_state_dict = {}
            for k, v in state_dict.items():
                # Handle DDP/FSDP prefixes
                if k.startswith('module.'):
                    k = k[7:]

                # Handle backbone prefixes (common in linear probing/finetuning checkpoints)
                if k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v

            # Load into the model
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded weights with msg: {msg}")
        else:
            super().init_weights()

    def forward(self, x):
        outs = self.model.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            return_class_token=False,
            norm=True
        )
        return outs
