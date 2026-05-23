import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS as MMSEG_MODELS
from mmdet.registry import MODELS as MMDET_MODELS

def register_neck(cls):
    MMSEG_MODELS.register_module(module=cls)
    MMDET_MODELS.register_module(module=cls)
    return cls

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

@register_neck
class MyFeature2Pyramid(BaseModule):
    """Feature2Pyramid Neck
    Args:
        embed_dim (int): The embedding dimension of the input features.
        rescales (list[float]): The rescale factors for the output features.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """
    def __init__(self,
                 embed_dim,
                 out_channels=None,
                 rescales=[4, 2, 1, 0.5],
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.rescales = rescales
        self.stages = nn.ModuleList()

        # Default to embed_dim if out_channels is not specified
        out_channels = out_channels or embed_dim

        for scale in rescales:
            layers = []
            if scale == 4:
                layers.extend([
                    nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
                    LayerNorm2d(embed_dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
                ])
                out_dim = embed_dim // 4
            elif scale == 2:
                layers.append(nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2))
                out_dim = embed_dim // 2
            elif scale == 1:
                layers.append(nn.Identity())
                out_dim = embed_dim
            elif scale == 0.5:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = embed_dim
            elif scale == 0.25:
                layers.append(nn.MaxPool2d(kernel_size=4, stride=4))
                out_dim = embed_dim

            # ViTDet style: 1x1 Conv -> LN -> 3x3 Conv -> LN
            layers.extend([
                nn.Conv2d(out_dim, out_channels, kernel_size=1, bias=False),
                LayerNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels)
            ])

            self.stages.append(nn.Sequential(*layers))

    def forward(self, inputs):
        # inputs is a tuple/list of tensors from backbone
        # We only use the last feature map from the backbone (ViTDet style)
        x = inputs[-1]

        # Ensure input is (B, C, H, W)
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(N**0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        outs = []
        for stage in self.stages:
            outs.append(stage(x))

        return tuple(outs)

@register_neck
class MyKoPEFeature2Pyramid(MyFeature2Pyramid):
    """Feature2Pyramid Neck with KoPE Phase reduction
    Args:
        embed_dim (int): The embedding dimension of the input features (before phase concatenation).
    """
    def __init__(self, embed_dim, **kwargs):
        super().__init__(embed_dim, **kwargs)
        # Input channels will be 2 * embed_dim (token + phase_cos + phase_sin)
        # token is embed_dim
        # cos is embed_dim // 2
        # sin is embed_dim // 2
        # So total channels is 2 * embed_dim

        self.dim_reduction = nn.Conv2d(2 * embed_dim, embed_dim, kernel_size=1)

    def forward(self, inputs):
        x = inputs[-1]
        # x shape is (B, 2*embed_dim, H, W) because MyViTKoPE sets reshape=True

        # Apply dimensionality reduction
        x = self.dim_reduction(x)

        # Pass to parent forward as a tuple
        return super().forward((x,))
