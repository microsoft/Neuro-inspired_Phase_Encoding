# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
import logging
import math
from typing import Sequence, Tuple, Union, Callable, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')))
from simdinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, Block, NestedTensorBlock
from simdinov2.layers.attention import Attention, MemEffAttention

logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformerRoPE(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        ffn_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float=0.0,
        drop_path_uniform=False,
        layerscale=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        drop_masks=False,
        gradient_checkpointing=False,
        qknorm=False,
        max_resolution=14,
        base=20,
        use_learnable_pos_embed: bool = False,
        rope_mix: bool = True,
        vo_rotation: bool = True,
        **kwargs
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            layerscale (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
            rope_mix: (bool) enable mixing of rotary embeddings (ablation study)
            vo_rotation: (bool) enable rotary embeddings on value and output (ablation study)
        """
        super().__init__()
        if kwargs:
            print("RoPE ViT Unknown args: ", kwargs)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.use_learnable_pos_embed = use_learnable_pos_embed
        if self.use_learnable_pos_embed:
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.empty(1, num_patches + 1, embed_dim))
            self.raw_pos_embed = None
        else:
            self.pos_embed = None
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.empty(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )
        self.ext_token_num = 1 + num_register_tokens
        self._coords_cache: dict = {}

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        self.block_chunks = block_chunks

        if rope_mix or vo_rotation:
            if isinstance(block_fn, partial) and 'attn_class' in block_fn.keywords:
                base_attn_class = block_fn.keywords['attn_class']
            else:
                base_attn_class = Attention

            attn_class = partial(base_attn_class, rope_mix=rope_mix, vo_rotation=vo_rotation)
        else:
            attn_class = block_fn.keywords.get('attn_class', Attention) if isinstance(block_fn, partial) else Attention

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                ffn_drop=ffn_drop,
                attn_drop=attn_drop,
                layerscale=layerscale,
                qknorm=qknorm,
                max_resolution=max_resolution,
                base=base,
                ext_token_num=1 + num_register_tokens,
                use_2d_rope=True,
                attn_class=attn_class,
                **kwargs
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.drop_masks = drop_masks
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        if self.use_learnable_pos_embed and self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        def init_weights_vit_timm(module: nn.Module, name: str = ""):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        named_apply(init_weights_vit_timm, self)

    def update_img_size(self, img_size, keep_raw=False):
        if img_size == self.patch_embed.img_size:
            return
        if self.use_learnable_pos_embed and self.pos_embed is not None:
            if keep_raw and getattr(self, "raw_pos_embed", None) is not None:
                self.pos_embed = self.raw_pos_embed
            N = self.pos_embed.shape[1] - 1
            M = int(math.sqrt(N))
            dim = self.pos_embed.shape[-1]
            pos_embed = self.pos_embed.float()
            class_pos_embed = pos_embed[:, :1]
            patch_pos_embed = pos_embed[:, 1:]
            if isinstance(img_size, (tuple, list)):
                h_target, w_target = img_size if len(img_size) == 2 else (img_size[0], img_size[0])
            else:
                h_target = w_target = img_size
            if isinstance(self.patch_size, (tuple, list)):
                patch_h, patch_w = self.patch_size if len(self.patch_size) == 2 else (self.patch_size[0], self.patch_size[0])
            else:
                patch_h = patch_w = self.patch_size
            w0 = w_target // patch_w
            h0 = h_target // patch_h
            kwargs = {}
            if self.interpolate_offset:
                sx = float(w0 + self.interpolate_offset) / M
                sy = float(h0 + self.interpolate_offset) / M
                kwargs["scale_factor"] = (sx, sy)
            else:
                kwargs["size"] = (w0, h0)
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=self.interpolate_antialias,
                **kwargs,
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            pos_embed = torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(self.pos_embed.dtype)
            if keep_raw:
                self.raw_pos_embed = self.pos_embed
            self.pos_embed = nn.Parameter(pos_embed)
        self.patch_embed.update_img_size(img_size)
        self._coords_cache.clear()

    def update_patch_size(self, patch_size, keep_raw=False):
        if patch_size == self.patch_size:
            return
        self.patch_embed.update_patch_size(patch_size, keep_raw)
        self.patch_size = patch_size
        self._coords_cache.clear()

    def interpolate_pos_encoding(self, tokens: torch.Tensor, w: int, h: int) -> torch.Tensor:
        if not self.use_learnable_pos_embed or self.pos_embed is None:
            raise RuntimeError("Learnable positional embeddings are disabled for this model")
        previous_dtype = tokens.dtype
        npatch = tokens.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]
        dim = self.pos_embed.shape[-1]
        if isinstance(self.patch_size, (tuple, list)):
            patch_h, patch_w = self.patch_size if len(self.patch_size) == 2 else (self.patch_size[0], self.patch_size[0])
        else:
            patch_h = patch_w = self.patch_size
        w0 = w // patch_w
        h0 = h // patch_h
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)

    def _get_patch_coords(self, height: int, width: int, batch_size: int, device: torch.device) -> torch.Tensor:
        device_index = device.index if device.index is not None else -1
        device_key = (height, width, device.type, device_index)
        if device_key not in self._coords_cache:
            if device.type == "cpu":
                ys = torch.arange(height, dtype=torch.long, device=device)
                xs = torch.arange(width, dtype=torch.long, device=device)
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
                coords = torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2)
                self._coords_cache[device_key] = coords
            else:
                cpu_key = (height, width, "cpu", -1)
                if cpu_key not in self._coords_cache:
                    ys = torch.arange(height, dtype=torch.long)
                    xs = torch.arange(width, dtype=torch.long)
                    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
                    coords_cpu = torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2)
                    self._coords_cache[cpu_key] = coords_cpu
                self._coords_cache[device_key] = self._coords_cache[cpu_key].to(device)
        base_coords = self._coords_cache[device_key]
        return base_coords.unsqueeze(0).repeat(batch_size, 1, 1)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        feat_h = h // self.patch_size
        feat_w = w // self.patch_size
        coords = self._get_patch_coords(feat_h, feat_w, B, x.device)
        keep_mask = None
        if masks is not None:
            if not self.drop_masks:  # ibot style
                mask_token = self.mask_token.to(dtype=x.dtype).unsqueeze(0)
                x = torch.where(masks.unsqueeze(-1), mask_token, x)
            else:
                keep_mask = (~masks)

        tokens = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        if self.use_learnable_pos_embed and self.pos_embed is not None:
            tokens = tokens + self.interpolate_pos_encoding(tokens, w, h)

        if keep_mask is not None:
            dim = tokens.shape[-1]
            tokens = torch.cat(
                (
                    tokens[:, :1],
                    torch.masked_select(
                        tokens[:, 1:],
                        keep_mask.unsqueeze(-1).expand(-1, -1, dim),
                    ).view(B, -1, dim),
                ),
                dim=1,
            )
            coords_mask = keep_mask.unsqueeze(-1).expand(-1, -1, 2)
            coords = torch.masked_select(coords, coords_mask).view(B, -1, 2)
        if self.register_tokens is not None:
            tokens = torch.cat(
                (
                    tokens[:, :1],
                    self.register_tokens.expand(B, -1, -1),
                    tokens[:, 1:],
                ),
                dim=1,
            )

        return tokens, coords

    def _iter_attention_modules(self):
        if self.chunked_blocks:
            for chunk in self.blocks:
                for blk in chunk:
                    if hasattr(blk, "attn"):
                        yield blk.attn
        else:
            for blk in self.blocks:
                if hasattr(blk, "attn"):
                    yield blk.attn

    def _set_rope_metadata(self, coords_batch: Optional[torch.Tensor], coords_flat: torch.Tensor) -> None:
        for attn in self._iter_attention_modules():
            if hasattr(attn, "set_rope_metadata"):
                attn.set_rope_metadata(coords_batch=coords_batch, coords_flat=coords_flat)

    def forward_features_list(self, x_list, masks_list):
        prepared = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        tokens_list = [p[0] for p in prepared]
        coords_list = [p[1] for p in prepared]
        batch_sizes = [coords.shape[0] for coords in coords_list]
        if len(coords_list) == 1:
            coords_batch = coords_list[0]
        elif len(set(batch_sizes)) == 1:
            coords_batch = torch.cat(coords_list, dim=1)
        else:
            coords_batch = None
        if len(set(batch_sizes)) == 1:
            coords_flat = torch.cat(coords_list, dim=1)
        else:
            coords_flat = torch.cat([coords.reshape(1, -1, 2) for coords in coords_list], dim=1)
        self._set_rope_metadata(coords_batch=coords_batch, coords_flat=coords_flat)
        x = tokens_list
        if self.gradient_checkpointing and self.training:
            for blk in self.blocks:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x)

        all_x = x
        output = []
        for tokens, masks in zip(all_x, masks_list):
            x_norm = self.norm(tokens)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": tokens,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        tokens, coords = self.prepare_tokens_with_masks(x, masks)
        coords_flat = coords.reshape(1, -1, 2)
        self._set_rope_metadata(coords_batch=coords, coords_flat=coords_flat)

        if self.gradient_checkpointing and self.training:
            for blk in self.blocks:
                tokens = torch.utils.checkpoint.checkpoint(blk, tokens, use_reentrant=False)
        else:
            for blk in self.blocks:
                tokens = blk(tokens)

        x_norm = self.norm(tokens)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": tokens,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        tokens, coords = self.prepare_tokens_with_masks(x)
        coords_flat = coords.reshape(1, -1, 2)
        self._set_rope_metadata(coords_batch=coords, coords_flat=coords_flat)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i in blocks_to_take:
                output.append(tokens)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        tokens, coords = self.prepare_tokens_with_masks(x)
        coords_flat = coords.reshape(1, -1, 2)
        self._set_rope_metadata(coords_batch=coords, coords_flat=coords_flat)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                tokens = blk(tokens)
                if i in blocks_to_take:
                    output.append(tokens)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    @torch.inference_mode()
    def forward_vis(self, x, vit_feat="k", depth=1):
        feat_out = {}
        def hook_fn_forward_input0(module, input, output, name="output"):
            feat_out[name] = input[0]
        def hook_fn_forward_output(module, input, output, name="output"):
            feat_out[name] = output
        if self.block_chunks > 0:
            module = self.blocks[-1-depth // self.block_chunks][-(depth%self.block_chunks)]
            print(f"module: model.blocks{-1-depth // self.block_chunks}.{-(depth%self.block_chunks)}")
        else:
            module = self.blocks[-depth]
        if vit_feat == "module":
            module.attn.register_forward_hook(partial(hook_fn_forward_output, name="attn"))
            module.mlp.register_forward_hook(partial(hook_fn_forward_output, name="mlp"))
            module.norm1.register_forward_hook(partial(hook_fn_forward_output, name="norm1"))
            module.norm2.register_forward_hook(partial(hook_fn_forward_output, name="norm2"))
            module.norm2.register_forward_hook(partial(hook_fn_forward_input0, name="attnres"))
            module.register_forward_hook(partial(hook_fn_forward_output, name="block"))
            module.attn.attn_drop.register_forward_hook(partial(hook_fn_forward_output, name="attn_map"))
        elif vit_feat == "attn":
            module.attn.register_forward_hook(hook_fn_forward_output)
        elif vit_feat == "attn_map":
            module.attn.attn_drop.register_forward_hook(hook_fn_forward_output)
        elif vit_feat == "mlp":
            module.mlp.register_forward_hook(hook_fn_forward_output)
        elif vit_feat == "norm1":
            module.norm1.register_forward_hook(hook_fn_forward_output)
        elif vit_feat == "norm2":
            module.norm2.register_forward_hook(hook_fn_forward_output)
        else:
            module.attn.qkv.register_forward_hook(hook_fn_forward_output)

        # Forward pass in the model
        bs, _, h, w = x.shape
        feat_h, feat_w = h // self.patch_size, w // self.patch_size
        num_patches = feat_h * feat_w
        self.forward_features(x)
        if vit_feat == "module":
            return feat_out
        elif vit_feat not in ["k", "q", "v", "kqv"]:
            return feat_out["output"]
        # print("feat out shape:", feat_out["qkv"].shape)
        qkv = (
                feat_out["output"][:, self.num_tokens + self.num_register_tokens:]
                .reshape(bs, num_patches, 3, -1)
            )
        q, k, v = qkv.unbind(dim=2) #B, N, C

        # Modality selection
        if vit_feat == "k":
            feats = k
        elif vit_feat == "q":
            feats = q
        elif vit_feat == "v":
            feats = v
        elif vit_feat == "kqv":
            feats = torch.cat([k, q, v], dim=-1)
        return feats.transpose(1, 2)



def vit_rope_small(patch_size=16, num_register_tokens=0, block="nested", **kwargs):
    model = DinoVisionTransformerRoPE(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=get_block_fn(block),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_rope_base(patch_size=16, num_register_tokens=0, block="nested", **kwargs):
    model = DinoVisionTransformerRoPE(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=get_block_fn(block),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_rope_large(patch_size=16, num_register_tokens=0, block="nested", **kwargs):
    model = DinoVisionTransformerRoPE(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=get_block_fn(block),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_rope_so150m(patch_size=16, num_register_tokens=0, block="nested", **kwargs):
    """ SO150M (shape optimized) """
    model = DinoVisionTransformerRoPE(
        patch_size=patch_size,
        embed_dim=896, depth=18, num_heads=14, mlp_ratio=2.572,
        block_fn=get_block_fn(block),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_rope_so150m2(patch_size=16, num_register_tokens=0, block="nested", **kwargs):
    """ SO150M v2 (shape optimized, but diff than paper def, optimized for GPU) """
    model = DinoVisionTransformerRoPE(
        patch_size=patch_size,
        embed_dim=832, depth=21, num_heads=13, mlp_ratio=34/13,
        block_fn=get_block_fn(block),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

def vit_rope_so300m(patch_size=16, num_register_tokens=0, block="nested", **kwargs):
    model = DinoVisionTransformerRoPE(
        patch_size=patch_size,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=3.375,
        block_fn=get_block_fn(block),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

def vit_rope_so400m(patch_size=16, num_register_tokens=0, block="nested", **kwargs):
    model = DinoVisionTransformerRoPE(
        patch_size=patch_size,
        embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362,
        block_fn=get_block_fn(block),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

def aimv2_rope_large_patch14_224(patch_size=16, num_register_tokens=0, block="nested", **kwargs):
    """ ViT Large AIM-v2 model
    """
    model = DinoVisionTransformerRoPE(
        patch_size=14, embed_dim=1024, depth=24, num_heads=8, class_token=False, fc_norm=False,
        mlp_ratio=2.75, global_pool='avg', qkv_bias=False, proj_bias=False, act_layer='silu',
        #norm_layer=partial(RmsNorm, eps=1e-5), embed_norm_layer=partial(RmsNorm, eps=1e-5), mlp_layer=SwiGLUFFNFused,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

def vit_rope_giant2(patch_size=16, num_register_tokens=0, block="nested", **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformerRoPE(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=get_block_fn(block),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

def get_block_fn(block):
    match block:
        case "nested":
            return partial(NestedTensorBlock, attn_class=MemEffAttention)
        case "memeff":
            return partial(Block, attn_class=MemEffAttention)
        case "base":
            return partial(Block, attn_class=Attention)
    raise ValueError(f"Unkown block type: {block}")

if __name__ == "__main__":
    #table for all models with params, embed, depth, heads, mlp_ratio
    models = [x for x in dir() if x.startswith("vit_rope_") or x.startswith("aimv2_rope_")]
    from prettytable import PrettyTable
    table = PrettyTable(["Model", "Params", "Embed", "Depth", "Heads", "MLP Ratio"])
    for key in models:
        model = vars()[key]()
        #print(f"Model: {key}, Params: {sum(p.numel() for p in model.parameters())}, Embed: {model.embed_dim}, Depth: {model.n_blocks}, Heads: {model.num_heads}, MLP Ratio: {model.mlp_ratio}")
        table.add_row([key, sum(p.numel() for p in model.parameters()), model.embed_dim, model.n_blocks, model.num_heads, model.mlp_ratio])
    table._rows.sort(key=lambda x: x[1])
    print(table)