# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Minimal PyTorch reference implementation of ViT-KoPE.

This is a dependency-light, single-file reference implementation of the model,
intended as a readable spec of the KoPE algorithm.

Compared with the SimDINOv2 version, this file:

* Uses ``torch.nn.functional.scaled_dot_product_attention`` (SDPA) for all
  attention; there are no xFormers / fused-kernel / nested-tensor paths.
* Drops gradient checkpointing, stochastic depth, multi-crop batching, FSDP
  block chunking, LayerScale, and mask tokens.
* Keeps the parameterization that matters for KoPE: 2D-RoPE phase init,
  per-head phase rotation on Q / K / V (paired form), per-head phase mixing
  matrix, a shared Kuramoto-style coupling module, and the projected,
  re-normalized phase update.

Defaults match the SSL recipe in the paper:
``kope_gamma=0.05``, ``base=20``, ``kope_mix=True``,
``kope_mix_phase_norm=True``, ``kope_vo_rotation=True``,
``coupling_qknorm=True``, ``coupling_qknorm_learn=True``, and a single
``shared_phase_coupling`` used by every block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# -----------------------------------------------------------------------------
# Generic building blocks
# -----------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Conv2d patch embedding (B, 3, H, W) -> (B, N, C)."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                     # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)     # [B, N, C]
        return x


class Mlp(nn.Module):
    """Standard transformer FFN with GELU."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


# -----------------------------------------------------------------------------
# 2D-RoPE phase initialization and paired-form rotary application
# -----------------------------------------------------------------------------


def _repeat_trig(angles: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(angles, 2, dim=-1)


def build_2d_rope_pair(coords: torch.Tensor,
                       inv_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) of paired form ``[B, P, D//2]`` from integer ``(y, x)``
    patch coordinates ``[B, P, 2]`` and ``inv_freq`` of shape ``[D//4]``.

    The first half of the ``D//2`` channels encode row, the second half encode
    column. This mirrors ``build_2d_rope_from_coords`` in the full model.
    """
    coords_fp32 = coords.to(dtype=torch.float32)
    inv_freq_fp32 = inv_freq.to(dtype=torch.float32)
    row_angles = _repeat_trig(coords_fp32[..., 0].unsqueeze(-1) * inv_freq_fp32)  # [B, P, D//2]
    col_angles = _repeat_trig(coords_fp32[..., 1].unsqueeze(-1) * inv_freq_fp32)
    cos_full = torch.cat((torch.cos(row_angles), torch.cos(col_angles)), dim=-1)  # [B, P, D]
    sin_full = torch.cat((torch.sin(row_angles), torch.sin(col_angles)), dim=-1)
    # Collapse adjacent pairs (each rotary pair shares the same cos/sin).
    cos = cos_full.view(*cos_full.shape[:-1], -1, 2)[..., 0]
    sin = sin_full.view(*sin_full.shape[:-1], -1, 2)[..., 0]
    return cos, sin


def apply_rotary_pairs(x: torch.Tensor, cos: torch.Tensor,
                       sin: torch.Tensor) -> torch.Tensor:
    """Apply paired-form rotary to the last dim ``D`` of ``x``.

    ``x``    : ``[..., D]``  (in practice ``[B, H, N, D]``)
    ``cos`` / ``sin`` : broadcastable to ``[..., D//2]``.

    Treats the last axis as ``D//2`` complex pairs ``(real, imag)`` and applies
    a rotation by the angle whose cos/sin is supplied per pair.
    """
    d = x.size(-1)
    half = d // 2
    assert cos.shape[-1] == half, "cos/sin must have last dim D//2"
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    x_view = x.reshape(*x.shape[:-1], half, 2)
    real, imag = x_view[..., 0], x_view[..., 1]
    new_real = real * cos - imag * sin
    new_imag = real * sin + imag * cos
    return torch.stack((new_real, new_imag), dim=-1).view_as(x)


# -----------------------------------------------------------------------------
# State container: tokens plus per-head phase carried through every block
# -----------------------------------------------------------------------------


@dataclass
class KoPEState:
    tokens: torch.Tensor        # [B, N, C]
    phase_cos: torch.Tensor     # [B, H, N, D//2]
    phase_sin: torch.Tensor     # [B, H, N, D//2]


# -----------------------------------------------------------------------------
# KoPE attention: standard QKV attention with phase rotation on Q / K / V
# -----------------------------------------------------------------------------


class KoPEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 proj_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, kope_vo_rotation: bool = True,
                 kope_mix: bool = True, kope_mix_init_gain: float = 0.1,
                 kope_mix_phase_norm: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 4 == 0, "KoPE requires head_dim divisible by 4"
        self.kope_vo_rotation = kope_vo_rotation
        self.kope_mix = kope_mix
        self.kope_mix_phase_norm = kope_mix_phase_norm
        self.attn_drop_p = attn_drop

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.kope_mix:
            rotary_dim = self.head_dim // 2
            self.kope_mix_coef = nn.Parameter(
                torch.empty(num_heads, rotary_dim, rotary_dim)
            )
            with torch.no_grad():
                nn.init.xavier_normal_(self.kope_mix_coef, gain=kope_mix_init_gain)
                eye = torch.eye(rotary_dim, dtype=self.kope_mix_coef.dtype)
                self.kope_mix_coef.add_(eye.unsqueeze(0))
        else:
            self.register_parameter("kope_mix_coef", None)

    def _mix_phase(self, cos: torch.Tensor,
                   sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head linear mixing of the (cos, sin) pair followed by a paired
        renormalization (the ``kope_mix_phase_norm=True`` variant).

        ``cos``/``sin``: ``[B, H, N, D//2]``.
        """
        mix = self.kope_mix_coef.to(dtype=cos.dtype)              # [H, D//2, D//2]
        # einsum is written in the [B, N, H, D//2] layout of the original;
        # we permute in/out so heads stay in front for SDPA.
        cos_bn = cos.permute(0, 2, 1, 3)                          # [B, N, H, D//2]
        sin_bn = sin.permute(0, 2, 1, 3)
        if self.kope_mix_phase_norm:
            cos_bn = torch.einsum("bnhc,hcd->bnhd", cos_bn, mix)
            sin_bn = torch.einsum("bnhc,hcd->bnhd", sin_bn, mix)
            inv = (cos_bn.square() + sin_bn.square()).clamp_min(1e-5).rsqrt()
            cos_bn = cos_bn * inv
            sin_bn = sin_bn * inv
        else:
            mix_n = F.normalize(mix, p=2, dim=-2, eps=1e-6)
            cos_bn = torch.einsum("bnhc,hcd->bnhd", cos_bn, mix_n)
            sin_bn = torch.einsum("bnhc,hcd->bnhd", sin_bn, mix_n)
        return cos_bn.permute(0, 2, 1, 3), sin_bn.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, phase_cos: torch.Tensor,
                phase_sin: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # qkv: [B, N, 3*C] -> [B, N, 3, H, D] -> [3, B, H, N, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = phase_cos, phase_sin
        if self.kope_mix:
            cos, sin = self._mix_phase(cos, sin)

        q = apply_rotary_pairs(q, cos, sin)
        k = apply_rotary_pairs(k, cos, sin)
        if self.kope_vo_rotation:
            v = apply_rotary_pairs(v, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )  # [B, H, N, D]

        if self.kope_vo_rotation:
            # Undo the V rotation in the output space (inverse rotation: sin -> -sin)
            out = apply_rotary_pairs(out, cos, -sin)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -----------------------------------------------------------------------------
# KoPE coupling: shared module that drives one Kuramoto-style phase update step
# -----------------------------------------------------------------------------


class KoPECoupling(nn.Module):
    """Attention-style module that consumes the current tokens and the current
    phase ``(cos, sin)``, and returns the updated phase.

    Q and K are produced from the tokens (with optional learnable per-channel
    rescaling after RMSNorm). The value tensor is the current phase itself,
    written as ``concat(cos, sin)`` along the head-dim. The attention output
    is interpreted as a phase update direction, projected onto the orthogonal
    complement of the current phase, added to the phase with a learnable-or-
    fixed step size ``gamma``, and renormalized so that ``cos^2 + sin^2 = 1``
    pair-wise (the Kuramoto step on the unit circle, generalized to ``D//2``
    independent oscillators per head).
    """

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 attn_drop: float = 0.0, coupling_qknorm: bool = True,
                 coupling_qknorm_learn: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0
        self.coupling_qknorm = coupling_qknorm
        self.attn_drop_p = attn_drop

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        if self.coupling_qknorm and coupling_qknorm_learn:
            # Per-head, per-channel learnable scale applied to Q after RMSNorm.
            self.qk_scale = nn.Parameter(torch.ones(1, self.num_heads, 1, self.head_dim))
        else:
            self.register_parameter("qk_scale", None)

    def forward(self, tokens: torch.Tensor, phase_cos: torch.Tensor,
                phase_sin: torch.Tensor,
                gamma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = tokens.shape
        H, D = self.num_heads, self.head_dim
        D2 = D // 2

        q = self.q_proj(tokens).reshape(B, N, H, D).permute(0, 2, 1, 3)  # [B, H, N, D]
        k = self.k_proj(tokens).reshape(B, N, H, D).permute(0, 2, 1, 3)

        if self.coupling_qknorm:
            q = rmsnorm(q)
            k = rmsnorm(k)
            if self.qk_scale is not None:
                q = q * self.qk_scale.to(dtype=q.dtype)

        # V is the current phase, laid out as [cos | sin] along head-dim.
        v = torch.cat((phase_cos, phase_sin), dim=-1).to(dtype=q.dtype)  # [B, H, N, D]

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )  # [B, H, N, D]

        # Reinterpret the head-dim as (pair-index, oscillator-index): the first
        # D//2 channels are the "cos update" and the next D//2 the "sin update".
        delta = out.reshape(B, H, N, 2, D2)
        phase_pair = torch.stack((phase_cos, phase_sin), dim=3)  # [B, H, N, 2, D//2]

        # Kuramoto step:
        #   1) project out the component of delta that lies along the current
        #      phase direction (preserves the unit-norm constraint to 1st order),
        #   2) take a step of size gamma along the tangential component,
        #   3) renormalize the (cos, sin) pair back onto the unit circle.
        alignment = (delta * phase_pair).sum(dim=3, keepdim=True)
        delta = delta - alignment * phase_pair
        updated = F.normalize(phase_pair + gamma * delta, dim=3, eps=1e-5)
        return updated[..., 0, :], updated[..., 1, :]


# -----------------------------------------------------------------------------
# Transformer block: attention + MLP residuals, then phase update
# -----------------------------------------------------------------------------


class KoPEBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, proj_bias: bool = True,
                 attn_drop: float = 0.0, ffn_drop: float = 0.0,
                 phase_coupling: KoPECoupling = None,
                 phase_gamma: float = 0.05, kope_vo_rotation: bool = True,
                 kope_mix: bool = True, kope_mix_init_gain: float = 0.1,
                 kope_mix_phase_norm: bool = True) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = KoPEAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=ffn_drop,
            kope_vo_rotation=kope_vo_rotation,
            kope_mix=kope_mix, kope_mix_init_gain=kope_mix_init_gain,
            kope_mix_phase_norm=kope_mix_phase_norm,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       drop=ffn_drop)
        # The coupling module is owned by the root model (shared across blocks);
        # store as a non-submodule reference to avoid duplicate parameters.
        object.__setattr__(self, "_phase_coupling", phase_coupling)
        self.register_buffer("phase_gamma",
                             torch.tensor(float(phase_gamma)), persistent=False)
        # paper default: no_phase_norm=True -> identity here
        self.phase_norm = nn.Identity()

    def forward(self, state: KoPEState) -> KoPEState:
        x, cos, sin = state.tokens, state.phase_cos, state.phase_sin
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        gamma = self.phase_gamma.to(dtype=cos.dtype)
        new_cos, new_sin = self._phase_coupling(self.phase_norm(x), cos, sin, gamma)
        return KoPEState(x, new_cos, new_sin)


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------


class DinoVisionTransformerKoPE(nn.Module):
    """Minimal SimDINOv2-style KoPE ViT.

    Forward returns a dict with:
      ``x_norm_clstoken``   : [B, C]
      ``x_norm_regtokens``  : [B, R, C]   (empty if num_register_tokens == 0)
      ``x_norm_patchtokens``: [B, P, C]
      ``kope_patch_phase_cos`` / ``kope_patch_phase_sin``: [B, H, P, D//2]
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 proj_bias: bool = True,
                 ffn_drop: float = 0.0,
                 attn_drop: float = 0.0,
                 num_register_tokens: int = 0,
                 use_learnable_pos_embed: bool = True,
                 base: int = 20,
                 kope_gamma: float = 0.05,
                 kope_vo_rotation: bool = True,
                 kope_mix: bool = True,
                 kope_mix_init_gain: float = 0.1,
                 kope_mix_phase_norm: bool = True,
                 coupling_qknorm: bool = True,
                 coupling_qknorm_learn: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.empty(1, num_register_tokens, embed_dim)
            )
        else:
            self.register_tokens = None
        if use_learnable_pos_embed:
            # pos_embed covers only [CLS | patches] (matches simdinov2).
            self.pos_embed = nn.Parameter(
                torch.empty(1, self.patch_embed.num_patches + 1, embed_dim)
            )
        else:
            self.pos_embed = None

        head_dim = embed_dim // num_heads
        assert head_dim % 4 == 0, "KoPE requires head_dim divisible by 4"
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32)
                     / (head_dim // 2))
        )
        self.register_buffer("_phase_inv_freq", inv_freq, persistent=False)

        # One coupling module, shared by every block (paper default).
        self.shared_phase_coupling = KoPECoupling(
            dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            coupling_qknorm=coupling_qknorm,
            coupling_qknorm_learn=coupling_qknorm_learn,
        )

        self.blocks = nn.ModuleList([
            KoPEBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias,
                attn_drop=attn_drop, ffn_drop=ffn_drop,
                phase_coupling=self.shared_phase_coupling,
                phase_gamma=kope_gamma,
                kope_vo_rotation=kope_vo_rotation,
                kope_mix=kope_mix,
                kope_mix_init_gain=kope_mix_init_gain,
                kope_mix_phase_norm=kope_mix_phase_norm,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    @staticmethod
    def _build_patch_coords(feat_h: int, feat_w: int,
                            device: torch.device) -> torch.Tensor:
        ys = torch.arange(feat_h, dtype=torch.float32, device=device)
        xs = torch.arange(feat_w, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((grid_y, grid_x), dim=-1).reshape(1, -1, 2)

    def _initial_phase(self, B: int, feat_h: int, feat_w: int,
                       device: torch.device,
                       dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the initial phase: identity ``(1, 0)`` for the CLS token and
        any register tokens, followed by 2D RoPE on the patch grid. The same
        phase is used across heads at init; heads diverge through later
        coupling steps."""
        ext = 1 + self.num_register_tokens
        coords = self._build_patch_coords(feat_h, feat_w, device)        # [1, P, 2]
        cos_p, sin_p = build_2d_rope_pair(coords, self._phase_inv_freq)  # [1, P, D//2]
        one = torch.ones((1, ext, cos_p.shape[-1]), dtype=cos_p.dtype, device=device)
        zero = torch.zeros_like(one)
        cos = torch.cat((one, cos_p), dim=1)                             # [1, ext+P, D//2]
        sin = torch.cat((zero, sin_p), dim=1)
        cos = cos.unsqueeze(1).expand(B, self.num_heads, -1, -1).to(dtype=dtype).contiguous()
        sin = sin.unsqueeze(1).expand(B, self.num_heads, -1, -1).to(dtype=dtype).contiguous()
        return cos, sin

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_embed(x)
        tokens = torch.cat((self.cls_token.expand(B, -1, -1), tokens), dim=1)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        if self.register_tokens is not None:
            # Inserted between CLS and patches AFTER pos_embed (which only
            # covers [CLS | patches]); matches simdinov2.
            tokens = torch.cat(
                (
                    tokens[:, :1],
                    self.register_tokens.expand(B, -1, -1),
                    tokens[:, 1:],
                ),
                dim=1,
            )
        return tokens

    def forward_features(self, x: torch.Tensor) -> dict:
        B, _, H, W = x.shape
        feat_h, feat_w = H // self.patch_size, W // self.patch_size
        tokens = self.prepare_tokens(x)
        cos, sin = self._initial_phase(B, feat_h, feat_w, x.device, tokens.dtype)
        state = KoPEState(tokens, cos, sin)
        for blk in self.blocks:
            state = blk(state)
        x_norm = self.norm(state.tokens)
        nr = self.num_register_tokens
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : 1 + nr],
            "x_norm_patchtokens": x_norm[:, 1 + nr :],
            "kope_patch_phase_cos": state.phase_cos[:, :, 1 + nr :],
            "kope_patch_phase_sin": state.phase_sin[:, :, 1 + nr :],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)["x_norm_clstoken"])


# -----------------------------------------------------------------------------
# Variants (matching simdinov2/models/vision_transformer_kope.py)
# -----------------------------------------------------------------------------


def vit_kope_small(patch_size: int = 16, **kwargs) -> DinoVisionTransformerKoPE:
    return DinoVisionTransformerKoPE(patch_size=patch_size, embed_dim=384,
                                     depth=12, num_heads=6, mlp_ratio=4.0,
                                     **kwargs)


def vit_kope_base(patch_size: int = 16, **kwargs) -> DinoVisionTransformerKoPE:
    return DinoVisionTransformerKoPE(patch_size=patch_size, embed_dim=768,
                                     depth=12, num_heads=12, mlp_ratio=4.0,
                                     **kwargs)


def vit_kope_large(patch_size: int = 16, **kwargs) -> DinoVisionTransformerKoPE:
    return DinoVisionTransformerKoPE(patch_size=patch_size, embed_dim=1024,
                                     depth=24, num_heads=16, mlp_ratio=4.0,
                                     **kwargs)


def vit_kope_giant(patch_size: int = 16, **kwargs) -> DinoVisionTransformerKoPE:
    return DinoVisionTransformerKoPE(patch_size=patch_size, embed_dim=1536,
                                     depth=40, num_heads=24, mlp_ratio=4.0,
                                     **kwargs)


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = vit_kope_base(img_size=224, patch_size=16).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vit_kope_base parameters: {n_params / 1e6:.2f}M")

    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model.forward_features(x)
    print("forward_features output shapes:")
    for k, v in out.items():
        print(f"  {k:24s}: {tuple(v.shape)}")

    # Smoke-test a backward pass on the CLS embedding.
    model.train()
    x = torch.randn(2, 3, 224, 224, device=device, requires_grad=False)
    loss = model(x).square().mean()
    loss.backward()
    print(f"backward ok, loss = {loss.item():.4f}")
