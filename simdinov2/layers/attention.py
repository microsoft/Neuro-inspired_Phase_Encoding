# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger("dinov2")

XFORMERS_AVAILABLE = False
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind
        #from xformers.ops.rmsnorm import rms_norm as rmsnorm
        XFORMERS_AVAILABLE = True
        warnings.warn("Using xFormers (Attention)")
        # from xformers.ops.fmha import _set_use_fa3
        # _set_use_fa3(True)
    else:
        warnings.warn("xFormers is disabled (Attention)")
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")

def rmsnorm(x):
    return F.rms_norm(x, (x.size(-1),))
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
_USE_FUSED_ATTN = int(os.environ.get('USE_FUSED_ATTN', _HAS_FUSED_ATTN))
if _USE_FUSED_ATTN:
    warnings.warn("Using SDPA Attention")

# added for rope
def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

# added for rope
def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (x * cos) + (rotate_half(x) * sin)

def apply_rotary_pos_emb_pairs(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [..., D] with explicit [B, H, S, D] in practice for attention tensors
    if x.size(-1) % 2 != 0:
        raise ValueError("Rotary embedding requires even feature dimension")

    feature_dim = x.size(-1)

    if cos.shape[-1] == feature_dim:
        if cos.dim() == x.dim() - 1:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        if cos.dtype != x.dtype:
            cos = cos.to(dtype=x.dtype)
        if sin.dtype != x.dtype:
            sin = sin.to(dtype=x.dtype)
        # rely on broadcast semantics; no per-head replication here
        return apply_rotary_pos_emb(x, cos, sin)

    half_dim = feature_dim // 2
    if cos.shape[-1] != half_dim:
        raise ValueError("Mismatched rotary pair dimension")

    x_contig = x.contiguous()
    batch, heads, seq_len, _ = x_contig.shape

    def _align_trig(tensor: Tensor, name: str) -> Tensor:
        aligned = tensor.to(dtype=x.dtype) if tensor.dtype != x.dtype else tensor
        if aligned.dim() == x.dim() - 1:
            aligned = aligned.unsqueeze(1)
        if aligned.dim() != x.dim():
            raise ValueError(f"{name} must broadcast to [B, H, S, D/2]")
        # Allow either [B, H, S, half] or [B, 1, S, half]
        if aligned.shape[0] not in (1, batch):
            raise ValueError(f"{name} batch dimension incompatible with input")
        if aligned.shape[1] not in (1, heads):
            raise ValueError(f"{name} head dimension incompatible with input")
        if aligned.shape[2] not in (1, seq_len):
            raise ValueError(f"{name} sequence dimension incompatible with input")
        return aligned

    cos_aligned = _align_trig(cos, "cos")
    sin_aligned = _align_trig(sin, "sin")

    # reshape last dim into complex pairs: [B, H, S, D/2, 2]
    x_view = x_contig.view(batch, heads, seq_len, half_dim, 2)
    real = x_view[..., 0]
    imag = x_view[..., 1]
    updated_real = real * cos_aligned - imag * sin_aligned
    updated_imag = real * sin_aligned + imag * cos_aligned
    stacked = torch.stack((updated_real, updated_imag), dim=-1)
    # restore flattened complex pairs back to [B, H, S, D]
    return stacked.view_as(x_contig)

# added for rope
def _repeat_trig(angles: Tensor) -> Tensor:
    return torch.repeat_interleave(angles, 2, dim=-1)

# added for rope
def build_2d_rope_from_coords(
    coords: Tensor,
    inv_freq: Tensor,
) -> Tuple[Tensor, Tensor]:
    # B, P, 2
    coords_fp32 = coords.to(dtype=torch.float32)
    # D//4
    inv_freq_fp32 = inv_freq.to(dtype=torch.float32)
    # B, P, D//4 -> B, P, D//2
    row_angles = _repeat_trig(coords_fp32[..., 0].unsqueeze(-1) * inv_freq_fp32)
    col_angles = _repeat_trig(coords_fp32[..., 1].unsqueeze(-1) * inv_freq_fp32)
    # B, P, D
    cos = torch.cat((torch.cos(row_angles), torch.cos(col_angles)), dim=-1)
    sin = torch.cat((torch.sin(row_angles), torch.sin(col_angles)), dim=-1)
    return cos, sin

# added for rope
def apply_rotary_emb_2d(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    start_index: int,
) -> Tuple[Tensor, Tensor]:
    if q.size(-2) <= start_index:
        return q, k
    q_patch = q[..., start_index:, :]
    k_patch = k[..., start_index:, :]
    q_rot = apply_rotary_pos_emb_pairs(q_patch, cos, sin)
    k_rot = apply_rotary_pos_emb_pairs(k_patch, cos, sin)
    q[..., start_index:, :] = q_rot
    k[..., start_index:, :] = k_rot
    return q, k

# added for rope
def apply_rotary_emb_2d_v(
    v: Tensor,
    cos: Tensor,
    sin: Tensor,
    start_index: int,
) -> Tensor:
    if v.size(-2) <= start_index:
        return v
    v_patch = v[..., start_index:, :]
    v_rot = apply_rotary_pos_emb_pairs(v_patch, cos, sin)
    v[..., start_index:, :] = v_rot
    return v

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qknorm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.head_dim = head_dim

        # added for rope
        self.use_2d_rope = kwargs.pop("use_2d_rope", False)
        self.ext_token_num = kwargs.get("ext_token_num", 0)
        self.rope_base = kwargs.get("base", 10000)
        self.rope_mix = kwargs.pop("rope_mix", False)
        self.vo_rotation = kwargs.pop("vo_rotation", False)

        self._rope_coords_batch: Optional[Tensor] = None
        self._rope_coords_flat: Optional[Tensor] = None
        self._rope_cache: dict = {}
        if self.use_2d_rope:
            if head_dim % 2 != 0 or (head_dim // 2) % 2 != 0:
                raise ValueError("Head dimension must be divisible by 4 to use 2D RoPE")
            inv_freq = 1.0 / (
                self.rope_base
                ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32) / (head_dim // 2))
            )
            self.register_buffer("_rope_inv_freq", inv_freq, persistent=False)
        else:
            self.register_buffer("_rope_inv_freq", torch.empty(0), persistent=False)

        if self.rope_mix:
            rotary_dim = head_dim // 2
            self.rope_mix_coef = nn.Parameter(torch.empty(self.num_heads, rotary_dim, rotary_dim))
            self._reset_rope_mix_parameters()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qknorm = qknorm

        self.enable_analysis = False
        self.last_metrics = {}

    # added for rope
    def set_rope_metadata(
        self,
        *,
        coords_batch: Optional[Tensor] = None,
        coords_flat: Optional[Tensor] = None,
    ) -> None:
        if not self.use_2d_rope:
            return
        self._rope_coords_batch = coords_batch
        self._rope_coords_flat = coords_flat
        self._rope_cache.clear()

    # added for rope
    def _select_rope_coords(
        self,
        batch_size: int,
        patch_tokens: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        for coords in (self._rope_coords_batch, self._rope_coords_flat):
            if coords is None:
                continue
            if coords.shape[0] == batch_size and coords.shape[1] == patch_tokens:
                if coords.device != device:
                    converted = coords.to(device=device)
                    if coords is self._rope_coords_batch:
                        self._rope_coords_batch = converted
                    else:
                        self._rope_coords_flat = converted
                    coords = converted
                return coords
        return None

    # added for rope
    def _get_rotary_cos_sin(
        self,
        coords: Tensor,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        cache_key = (coords.data_ptr(), coords.shape)
        cached = self._rope_cache.get(cache_key)
        if cached is None:
            cos, sin = build_2d_rope_from_coords(coords, self._rope_inv_freq)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            self._rope_cache[cache_key] = (cos, sin)
        else:
            cos, sin = cached
        if cos.dtype != dtype:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.enable_analysis:
            # Calculate metrics for CLS token (index 0)
            # q: [B, H, N, D] -> q_cls: [B, H, 1, D]
            q_cls = q[:, :, :1, :]
            # k: [B, H, N, D]

            # Calculate attention scores for CLS token
            # [B, H, 1, D] @ [B, H, D, N] -> [B, H, 1, N]
            attn_logits = (q_cls * self.scale) @ k.transpose(-2, -1)
            attn_probs = attn_logits.softmax(dim=-1)

            # Here we calculate entropy and gini over the full distribution.
            p = attn_probs.squeeze(2) # [B, H, N]

            # --- Entropy Analysis ---
            entropy_map = -(p * torch.log(p + 1e-6)).sum(dim=-1) # [B, H]
            entropy_per_head = entropy_map.mean(dim=0) # [H]
            entropy_avg = entropy_per_head.mean()

            # Lowest entropy = Most focused
            k_ent = min(2, len(entropy_per_head))
            k_ent_half = max(1, len(entropy_per_head) // 2)

            focused_entropy_vals, _ = torch.topk(entropy_per_head, k=k_ent, largest=False)
            entropy_top1 = focused_entropy_vals[0]
            entropy_top2 = focused_entropy_vals.mean()

            focused_entropy_half_vals, _ = torch.topk(entropy_per_head, k=k_ent_half, largest=False)
            entropy_tophalf = focused_entropy_half_vals.mean()

            # --- Gini Index Analysis ---
            p_sorted, _ = torch.sort(p, dim=-1)
            n = p.shape[-1]
            index = torch.arange(1, n + 1, device=p.device).view(1, 1, -1)
            gini_map = ((2 * index - n - 1) * p_sorted).sum(dim=-1) / (n * p_sorted.sum(dim=-1) + 1e-6)

            # Per-head analysis
            gini_per_head = gini_map.mean(dim=0) # [H]
            gini_avg = gini_per_head.mean()

            k_val = min(2, len(gini_per_head))
            top_ginis, top_gini_indices = torch.topk(gini_per_head, k=k_val)
            gini_top1 = top_ginis[0]
            gini_top2 = top_ginis.mean()

            k_half = max(1, len(gini_per_head) // 2)
            top_half_vals, top_half_indices = torch.topk(gini_per_head, k=k_half)
            gini_tophalf = top_half_vals.mean()

            metrics = {
                "attn_entropy": entropy_avg.item(),
                "attn_entropy_top1": entropy_top1.item(),
                "attn_entropy_top2": entropy_top2.item(),
                "attn_entropy_tophalf": entropy_tophalf.item(),
                "attn_gini": gini_avg.item(),
                "attn_gini_top1": gini_top1.item(),
                "attn_gini_top2": gini_top2.item(),
                "attn_gini_tophalf": gini_tophalf.item(),
            }

            # Add per-head stats
            for h_idx in range(len(gini_per_head)):
                metrics[f"attn_gini_h{h_idx}"] = gini_per_head[h_idx].item()
                metrics[f"attn_entropy_h{h_idx}"] = entropy_per_head[h_idx].item()

            self.last_metrics = metrics

        # added for rope
        if self.use_2d_rope:
            patch_tokens = N - self.ext_token_num
            if patch_tokens > 0:
                coords = self._select_rope_coords(B, patch_tokens, x.device)
                if coords is None:
                    raise RuntimeError("2D RoPE metadata not set or mismatched with current input")
                cos, sin = self._get_rotary_cos_sin(coords, q.dtype)

                if self.rope_mix:
                    mix = self.rope_mix_coef.to(dtype=q.dtype)
                    cos = torch.einsum("bnhc,hcd->bnhd", cos, mix)
                    sin = torch.einsum("bnhc,hcd->bnhd", sin, mix)
                    inv = (cos.square().add_(sin.square()).clamp_min(1e-5)).rsqrt()
                    cos = cos * inv
                    sin = sin * inv

                q, k = apply_rotary_emb_2d(q, k, cos, sin, self.ext_token_num)
                if self.vo_rotation:
                    v = apply_rotary_emb_2d_v(v, cos, sin, self.ext_token_num)

        if self.qknorm:
            q, k = rmsnorm(q), rmsnorm(k)

        if _USE_FUSED_ATTN:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2)

        if self.use_2d_rope and self.vo_rotation:
            # Demodulation
            # x is [B, H, N, D] here (after transpose)
            x = apply_rotary_emb_2d_v(x, cos, -sin, self.ext_token_num)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _reset_rope_mix_parameters(self):
        gain = 0.1
        with torch.no_grad():
            nn.init.xavier_normal_(self.rope_mix_coef, gain=gain)
            rotary_dim = self.rope_mix_coef.shape[-1]
            eye = torch.eye(rotary_dim, device=self.rope_mix_coef.device, dtype=self.rope_mix_coef.dtype)
            self.rope_mix_coef.add_(eye.unsqueeze(0))

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)

        q, k, v = unbind(qkv, 2)

        # added for rope
        if self.use_2d_rope:
            patch_tokens = N - self.ext_token_num
            if patch_tokens > 0:
                coords = self._select_rope_coords(B, patch_tokens, x.device)
                if coords is None:
                    raise RuntimeError("2D RoPE metadata not set or mismatched with current input")
                cos, sin = self._get_rotary_cos_sin(coords, q.dtype)

                if self.rope_mix:
                    mix = self.rope_mix_coef.to(dtype=q.dtype)
                    cos = torch.einsum("bnhc,hcd->bnhd", cos, mix)
                    sin = torch.einsum("bnhc,hcd->bnhd", sin, mix)
                    inv = (cos.square().add_(sin.square()).clamp_min(1e-5)).rsqrt()
                    cos = cos * inv
                    sin = sin * inv

                q_perm = q.permute(0, 2, 1, 3)
                k_perm = k.permute(0, 2, 1, 3)
                q_rot, k_rot = apply_rotary_emb_2d(q_perm, k_perm, cos, sin, self.ext_token_num)
                q = q_rot.permute(0, 2, 1, 3).contiguous()
                k = k_rot.permute(0, 2, 1, 3).contiguous()

                if self.vo_rotation:
                    v_perm = v.permute(0, 2, 1, 3)
                    v_perm = apply_rotary_emb_2d_v(v_perm, cos, sin, self.ext_token_num)
                    v = v_perm.permute(0, 2, 1, 3).contiguous()

        if self.qknorm:
            # Attention.forward applies qknorm after RoPE; do the same here.
            q, k = rmsnorm(q), rmsnorm(k)

        if self.enable_analysis:
            # q, k are [B, N, H, D]
            # Permute to [B, H, N, D] for consistency with Attention analysis
            q_anal = q.permute(0, 2, 1, 3)
            k_anal = k.permute(0, 2, 1, 3)

            q_cls = q_anal[:, :, :1, :]

            attn_logits = (q_cls * self.scale) @ k_anal.transpose(-2, -1)
            attn_probs = attn_logits.softmax(dim=-1)
            p = attn_probs.squeeze(2) # [B, H, N]

            # --- Entropy Analysis ---
            entropy_map = -(p * torch.log(p + 1e-6)).sum(dim=-1) # [B, H]
            entropy_per_head = entropy_map.mean(dim=0) # [H]
            entropy_avg = entropy_per_head.mean()

            # Lowest entropy = Most focused
            k_ent = min(2, len(entropy_per_head))
            k_ent_half = max(1, len(entropy_per_head) // 2)

            focused_entropy_vals, _ = torch.topk(entropy_per_head, k=k_ent, largest=False)
            entropy_top1 = focused_entropy_vals[0]
            entropy_top2 = focused_entropy_vals.mean()

            focused_entropy_half_vals, _ = torch.topk(entropy_per_head, k=k_ent_half, largest=False)
            entropy_tophalf = focused_entropy_half_vals.mean()

            # --- Gini Index Analysis ---
            p_sorted, _ = torch.sort(p, dim=-1)
            n = p.shape[-1]
            index = torch.arange(1, n + 1, device=p.device).view(1, 1, -1)
            gini_map = ((2 * index - n - 1) * p_sorted).sum(dim=-1) / (n * p_sorted.sum(dim=-1) + 1e-6)

            # Per-head analysis
            gini_per_head = gini_map.mean(dim=0) # [H]
            gini_avg = gini_per_head.mean()

            k_val = min(2, len(gini_per_head))
            top_ginis, top_gini_indices = torch.topk(gini_per_head, k=k_val)
            gini_top1 = top_ginis[0]
            gini_top2 = top_ginis.mean()

            k_half = max(1, len(gini_per_head) // 2)
            top_half_vals, top_half_indices = torch.topk(gini_per_head, k=k_half)
            gini_tophalf = top_half_vals.mean()

            metrics = {
                "attn_entropy": entropy_avg.item(),
                "attn_entropy_top1": entropy_top1.item(),
                "attn_entropy_top2": entropy_top2.item(),
                "attn_entropy_tophalf": entropy_tophalf.item(),
                "attn_gini": gini_avg.item(),
                "attn_gini_top1": gini_top1.item(),
                "attn_gini_top2": gini_top2.item(),
                "attn_gini_tophalf": gini_tophalf.item(),
            }

            # Add per-head stats
            for h_idx in range(len(gini_per_head)):
                metrics[f"attn_gini_h{h_idx}"] = gini_per_head[h_idx].item()
                metrics[f"attn_entropy_h{h_idx}"] = entropy_per_head[h_idx].item()

            self.last_metrics = metrics

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.attn_drop.p if self.training else 0.)

        if self.use_2d_rope and self.vo_rotation:
            # Demodulation
            x_perm = x.permute(0, 2, 1, 3) # [B, H, N, D]
            x_perm = apply_rotary_emb_2d_v(x_perm, cos, -sin, self.ext_token_num)
            x = x_perm.permute(0, 2, 1, 3) # Back to [B, N, H, D]

        x = x.view([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
