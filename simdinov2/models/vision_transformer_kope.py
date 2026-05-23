# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from functools import partial
import logging
import math
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
)
from simdinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused
import simdinov2.layers.attention as attention_ops
from simdinov2.layers.block import (
    DropPath,
    LayerScale,
    setup_layer_scales,
    get_attn_bias_and_cat,
    get_indexs_scales,
    add_residual,
)

logger = logging.getLogger("dinov2")


@torch.compiler.disable
def memory_efficient_attention_nocompile(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[object] = None,
    p: float = 0.0,
) -> torch.Tensor:
    """A memory-efficient attention function without torch.compile.

    This is a wrapper around attention_ops.memory_efficient_attention to
    disable torch.compile within its scope.
    """
    return attention_ops.memory_efficient_attention(
        q,
        k,
        v,
        attn_bias=attn_bias,
        p=p,
    )


#@torch.compile(mode='default')
def apply_rotary_pairs(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    d = x.size(-1)
    half = d // 2
    if cos.shape[-1] == d:
        # Full-dim trig tensors aren't expected in KoPE path, but keep a correct
        # implementation here for completeness.
        if cos.dim() == x.dim() - 1:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        cos = cos.to(dtype=x.dtype) if cos.dtype != x.dtype else cos
        sin = sin.to(dtype=x.dtype) if sin.dtype != x.dtype else sin
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (x * cos) + (rot * sin)

    if cos.shape[-1] != half:
        raise ValueError("Mismatched rotary pair dimension")

    cos = cos.to(dtype=x.dtype) if cos.dtype != x.dtype else cos
    sin = sin.to(dtype=x.dtype) if sin.dtype != x.dtype else sin

    # x: [B, N, H, D] -> [B, N, H, D/2, 2]
    x_view = x.reshape(*x.shape[:-1], half, 2)
    real = x_view[..., 0]
    imag = x_view[..., 1]
    updated_real = real * cos - imag * sin
    updated_imag = real * sin + imag * cos
    return torch.stack((updated_real, updated_imag), dim=-1).view_as(x)


def named_apply(
    fn: Callable, module: nn.Module, name: str = "", depth_first: bool = True, include_root: bool = False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class PhaseStep(nn.Module):
    def __init__(self, init=0.01, max_value=1.0):
        super().__init__()
        # inverse softplus
        self._raw = nn.Parameter(torch.log(torch.expm1(torch.tensor([init], dtype=torch.float32))))
        self.max_value = max_value

    def forward(self) -> torch.Tensor:
        return torch.clamp(F.softplus(self._raw), max=self.max_value)


class KoPEIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.supports_state_list = True

    def forward(
        self,
        state: Union["KoPEState", List["KoPEState"]],
        **kwargs,
    ) -> Union["KoPEState", List["KoPEState"]]:
        return state


@dataclass
class KoPEState:
    tokens: torch.Tensor
    phase_cos: torch.Tensor
    phase_sin: torch.Tensor


class KoPEBlockChunk(nn.ModuleList):
    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__(modules)
        self.supports_state_list = True

    def forward(
        self,
        states: Union["KoPEState", List["KoPEState"]],
        **kwargs,
    ) -> Union["KoPEState", List["KoPEState"]]:
        current: Union["KoPEState", List["KoPEState"]] = states
        for block in self:
            if isinstance(current, list):
                if getattr(block, "supports_state_list", False):
                    current = block(current, **kwargs)  # type: ignore[assignment]
                else:
                    current = [block(item, **kwargs) for item in current]
            else:
                current = block(current, **kwargs)
        return current


class KoPEAttention(attention_ops.Attention):
    def __init__(self, *args, **kwargs) -> None:
        kwargs = dict(kwargs)
        self.kope_vo_rotation = kwargs.pop("kope_vo_rotation", False)
        self.kope_mix = kwargs.pop("kope_mix", False)
        self.kope_mix_init_gain = kwargs.pop("kope_mix_init_gain", 0.1)
        self.kope_mix_phase_norm = kwargs.pop("kope_mix_phase_norm", False)
        self.checkpoint_rotation = kwargs.pop("checkpoint_rotation", True)
        kwargs.setdefault("use_2d_rope", False)
        super().__init__(*args, **kwargs)

        if self.kope_mix:
            rotary_dim = self.head_dim // 2
            if rotary_dim == 0:
                raise ValueError("KoPE mix requires head_dim // 2 > 0")
            self.kope_mix_coef = nn.Parameter(torch.empty(self.num_heads, rotary_dim, rotary_dim))
            self._reset_kope_mix_parameters()
        else:
            self.register_parameter("kope_mix_coef", None)

        self.enable_analysis = False
        self.last_metrics = {}

    def _reset_kope_mix_parameters(self) -> None:
        gain = float(self.kope_mix_init_gain)
        with torch.no_grad():
            nn.init.xavier_normal_(self.kope_mix_coef, gain=gain)
            rotary_dim = self.kope_mix_coef.shape[-1]
            eye = torch.eye(rotary_dim, device=self.kope_mix_coef.device, dtype=self.kope_mix_coef.dtype)
            self.kope_mix_coef.add_(eye.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        phase_cos: torch.Tensor,
        phase_sin: torch.Tensor,
        attn_bias: Optional[object] = None,
    ) -> torch.Tensor:
        if attn_bias is not None and not attention_ops.XFORMERS_AVAILABLE:
            raise RuntimeError("KoPEAttention with attn_bias requires xFormers support")

        B, N, C = x.shape
        # Rotary params follow the same layout as q/k: [B, N, H, D/2]
        cos_heads = phase_cos.to(dtype=x.dtype)
        sin_heads = phase_sin.to(dtype=x.dtype)

        if self.checkpoint_rotation:
            def phase_rotation(cos, sin, x):
                qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
                if attention_ops.XFORMERS_AVAILABLE:
                    from xformers.ops import unbind
                    q, k, v = unbind(qkv, dim=2)
                else:
                    q, k, v = qkv.unbind(dim=2)

                if self.kope_mix:
                    if not self.kope_mix_phase_norm:
                        mix = F.normalize(self.kope_mix_coef.to(dtype=cos.dtype), p=2, dim=-2, eps=1e-6)
                        cos = torch.einsum("bnhc,hcd->bnhd", cos, mix)
                        sin = torch.einsum("bnhc,hcd->bnhd", sin, mix)
                    else:
                        mix = self.kope_mix_coef.to(dtype=cos.dtype)
                        cos = torch.einsum("bnhc,hcd->bnhd", cos, mix)
                        sin = torch.einsum("bnhc,hcd->bnhd", sin, mix)
                        inv = (cos.square().add_(sin.square()).clamp_min(1e-5)).rsqrt()
                        cos = cos * inv
                        sin = sin * inv
                q = apply_rotary_pairs(q, cos, sin)
                k = apply_rotary_pairs(k, cos, sin)
                if self.kope_vo_rotation:
                    v = apply_rotary_pairs(v, cos, sin)

                return cos, sin, q, k, v
            cos_heads, sin_heads, q, k, v = torch.utils.checkpoint.checkpoint(
                phase_rotation,
                cos_heads,
                sin_heads,
                x,
                use_reentrant=False,
            )
        else:
            qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
            if attention_ops.XFORMERS_AVAILABLE:
                from xformers.ops import unbind
                q, k, v = unbind(qkv, dim=2)
            else:
                q, k, v = qkv.unbind(dim=2)

            if self.kope_mix:
                # Mixture of phases: [B,N,H,R] x [H,R,R] -> [B,N,H,R]
                if not self.kope_mix_phase_norm:
                    mix = F.normalize(self.kope_mix_coef.to(dtype=cos_heads.dtype), p=2, dim=-2, eps=1e-6)
                    cos_heads = torch.einsum("bnhc,hcd->bnhd", cos_heads, mix)
                    sin_heads = torch.einsum("bnhc,hcd->bnhd", sin_heads, mix)
                else:
                    def mix_phase(mix, cos, sin):
                        cos = torch.einsum("bnhc,hcd->bnhd", cos, mix)
                        sin = torch.einsum("bnhc,hcd->bnhd", sin, mix)
                        inv = (cos.square().add_(sin.square()).clamp_min(1e-5)).rsqrt()
                        return cos * inv, sin * inv
                    cos_heads, sin_heads = torch.utils.checkpoint.checkpoint(
                        mix_phase,
                        self.kope_mix_coef.to(dtype=cos_heads.dtype),
                        cos_heads,
                        sin_heads,
                        use_reentrant=False,
                    )

            q = apply_rotary_pairs(q, cos_heads, sin_heads)
            k = apply_rotary_pairs(k, cos_heads, sin_heads)

            if self.kope_vo_rotation:
                v = apply_rotary_pairs(v, cos_heads, sin_heads)

        if self.enable_analysis:
            # q, k: [B, N, H, D] -> permute to [B, H, N, D]
            q_anal = q.permute(0, 2, 1, 3)
            k_anal = k.permute(0, 2, 1, 3)

            q_cls = q_anal[:, :, :1, :]
            attn_logits = (q_cls * self.scale) @ k_anal.transpose(-2, -1)
            attn_probs = attn_logits.softmax(dim=-1)
            p = attn_probs.squeeze(2) # [B, H, N]

            # Entropy Analysis
            entropy_map = -(p * torch.log(p + 1e-6)).sum(dim=-1) # [B, H]
            entropy_per_head = entropy_map.mean(dim=0) # [H]
            entropy_avg = entropy_per_head.mean()

            # Entropy Top-K (Most Focused = Lowest Entropy)
            # We sort ascending because lower entropy = higher focus
            k_ent = min(2, len(entropy_per_head))
            k_ent_half = max(1, len(entropy_per_head) // 2)

            focused_entropy_vals, _ = torch.topk(entropy_per_head, k=k_ent, largest=False)
            entropy_top1 = focused_entropy_vals[0]
            entropy_top2 = focused_entropy_vals.mean()

            focused_entropy_half_vals, _ = torch.topk(entropy_per_head, k=k_ent_half, largest=False)
            entropy_tophalf = focused_entropy_half_vals.mean()

            p_sorted, _ = torch.sort(p, dim=-1)
            n = p.shape[-1]
            index = torch.arange(1, n + 1, device=p.device).view(1, 1, -1)
            # gini per sample per head: [B, H]
            gini_map = ((2 * index - n - 1) * p_sorted).sum(dim=-1) / (n * p_sorted.sum(dim=-1) + 1e-6)

            # Analyze per-head stats (Average over Batch)
            gini_per_head = gini_map.mean(dim=0) # [H]
            gini_avg = gini_per_head.mean()

            # Top-K Gini (Most specialized heads)
            k_val = min(2, len(gini_per_head))
            top_ginis, top_gini_indices = torch.topk(gini_per_head, k=k_val)
            gini_top1 = top_ginis[0]
            gini_top2 = top_ginis.mean()

            # Top-Half Gini (Broader specialization check)
            k_half = max(1, len(gini_per_head) // 2)
            top_half_vals, top_half_indices = torch.topk(gini_per_head, k=k_half)
            gini_tophalf = top_half_vals.mean()

            # Phase Sync
            # p: [B, H, N] -> weights: [B, H, N, 1]
            weights = p.unsqueeze(-1)
            # cos_heads: [B, N, H, D/2] -> [B, H, N, D/2]
            phase_cos_anal = cos_heads.permute(0, 2, 1, 3)
            phase_sin_anal = sin_heads.permute(0, 2, 1, 3)

            R_cos = (weights * phase_cos_anal).sum(dim=2) # [B, H, D/2]
            R_sin = (weights * phase_sin_anal).sum(dim=2)
            R = torch.sqrt(R_cos.square() + R_sin.square())

            # Sync Per Head (Average over Batch and D/2)
            sync_per_head = R.mean(dim=[0, 2]) # [H]
            sync_avg = sync_per_head.mean()

            # Sync for the Gini-Specialized heads
            sync_gini_top1 = sync_per_head[top_gini_indices[0]]
            sync_gini_top2 = sync_per_head[top_gini_indices].mean()
            sync_gini_tophalf = sync_per_head[top_half_indices].mean()

            metrics = {
                "attn_entropy": entropy_avg.item(),
                "attn_entropy_top1": entropy_top1.item(),
                "attn_entropy_top2": entropy_top2.item(),
                "attn_entropy_tophalf": entropy_tophalf.item(),
                "attn_gini": gini_avg.item(),
                "attn_gini_top1": gini_top1.item(),
                "attn_gini_top2": gini_top2.item(),
                "attn_gini_tophalf": gini_tophalf.item(),
                "phase_sync": sync_avg.item(),
                "phase_sync_gini_top1": sync_gini_top1.item(),
                "phase_sync_gini_top2": sync_gini_top2.item(),
                "phase_sync_gini_tophalf": sync_gini_tophalf.item()
            }

            # Add per-head metrics
            for h_idx in range(len(gini_per_head)):
                metrics[f"attn_gini_h{h_idx}"] = gini_per_head[h_idx].item()
                metrics[f"attn_entropy_h{h_idx}"] = entropy_per_head[h_idx].item()
                metrics[f"phase_sync_h{h_idx}"] = sync_per_head[h_idx].item()

            self.last_metrics = metrics

        if attention_ops.XFORMERS_AVAILABLE:
            q_xf, k_xf, v_xf = q, k, v
            if self.qknorm:
                q_xf = attention_ops.rmsnorm(q_xf)
                k_xf = attention_ops.rmsnorm(k_xf)
            x = memory_efficient_attention_nocompile(
                q_xf,
                k_xf,
                v_xf,
                attn_bias=attn_bias,
                p=self.attn_drop.p if self.training else 0.0,
            )
            if self.kope_vo_rotation:
                x_heads = x.view(B, N, self.num_heads, self.head_dim)
                if self.checkpoint_rotation:
                    demod_heads = torch.utils.checkpoint.checkpoint(
                        apply_rotary_pairs,
                        x_heads,
                        cos_heads,
                        -sin_heads,
                        use_reentrant=False,
                    )
                else:
                    demod_heads = apply_rotary_pairs(
                        x_heads,
                        cos_heads,
                        -sin_heads,
                    )
                x = demod_heads.reshape(B, N, C)
            else:
                x = x.reshape(B, N, C)
        else:
            # q/k already rotated in [B,N,H,D]; move to [B,H,N,D] for non-xFormers attention.
            q_rot = q.permute(0, 2, 1, 3)
            k_rot = k.permute(0, 2, 1, 3)
            v_perm = v.permute(0, 2, 1, 3)
            if self.qknorm:
                q_rot = attention_ops.rmsnorm(q_rot)
                k_rot = attention_ops.rmsnorm(k_rot)
            if attn_bias is not None:
                raise RuntimeError("attn_bias is unsupported without xFormers")
            if attention_ops._USE_FUSED_ATTN:
                x = F.scaled_dot_product_attention(
                    q_rot,
                    k_rot,
                    v_perm,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                q_scaled = q_rot * self.scale
                attn = q_scaled @ k_rot.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v_perm
            x = x.transpose(1, 2)
            if self.kope_vo_rotation:
                x_heads = x
                demod_heads = apply_rotary_pairs(
                    x_heads,
                    cos_heads,
                    -sin_heads,
                )
                x = demod_heads.reshape(B, N, C)
            else:
                x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KoPECoupling(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ext_token_num: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        coupling_use_vo_proj: bool = False,
        fixed_layout: bool = True,
        coupling_use_tanh: bool = False,
        qknorm: bool = False,
        qknorm_learn: bool = False,
        coupling_qk_multilayer: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ext_token_num = ext_token_num
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("KoPECoupling requires even head dimension")
        self.scale = self.head_dim**-0.5
        self.coupling_qk_multilayer = coupling_qk_multilayer
        if coupling_qk_multilayer:
            self.q_proj = nn.Sequential(
                nn.Linear(dim, dim, bias=qkv_bias),
                nn.GELU(),
                nn.Linear(dim, dim, bias=qkv_bias),
            )
            self.k_proj = nn.Sequential(
                nn.Linear(dim, dim, bias=qkv_bias),
                nn.GELU(),
                nn.Linear(dim, dim, bias=qkv_bias),
            )
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.qknorm = qknorm
        self.qknorm_learn = qknorm_learn
        if self.qknorm and self.qknorm_learn:
            self.qk_scale = nn.Parameter(torch.ones(1, 1, self.num_heads, self.head_dim))
        else:
            self.register_parameter("qk_scale", None)
        self.coupling_use_vo_proj = coupling_use_vo_proj
        if self.coupling_use_vo_proj:
            self.v_proj = nn.Linear(dim, dim, bias=proj_bias)
            self.o_proj = nn.Linear(dim, dim, bias=proj_bias)
        else:
            self.v_proj = None
            self.o_proj = None

        self.attn_dropout = nn.Dropout(attn_drop)
        self.output_dropout = nn.Dropout(proj_drop)
        self.coupling_use_tanh = coupling_use_tanh

        self.fixed_layout = fixed_layout
        if not coupling_use_vo_proj:
            if not self.fixed_layout:
                warnings.warn(
                    "KoPECoupling: disabling vo_proj requires fixed_layout=True to preserve cos/sin pairing",
                    stacklevel=2,
                )
            self.fixed_layout = True

    @torch.compile(mode='default')
    def forward(
        self,
        tokens: torch.Tensor,
        phase_cos: torch.Tensor,
        phase_sin: torch.Tensor,
        attn_bias: Optional[object] = None,
        update_phase: bool = False,
        gamma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # tokens: [B, N, C], phase_*: [B, N, H, D/2]
        if attn_bias is not None and not attention_ops.XFORMERS_AVAILABLE:
            raise RuntimeError("KoPECoupling with attn_bias requires xFormers support")

        B, N, head_count, half_dim = phase_cos.shape

        q_full = self.q_proj(tokens).view(B, N, self.num_heads, self.head_dim)
        k_full = self.k_proj(tokens).view(B, N, self.num_heads, self.head_dim)

        if self.qknorm:
            q_full = attention_ops.rmsnorm(q_full)
            k_full = attention_ops.rmsnorm(k_full)
            if self.qk_scale is not None:
                q_full = q_full * self.qk_scale.to(dtype=q_full.dtype)
                #k_full = k_full * self.qk_scale

        phase_head = torch.cat((phase_cos, phase_sin), dim=-1).reshape(B, N, self.num_heads, self.head_dim)
        if self.coupling_use_vo_proj:
            v_tokens = self.v_proj(phase_head.reshape(B, N, self.dim))
            v_full = v_tokens.view(B, N, self.num_heads, self.head_dim)
        else:
            v_full = phase_head.to(dtype=q_full.dtype)

        # Keep q/k/v in [B, N, H, D] for xFormers path; only build BHND for fallback.
        q = q_full
        k = k_full
        v = v_full

        use_xformers_attn = attention_ops.XFORMERS_AVAILABLE and not self.coupling_use_tanh

        if use_xformers_attn:
            attn_output = memory_efficient_attention_nocompile(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.attn_dropout.p if self.training else 0.0,
            )
            o_heads = attn_output
        else:
            # fallback path uses [B, H, N, D]
            q_perm = q.permute(0, 2, 1, 3)
            k_perm = k.permute(0, 2, 1, 3)
            attn_scores = (q_perm * self.scale) @ k_perm.transpose(-2, -1)
            if self.coupling_use_tanh:
                attn_probs = torch.tanh(attn_scores) * self.scale
            else:
                attn_probs = attn_scores.softmax(dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            v_perm = v.permute(0, 2, 1, 3)
            o_heads = torch.matmul(attn_probs, v_perm)
            o_heads = o_heads.permute(0, 2, 1, 3)

        o_tokens = o_heads.reshape([B, N, self.dim])
        if self.coupling_use_vo_proj:
            o_tokens = self.o_proj(o_tokens)
        o_tokens = self.output_dropout(o_tokens)

        if update_phase:
            assert gamma is not None, "Gamma scaling is required for phase update"
            delta = o_tokens.view(B, N, self.num_heads, 2, self.head_dim // 2)
            phase_head = phase_head.view(B, N, self.num_heads, 2, self.head_dim // 2)
            # projection
            alignment = (delta * phase_head).sum(dim=-2, keepdim=True)
            delta = delta.addcmul(alignment, phase_head, value=-1.0)
            phase_update = F.normalize(phase_head + gamma * delta, dim=-2, eps=1e-5)
            return phase_update[..., 0, :], phase_update[..., 1, :]

        if self.fixed_layout:
            o_full = (
                o_tokens.view(B, N, self.num_heads, 2, self.head_dim // 2)
                .permute(0, 1, 2, 4, 3)
                .contiguous()
            )
        else:
            o_full = o_tokens.view(B, N, self.num_heads, self.head_dim // 2, 2)

        return o_full

class KoPEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        ffn_drop: float = 0.0,
        attn_drop: float = 0.0,
        layerscale: Optional[float] = None,
        drop_path: float = 0.0,
        qknorm: bool = False,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        ext_token_num: int = 1,
        phase_gamma: float = 0.05,
        update_ext_token_phase: bool = True,
        phase_coupling_fn: Optional[Callable[..., torch.Tensor]] = None,
        *,
        coupling_use_vo_proj: bool = False,
        coupling_use_tanh: bool = False,
        kope_vo_rotation: bool = True,
        fixed_layout: Optional[bool] = None,
        coupling_qknorm: bool = True,
        coupling_qknorm_learn: bool = True,
        coupling_qk_multilayer: bool = False,
        use_kope_attn: bool = True,
        kope_mix: bool = True,
        kope_mix_init_gain: float = 0.1,
        kope_mix_phase_norm: bool = True,
        checkpoint_phase_update: bool = True,
        checkpoint_ffn_func: bool = True,
        checkpoint_rotation: bool = True,
        no_phase_norm: bool = True,
        learn_phase_gamma: bool = False,
        shared_phase_coupling_gamma: Optional[PhaseStep] = None,
    ) -> None:
        super().__init__()
        self.ext_token_num = ext_token_num
        self.learn_phase_gamma = learn_phase_gamma
        if learn_phase_gamma:
            #self.phase_gamma = PhaseStep(init=phase_gamma)
            self.phase_gamma = shared_phase_coupling_gamma if shared_phase_coupling_gamma is not None else PhaseStep(init=phase_gamma)
        else:
            self.register_buffer("phase_gamma", torch.tensor(float(phase_gamma), dtype=torch.float32))
        self.supports_state_list = False
        self.update_ext_token_phase = update_ext_token_phase
        self._freeze_mask_cache: Dict[Tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}
        self.coupling_use_vo_proj = coupling_use_vo_proj
        self.coupling_use_tanh = coupling_use_tanh
        self.kope_vo_rotation = kope_vo_rotation
        self.kope_mix = kope_mix
        self.kope_mix_init_gain = kope_mix_init_gain
        self.kope_mix_phase_norm = kope_mix_phase_norm
        if fixed_layout is None:
            fixed_layout = not coupling_use_vo_proj
        self.fixed_layout = fixed_layout
        self.coupling_qknorm = coupling_qknorm
        self.coupling_qknorm_learn = coupling_qknorm_learn
        self.use_kope_attn = use_kope_attn
        self.checkpoint_phase_update = checkpoint_phase_update
        self.checkpoint_ffn_func = checkpoint_ffn_func
        self.checkpoint_rotation = checkpoint_rotation

        self.norm1 = norm_layer(dim)
        attn_kwargs = dict(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=ffn_drop,
            qknorm=qknorm,
        )
        if True:
            attn_kwargs["ext_token_num"] = ext_token_num
            attn_kwargs["kope_vo_rotation"] = kope_vo_rotation
            attn_kwargs["kope_mix"] = kope_mix
            attn_kwargs["kope_mix_init_gain"] = kope_mix_init_gain
            attn_kwargs["kope_mix_phase_norm"] = kope_mix_phase_norm
            attn_kwargs["checkpoint_rotation"] = checkpoint_rotation
            self.attn = KoPEAttention(**attn_kwargs)
        self.ls1, self.ls2 = setup_layer_scales(dim, layerscale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_drop,
            bias=ffn_bias,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if no_phase_norm:
            self.phase_norm = nn.Identity()
        else:
            self.phase_norm = norm_layer(dim)
        if phase_coupling_fn is None:
            self._phase_coupling_module = KoPECoupling(
                dim=dim,
                num_heads=num_heads,
                ext_token_num=ext_token_num,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=ffn_drop,
                coupling_use_vo_proj=coupling_use_vo_proj,
                fixed_layout=self.fixed_layout,
                coupling_use_tanh=coupling_use_tanh,
                qknorm=coupling_qknorm,
                qknorm_learn=coupling_qknorm_learn,
                coupling_qk_multilayer=coupling_qk_multilayer,
            )
            self._phase_coupling_fn: Callable[..., torch.Tensor] = self._phase_coupling_module
        else:
            self._phase_coupling_module = None
            self._phase_coupling_fn = phase_coupling_fn
        self.sample_drop_ratio = drop_path

    def forward(
        self,
        state: "KoPEState",
    ) -> "KoPEState":
        tokens = state.tokens
        #if self.use_kope_attn:
        assert self.use_kope_attn
        phase_cos = state.phase_cos
        phase_sin = state.phase_sin

        x = tokens
        if self.training and self.sample_drop_ratio > 0.1:
            x = self._apply_stochastic_residual(
                x,
                lambda keep_index: self.ls1(
                    self.attn(
                        self.norm1(x[keep_index]),
                        phase_cos[keep_index],
                        phase_sin[keep_index],
                    )
                ),
            )
            x = self._apply_stochastic_residual(
                x,
                lambda keep_index: (
                    torch.utils.checkpoint.checkpoint(
                        #lambda _t: self.ls2(self.mlp(_t)),
                        #self.norm2(x[keep_index]),
                        lambda _t: self.ls2(self.mlp(self.norm2(_t))),
                        x[keep_index],
                        use_reentrant=False,
                    )
                    if self.checkpoint_ffn_func and self.training
                    else self.ls2(self.mlp(self.norm2(x[keep_index])))
                ),
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), phase_cos, phase_sin)))
            if self.checkpoint_ffn_func:
                ffn_residual = torch.utils.checkpoint.checkpoint(
                    #lambda _t: self.ls2(self.mlp(_t)),
                    #self.norm2(x),
                    lambda _t: self.ls2(self.mlp(self.norm2(_t))),
                    x,
                    use_reentrant=False,
                )
            else:
                ffn_residual = self.ls2(self.mlp(self.norm2(x)))
            x = x + self.drop_path(ffn_residual)
        else:
            x = x + self.ls1(self.attn(self.norm1(x), phase_cos, phase_sin))
            if self.checkpoint_ffn_func:
                ffn_residual = torch.utils.checkpoint.checkpoint(
                    #lambda _t: self.ls2(self.mlp(_t)),
                    #self.norm2(x),
                    lambda _t: self.ls2(self.mlp(self.norm2(_t))),
                    x,
                    use_reentrant=False,
                )
            else:
                ffn_residual = self.ls2(self.mlp(self.norm2(x)))
            x = x + ffn_residual

        phase_cos, phase_sin = self._update_phase(
            x, state.phase_cos, state.phase_sin
        )
        return KoPEState(x, phase_cos, phase_sin)

    def _apply_stochastic_residual(
        self,
        tokens: torch.Tensor,
        residual_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        B = tokens.shape[0]
        sample_subset_size = max(int(B * (1 - self.sample_drop_ratio)), 1)
        keep_index = torch.randperm(B, device=tokens.device)[:sample_subset_size]
        residual = residual_fn(keep_index)
        tokens_flat = tokens.flatten(1)
        residual_flat = residual.flatten(1).to(dtype=tokens.dtype)
        scale = B / sample_subset_size
        tokens_flat = torch.index_add(
            tokens_flat,
            0,
            keep_index,
            residual_flat,
            alpha=scale,
        )
        return tokens_flat.view_as(tokens)

    def _get_phase_freeze_mask(
        self,
        reference: torch.Tensor,
        freeze_count: int,
    ) -> torch.Tensor:
        seq_len = reference.shape[1]
        key = (reference.device, reference.dtype, seq_len, freeze_count)
        mask = self._freeze_mask_cache.get(key)
        if mask is None:
            mask = torch.ones(
                (1, seq_len, 1, 1),
                dtype=reference.dtype,
                device=reference.device,
            )
            if freeze_count > 0:
                mask[:, :freeze_count] = 0
            self._freeze_mask_cache[key] = mask
        return mask

    def _update_phase(
        self,
        tokens: torch.Tensor,
        phase_cos: torch.Tensor,
        phase_sin: torch.Tensor,
        *,
        coupling: Optional[torch.Tensor] = None,
        attn_bias: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        freeze_count = 0
        if not self.update_ext_token_phase and self.ext_token_num > 0:
            freeze_count = min(self.ext_token_num, phase_cos.shape[1])

        state_dtype = phase_cos.dtype
        state_cos = phase_cos
        state_sin = phase_sin

        if True:
            def _coupling_and_update(
                _tokens: torch.Tensor,
                _state_cos: torch.Tensor,
                _state_sin: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                assert freeze_count == 0
                gamma = self.phase_gamma() if self.learn_phase_gamma else self.phase_gamma
                return self._phase_coupling_fn(
                    self.phase_norm(_tokens),
                    _state_cos,
                    _state_sin,
                    attn_bias=attn_bias,
                    update_phase=True,
                    gamma=gamma.to(dtype=_state_cos.dtype, device=_state_cos.device),
                )

            # TODO: check
            if self.checkpoint_phase_update: # and self.training:
                next_cos, next_sin = torch.utils.checkpoint.checkpoint(
                    _coupling_and_update,
                    tokens,
                    state_cos,
                    state_sin,
                    use_reentrant=False,
                )
            else:
                next_cos, next_sin = _coupling_and_update(
                    tokens,
                    state_cos,
                    state_sin,
                )

        next_cos = next_cos.to(dtype=state_dtype)
        next_sin = next_sin.to(dtype=state_dtype)

        return next_cos, next_sin


from xformers.ops import fmha, index_select_cat
attn_bias_cache: Dict[Tuple, Any] = {}

def get_attn_bias_and_cat_states(states, keep_indexs=None, get_phase=True):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    x_list = [state.tokens for state in states]
    if get_phase:
        cos_list = [state.phase_cos for state in states]
        sin_list = [state.phase_sin for state in states]
    batch_sizes = (
        [b.shape[0] for b in keep_indexs]
        if keep_indexs is not None
        else [x.shape[0] for x in x_list]
    )
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if keep_indexs is not None:
        cat_tensors = index_select_cat(
            [x.flatten(1) for x in x_list], keep_indexs
        ).view(1, -1, x_list[0].shape[-1])
        if get_phase:
            cat_tensors_cos = index_select_cat(
                [x.flatten(1) for x in cos_list], keep_indexs
            ).view(1, -1, *cos_list[0].shape[2:])
            cat_tensors_sin = index_select_cat(
                [x.flatten(1) for x in sin_list], keep_indexs
            ).view(1, -1, *sin_list[0].shape[2:])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)
        if get_phase:
            tensors_bs1_cos = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in cos_list)
            cat_tensors_cos = torch.cat(tensors_bs1_cos, dim=1)
            tensors_bs1_sin = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in sin_list)
            cat_tensors_sin = torch.cat(tensors_bs1_sin, dim=1)

    if get_phase:
        return attn_bias_cache[all_shapes], cat_tensors, cat_tensors_cos, cat_tensors_sin
    else:
        return attn_bias_cache[all_shapes], cat_tensors

def drop_add_residual_stochastic_depth_list_states(
    states,
    residual_func,
    sample_drop_ratio=0.0,
    scaling_vector=None,
    require_phase=True,
    checkpoint_residual_func=False,
):
    x_list = [state.tokens for state in states]
    # 1) generate random set of indices for dropping samples in the batch
    keep_indexs_scales = [
        get_indexs_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list
    ]
    keep_indexs = [s[0] for s in keep_indexs_scales]
    residual_scale_factors = [s[1] for s in keep_indexs_scales]

    # 2) get attention bias and index+concat the tensors
    if require_phase:
        attn_bias, x_cat, phase_cos_cat, phase_sin_cat = get_attn_bias_and_cat_states(
            states, keep_indexs, get_phase=True
        )
    else:
        attn_bias, x_cat = get_attn_bias_and_cat_states(states, keep_indexs, get_phase=False)

    # 3) apply residual_func to get residual, and split the result
    if require_phase:
        residual_list = attn_bias.split(residual_func(x_cat, phase_cos_cat, phase_sin_cat, attn_bias=attn_bias))
    else:
        if checkpoint_residual_func:
            residual_list = attn_bias.split(
                torch.utils.checkpoint.checkpoint(
                    residual_func,
                    x_cat,
                    attn_bias,
                    use_reentrant=False,
                )
            )
        else:
            residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    #outputs = []
    for state, x, keep_index, residual, residual_scale_factor in zip(
        states, x_list, keep_indexs, residual_list, residual_scale_factors
    ):
        #outputs.append(
        updated = add_residual(
            x, keep_index, residual, residual_scale_factor, scaling_vector
        ).view_as(x)
        state.tokens = updated
        #)
    #return outputs


class KoPENestedTensorBlock(KoPEBlock):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supports_state_list = True

    def forward(
        self,
        states: Union["KoPEState", List["KoPEState"]],
    ) -> Union["KoPEState", List["KoPEState"]]:
        if isinstance(states, list):
            if not states:
                return states
            if not attention_ops.XFORMERS_AVAILABLE:
                raise RuntimeError("KoPE nested blocks require xFormers to be installed")

            if self.training and self.sample_drop_ratio > 0.0:
                self._forward_states_stochastic(states)
            else:
                self._forward_states_deterministic(states)
            return states

        return super().forward(states)

    def _forward_states_deterministic(self, states: List[KoPEState]) -> None:
        attn_bias, tokens_cat, phase_cos_cat, phase_sin_cat = get_attn_bias_and_cat_states(states, get_phase=True)

        attn_residual = self.ls1(
            self.attn(
                self.norm1(tokens_cat),
                phase_cos_cat,
                phase_sin_cat,
                attn_bias=attn_bias,
            )
        )
        tokens_cat = tokens_cat + attn_residual

        if self.checkpoint_ffn_func and self.training:
            mlp_residual = torch.utils.checkpoint.checkpoint(
                lambda _t: self.ls2(self.mlp(self.norm2(_t))),
                tokens_cat,
                use_reentrant=False,
            )
        else:
            mlp_residual = self.ls2(self.mlp(self.norm2(tokens_cat)))
        tokens_cat = tokens_cat + mlp_residual

        next_cos_cat, next_sin_cat = self._update_phase(
            tokens_cat,
            phase_cos_cat,
            phase_sin_cat,
            attn_bias=attn_bias,
        )

        token_list = attn_bias.split(tokens_cat)
        next_cos_list = attn_bias.split(next_cos_cat)
        next_sin_list = attn_bias.split(next_sin_cat)
        for state, token, next_cos, next_sin in zip(
            states,
            token_list,
            next_cos_list,
            next_sin_list,
        ):
            state.tokens = token.view_as(state.tokens)
            state.phase_cos = next_cos.view_as(state.phase_cos)
            state.phase_sin = next_sin.view_as(state.phase_sin)

    def _forward_states_stochastic(self, states: List[KoPEState]) -> None:
        def attn_residual_func(x_cat: torch.Tensor, phase_cos_cat: torch.Tensor, phase_sin_cat: torch.Tensor, attn_bias: Optional[object] = None) -> torch.Tensor:
            return self.attn(
                    self.norm1(x_cat),
                    phase_cos_cat,
                    phase_sin_cat,
                    attn_bias=attn_bias,
                )

        def ffn_residual_func(x_cat: torch.Tensor, attn_bias: Optional[object] = None) -> torch.Tensor:
            return self.mlp(self.norm2(x_cat))

        scaling_vector = getattr(self.ls1, "gamma", None)
        drop_add_residual_stochastic_depth_list_states(
            states,
            attn_residual_func,
            sample_drop_ratio=self.sample_drop_ratio,
            scaling_vector=scaling_vector,
            require_phase=True,
        )
        scaling_vector_ffn = getattr(self.ls2, "gamma", None)
        drop_add_residual_stochastic_depth_list_states(
            states,
            ffn_residual_func,
            sample_drop_ratio=self.sample_drop_ratio,
            scaling_vector=scaling_vector_ffn,
            require_phase=False,
            checkpoint_residual_func=self.checkpoint_ffn_func and self.training,
        )

        # Phase update
        attn_bias, tokens_cat, phase_cos_cat, phase_sin_cat = get_attn_bias_and_cat_states(states, get_phase=True)
        next_cos_cat, next_sin_cat = self._update_phase(
            tokens_cat,
            phase_cos_cat,
            phase_sin_cat,
            attn_bias=attn_bias,
        )

        next_cos_list = attn_bias.split(next_cos_cat)
        next_sin_list = attn_bias.split(next_sin_cat)
        for state, next_cos, next_sin in zip(
            states,
            next_cos_list,
            next_sin_list,
        ):
            state.phase_cos = next_cos.view_as(state.phase_cos)
            state.phase_sin = next_sin.view_as(state.phase_sin)


class DinoVisionTransformerKoPE(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        ffn_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        layerscale: Optional[float] = None,
        embed_layer: Callable[..., nn.Module] = PatchEmbed,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        ffn_layer: str = "mlp",
        block_chunks: int = 1,
        block_type: str = "nested",
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
        drop_masks: bool = False,
        gradient_checkpointing: bool = False,
        qknorm: bool = False,
        max_resolution: int = 14,
        base: int = 20,
        kope_gamma: float = 0.05,
        update_ext_token_phase: bool = True,
        use_learnable_pos_embed: bool = True,
        share_kope_coupling: bool = True,
        coupling_use_vo_proj: bool = False,
        coupling_use_tanh: bool = False,
        kope_vo_rotation: bool = True,
        fixed_layout: Optional[bool] = True,
        coupling_qknorm: bool = True,
        coupling_qknorm_learn: bool = True,
        coupling_qk_multilayer: bool = False,
        start_kope_idx: int = 0,
        kope_mix: bool = True,
        kope_mix_init_gain: float = 0.1,
        kope_mix_phase_norm: bool = True,
        checkpoint_phase_update: bool = True,
        checkpoint_ffn_func: bool = True,
        checkpoint_rotation: bool = True,
        no_phase_norm: bool = True,
        learn_phase_gamma: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            logger.warning("KoPE ViT received unused kwargs: %s", list(kwargs.keys()))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.gradient_checkpointing = gradient_checkpointing
        self.drop_masks = drop_masks
        self.ext_token_num = 1 + num_register_tokens
        self.block_type = block_type
        self.update_ext_token_phase = update_ext_token_phase
        self.use_learnable_pos_embed = use_learnable_pos_embed
        self.share_kope_coupling = share_kope_coupling
        self.coupling_use_vo_proj = coupling_use_vo_proj
        self.coupling_use_tanh = coupling_use_tanh
        self.kope_vo_rotation = kope_vo_rotation
        # Keep as top-level attributes for compatibility with training scripts/configs.
        self.kope_mix = kope_mix
        self.kope_mix_init_gain = kope_mix_init_gain
        self.kope_mix_phase_norm = kope_mix_phase_norm
        if fixed_layout is None:
            fixed_layout = not coupling_use_vo_proj
        self.fixed_layout = fixed_layout
        self.coupling_qknorm = coupling_qknorm
        self.coupling_qknorm_learn = coupling_qknorm_learn
        self.coupling_qk_multilayer = coupling_qk_multilayer
        self.start_kope_idx = start_kope_idx
        self.checkpoint_phase_update = checkpoint_phase_update
        self.checkpoint_ffn_func = checkpoint_ffn_func
        self.checkpoint_rotation = checkpoint_rotation
        self.no_phase_norm = no_phase_norm
        self.learn_phase_gamma = learn_phase_gamma

        head_dim = embed_dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError("KoPE requires head dimension divisible by 4")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32) / (head_dim // 2))
        )
        self.register_buffer("_phase_inv_freq", inv_freq, persistent=False)

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        if self.use_learnable_pos_embed:
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.empty(1, num_patches + 1, embed_dim))
            self.raw_pos_embed = None
        else:
            self.pos_embed = None
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.empty(1, num_register_tokens, embed_dim)
            )
        else:
            self.register_tokens = None
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self._coords_cache: dict = {}

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            logger.info("KoPE ViT using MLP FFN")
            ffn_layer_cls = Mlp
        elif ffn_layer in {"swiglufused", "swiglu"}:
            logger.info("KoPE ViT using SwiGLU FFN")
            ffn_layer_cls = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("KoPE ViT using Identity FFN")

            def _identity_ffn(*args, **unused):
                return nn.Identity()

            ffn_layer_cls = _identity_ffn
        else:
            raise NotImplementedError(f"Unknown FFN layer: {ffn_layer}")

        if block_type not in {"base", "nested"}:
            raise ValueError(f"Unknown block_type '{block_type}' for KoPE ViT")
        #if block_type == "nested" and not attention_ops.XFORMERS_AVAILABLE:
        #    raise RuntimeError("KoPE nested blocks require xFormers but it is not available")

        block_cls = KoPENestedTensorBlock if block_type == "nested" else KoPEBlock

        shared_phase_coupling: Optional[KoPECoupling] = None
        if self.share_kope_coupling:
            shared_phase_coupling = KoPECoupling(
                dim=embed_dim,
                num_heads=num_heads,
                ext_token_num=self.ext_token_num,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=ffn_drop,
                coupling_use_vo_proj=coupling_use_vo_proj,
                fixed_layout=self.fixed_layout,
                coupling_use_tanh=coupling_use_tanh,
                qknorm=coupling_qknorm,
                qknorm_learn=coupling_qknorm_learn,
                coupling_qk_multilayer=coupling_qk_multilayer,
            )
            self.shared_phase_coupling = shared_phase_coupling
            if self.learn_phase_gamma:
                self.shared_phase_coupling_gamma = PhaseStep(init=kope_gamma)
            else:
                self.shared_phase_coupling_gamma = None
        else:
            self.shared_phase_coupling = None
            self.shared_phase_coupling_gamma = None

        blocks_list: List[nn.Module] = []
        for i in range(depth):
            blocks_list.append(
                block_cls(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    ffn_drop=ffn_drop,
                    attn_drop=attn_drop,
                    layerscale=layerscale,
                    drop_path=dpr[i],
                    qknorm=qknorm,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    ffn_layer=ffn_layer_cls,
                    ext_token_num=self.ext_token_num,
                    phase_gamma=kope_gamma,
                    update_ext_token_phase=update_ext_token_phase,
                    phase_coupling_fn=(
                        # TODO: check
                        shared_phase_coupling.forward if shared_phase_coupling is not None else None
                    ),
                    coupling_use_vo_proj=coupling_use_vo_proj,
                    coupling_use_tanh=self.coupling_use_tanh,
                    kope_vo_rotation=self.kope_vo_rotation,
                    fixed_layout=self.fixed_layout,
                    coupling_qknorm=self.coupling_qknorm,
                    coupling_qknorm_learn=self.coupling_qknorm_learn,
                    coupling_qk_multilayer=coupling_qk_multilayer,
                    use_kope_attn=(i >= self.start_kope_idx),
                    kope_mix=self.kope_mix,
                    kope_mix_init_gain=self.kope_mix_init_gain,
                    kope_mix_phase_norm=self.kope_mix_phase_norm,
                    checkpoint_phase_update=self.checkpoint_phase_update,
                    checkpoint_ffn_func=self.checkpoint_ffn_func,
                    checkpoint_rotation=self.checkpoint_rotation,
                    no_phase_norm=self.no_phase_norm,
                    learn_phase_gamma=self.learn_phase_gamma,
                    shared_phase_coupling_gamma=self.shared_phase_coupling_gamma.forward if self.shared_phase_coupling_gamma is not None else None,
                )
            )

        if block_chunks > 0:
            chunked_blocks: List[nn.Module] = []
            chunk_size = max(depth // block_chunks, 1)
            for start in range(0, depth, chunk_size):
                prefix = [KoPEIdentity()] * start
                chunk = blocks_list[start : start + chunk_size]
                chunked_blocks.append(KoPEBlockChunk(prefix + chunk))
            self.blocks = nn.ModuleList(chunked_blocks)
            self.chunked_blocks = True
        else:
            self.blocks = nn.ModuleList(blocks_list)
            self.chunked_blocks = False

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        if self.use_learnable_pos_embed and self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        named_apply(init_weights_vit_timm, self)

    def update_img_size(self, img_size, keep_raw: bool = False) -> None:
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
            patch_pos_embed = F.interpolate(
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

    def update_patch_size(self, patch_size, keep_raw: bool = False) -> None:
        if patch_size == self.patch_size:
            return
        self.patch_embed.update_patch_size(patch_size, keep_raw)
        self.patch_size = patch_size
        self._coords_cache.clear()

    @torch._dynamo.disable
    def interpolate_pos_encoding(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if not self.use_learnable_pos_embed or self.pos_embed is None:
            raise RuntimeError("Learnable positional embeddings are disabled for this model")
        previous_dtype = tokens.dtype
        npatch = tokens.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and H == W:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]
        dim = self.pos_embed.shape[-1]
        if isinstance(self.patch_size, (tuple, list)):
            patch_h, patch_w = self.patch_size if len(self.patch_size) == 2 else (self.patch_size[0], self.patch_size[0])
        else:
            patch_h = patch_w = self.patch_size
        w0 = W // patch_w
        h0 = H // patch_h
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)

    def _get_patch_coords(
        self, height: int, width: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
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

    def prepare_tokens_with_masks(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        tokens = self.patch_embed(x)
        feat_h = H // self.patch_size
        feat_w = W // self.patch_size
        coords = self._get_patch_coords(feat_h, feat_w, B, tokens.device)

        keep_mask = None
        if masks is not None:
            if not self.drop_masks:
                mask_token = self.mask_token.to(dtype=tokens.dtype).unsqueeze(0)
                tokens = torch.where(masks.unsqueeze(-1), mask_token, tokens)
            else:
                keep_mask = (~masks)

        tokens = torch.cat((self.cls_token.expand(B, -1, -1), tokens), dim=1)
        if self.use_learnable_pos_embed and self.pos_embed is not None:
            tokens = tokens + self.interpolate_pos_encoding(tokens, H, W)

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

    def _initialize_phase(
        self, coords: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_components, sin_components = attention_ops.build_2d_rope_from_coords(
            #coords.to(device=device, dtype=torch.float32), self._phase_inv_freq
            coords[:1].to(device=device, dtype=torch.float32), self._phase_inv_freq
        )
        cos_components = cos_components.to(device=device, dtype=dtype)
        sin_components = sin_components.to(device=device, dtype=dtype)
        # collapse trig components into paired form: [1, P, D] -> [1, P, D/2]
        base_cos = cos_components.view(cos_components.shape[0], cos_components.shape[1], -1, 2)[..., 0]
        base_sin = sin_components.view(sin_components.shape[0], sin_components.shape[1], -1, 2)[..., 0]
        half_dim = base_cos.shape[-1] if base_cos.numel() > 0 else self.embed_dim // self.num_heads // 2
        identity_cos_base = torch.ones(
            #(coords.shape[0], self.ext_token_num, half_dim),
            (1, self.ext_token_num, half_dim),
            dtype=dtype,
            device=device,
        )
        identity_sin_base = torch.zeros_like(identity_cos_base)
        if base_cos.numel() > 0:
            cos_base = torch.cat((identity_cos_base, base_cos), dim=1)
            sin_base = torch.cat((identity_sin_base, base_sin), dim=1)
        else:
            cos_base = identity_cos_base
            sin_base = identity_sin_base

        phase_bias_cos = cos_base.unsqueeze(2)
        phase_bias_sin = sin_base.unsqueeze(2)
        phase_cos = phase_bias_cos.expand(coords.shape[0], -1, self.num_heads, -1).clone()
        phase_sin = phase_bias_sin.expand(coords.shape[0], -1, self.num_heads, -1).clone()
        return phase_cos, phase_sin

    def _forward_through_blocks(
        self,
        state: KoPEState,
    ) -> KoPEState:
        for idx, blk in enumerate(self.blocks):
            if self.gradient_checkpointing: # and self.training:

                def run_block(
                    tokens: torch.Tensor,
                    phase_cos: torch.Tensor,
                    phase_sin: torch.Tensor,
                    module=blk,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    input_state = KoPEState(tokens, phase_cos, phase_sin)
                    output_state = module(input_state)
                    return (
                        output_state.tokens,
                        output_state.phase_cos,
                        output_state.phase_sin,
                    )

                tensors = torch.utils.checkpoint.checkpoint(
                    run_block,
                    state.tokens,
                    state.phase_cos,
                    state.phase_sin,
                    use_reentrant=False,
                )
                state = KoPEState(*tensors)
            else:
                state = blk(state)
        return state

    def _forward_through_blocks_states(
        self,
        states: List[KoPEState],
    ) -> List[KoPEState]:
        for idx, blk in enumerate(self.blocks):
            if self.gradient_checkpointing: # and self.training:
                def run_block(*args):
                    current_states = []
                    for i in range(0, len(args), 3):
                        current_states.append(KoPEState(args[i], args[i+1], args[i+2]))

                    if getattr(blk, "supports_state_list", False):
                        result = blk(current_states)
                        if isinstance(result, list):
                            out_states = result
                        else:
                            out_states = [result]
                    else:
                        out_states = [blk(s) for s in current_states]

                    out_tensors = []
                    for s in out_states:
                        out_tensors.extend([s.tokens, s.phase_cos, s.phase_sin])
                    return tuple(out_tensors)

                flat_inputs = []
                for s in states:
                    flat_inputs.extend([s.tokens, s.phase_cos, s.phase_sin])

                outputs = torch.utils.checkpoint.checkpoint(
                    run_block,
                    *flat_inputs,
                    use_reentrant=False,
                )

                new_states = []
                for i in range(0, len(outputs), 3):
                    new_states.append(KoPEState(outputs[i], outputs[i+1], outputs[i+2]))
                states = new_states
            elif getattr(blk, "supports_state_list", False):
                result = blk(states)
                if isinstance(result, list):
                    states = result
                else:
                    states = [result]
            else:
                states = [blk(state) for state in states]
        return states

    def _forward_features_single(
        self, x: torch.Tensor, masks: Optional[torch.Tensor]
    ) -> dict:
        tokens, coords = self.prepare_tokens_with_masks(x, masks)
        phase_cos, phase_sin = self._initialize_phase(
            coords, tokens.dtype, tokens.device
        )
        state = KoPEState(tokens, phase_cos, phase_sin)
        state = self._forward_through_blocks(state)
        x_norm = self.norm(state.tokens)
        patch_phase_cos = state.phase_cos[:, self.ext_token_num :]
        patch_phase_sin = state.phase_sin[:, self.ext_token_num :]
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": state.tokens,
            #"kope_phase_cos": state.phase_cos,
            #"kope_phase_sin": state.phase_sin,
            "kope_patch_phase_cos": patch_phase_cos,
            "kope_patch_phase_sin": patch_phase_sin,
            "masks": masks,
        }

    def _forward_features_multicrop(
        self,
        x_list: Sequence[torch.Tensor],
        masks_list: Sequence[Optional[torch.Tensor]],
    ) -> List[dict]:
        states: List[KoPEState] = []
        for x, masks in zip(x_list, masks_list):
            tokens, coords = self.prepare_tokens_with_masks(x, masks)
            phase_cos, phase_sin = self._initialize_phase(coords, tokens.dtype, tokens.device)
            states.append(
                KoPEState(
                    tokens=tokens,
                    phase_cos=phase_cos,
                    phase_sin=phase_sin,
                )
            )

        states = self._forward_through_blocks_states(states)

        outputs: List[dict] = []
        for state, masks in zip(states, masks_list):
            x_norm = self.norm(state.tokens)
            outputs.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": state.tokens,
                    #"kope_phase_cos": state.phase_cos,
                    #"kope_phase_sin": state.phase_sin,
                    "kope_patch_phase_cos": state.phase_cos[:, self.ext_token_num :],
                    "kope_patch_phase_sin": state.phase_sin[:, self.ext_token_num :],
                    "masks": masks,
                }
            )
        return outputs

    def _iter_block_modules(self) -> Sequence[nn.Module]:
        if self.chunked_blocks:
            modules: List[nn.Module] = []
            for chunk in self.blocks:
                modules.extend(list(chunk))
            return modules
        return list(self.blocks)

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence[int]] = 1,
        *,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
        # TODO
        only_return_phase: bool = False,
        return_token_and_phase: bool = False,
        return_rotated_token: bool = False,
        return_token_x_phase: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if isinstance(n, int):
            if n <= 0 or n > self.n_blocks:
                raise ValueError(f"Requested {n} layers, but model has {self.n_blocks}")
            target_indices = list(range(self.n_blocks - n, self.n_blocks))
        else:
            target_indices = list(n)
            if not target_indices:
                raise ValueError("At least one layer index must be provided")
            invalid = [idx for idx in target_indices if idx < 0 or idx >= self.n_blocks]
            if invalid:
                raise ValueError(f"Layer indices out of range for KoPE model: {invalid}")

        target_set = set(target_indices)

        tokens, coords = self.prepare_tokens_with_masks(x, masks=None)
        phase_cos, phase_sin = self._initialize_phase(
            coords, tokens.dtype, tokens.device
        )
        state = KoPEState(tokens, phase_cos, phase_sin)

        collected: List[torch.Tensor] = []
        block_outputs: Dict[int, torch.Tensor] = {}
        block_idx = -1
        for module in self._iter_block_modules():
            if isinstance(module, (KoPEBlock, KoPENestedTensorBlock)):
                block_idx += 1
            state = module(state)  # type: ignore[assignment]
            if isinstance(module, KoPEIdentity):
                continue
            if not isinstance(module, (KoPEBlock, KoPENestedTensorBlock)):
                continue
            if block_idx in target_set:
                if only_return_phase:
                    block_outputs[block_idx] = torch.cat((state.phase_cos.flatten(-2), state.phase_sin.flatten(-2)), dim=-1)
                elif return_token_and_phase:
                    if norm:
                        tokens = self.norm(state.tokens)
                    else:
                        tokens = state.tokens
                    block_outputs[block_idx] = torch.cat(
                        (tokens, state.phase_cos.flatten(-2), state.phase_sin.flatten(-2)), dim=-1
                    )
                elif return_rotated_token:
                    if norm:
                        tokens = self.norm(state.tokens)
                    else:
                        tokens = state.tokens
                    tokens = tokens.view(state.tokens.shape[0], state.tokens.shape[1], self.num_heads, -1)
                    cos_heads = state.phase_cos.to(dtype=tokens.dtype)
                    sin_heads = state.phase_sin.to(dtype=tokens.dtype)
                    rotated_token = apply_rotary_pairs(tokens, cos_heads, sin_heads)
                    rotated_token = rotated_token.view_as(state.tokens)
                    ## TODO norm before or after rotary?
                    #if norm:
                    #    rotated_token = self.norm(rotated_token)
                    block_outputs[block_idx] = rotated_token
                elif return_token_x_phase:
                    if norm:
                        tokens = self.norm(state.tokens)
                    else:
                        tokens = state.tokens
                    tokens = tokens.view(state.tokens.shape[0], state.tokens.shape[1], self.num_heads, self.embed_dim // self.num_heads // 2, 2)
                    cos_heads = state.phase_cos.to(dtype=state.tokens.dtype).unsqueeze(-1)
                    sin_heads = state.phase_sin.to(dtype=state.tokens.dtype).unsqueeze(-1)
                    tokens = torch.stack((tokens * cos_heads, tokens * sin_heads), dim=-1).flatten(2)
                    block_outputs[block_idx] = tokens
                else:
                    if norm:
                        tokens = self.norm(state.tokens)
                    else:
                        tokens = state.tokens
                    block_outputs[block_idx] = tokens

        if len(block_outputs) != len(target_indices):
            missing = sorted(set(target_indices) - set(block_outputs.keys()))
            raise RuntimeError(f"Failed to collect KoPE intermediate layers for indices: {missing}")

        for idx in target_indices:
            collected.append(block_outputs[idx])

#        if norm:
#            collected = [self.norm(out) for out in collected]

        class_tokens = [out[:, 0] for out in collected]
        patch_tokens = [out[:, 1 + self.num_register_tokens :] for out in collected]

        if reshape:
            B, _, H, W = x.shape
            if isinstance(self.patch_size, tuple):
                patch_h, patch_w = self.patch_size
            else:
                patch_h = patch_w = self.patch_size
            feat_h = H // patch_h
            feat_w = W // patch_w
            patch_tokens = [
                out.reshape(B, feat_h, feat_w, -1).permute(0, 3, 1, 2).contiguous()
                for out in patch_tokens
            ]

        if return_class_token:
            return tuple(zip(patch_tokens, class_tokens))

        return tuple(patch_tokens)

    def forward_features(
        self,
        x: Union[torch.Tensor, Sequence[torch.Tensor]],
        masks: Optional[Union[torch.Tensor, Sequence[Optional[torch.Tensor]]]] = None,
    ) -> Union[dict, List[dict]]:
        if isinstance(x, list):
            if masks is None:
                masks = [None] * len(x)
            assert isinstance(masks, Sequence)
            return self._forward_features_multicrop(x, masks)
        else:
            assert not isinstance(masks, list)
            return self._forward_features_single(x, masks)

    def forward(
        self,
        *args,
        is_training: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], dict, List[dict]]:
        feats = self.forward_features(*args, **kwargs)
        if is_training:
            return feats
        if isinstance(feats, list):
            return [self.head(item["x_norm_clstoken"]) for item in feats]
        return self.head(feats["x_norm_clstoken"])

    def set_analysis_mode(self, mode: bool) -> None:
        for m in self.modules():
            if hasattr(m, "enable_analysis"):
                m.enable_analysis = mode

    def get_analysis_results(self) -> Dict[str, float]:
        results = {}
        layer_metrics = []

        # Iterate over blocks to get metrics in order
        for i, block in enumerate(self.blocks):
            # Handle BlockChunk
            if isinstance(block, nn.ModuleList): # KoPEBlockChunk
                for sub_block in block:
                    if hasattr(sub_block, "attn") and hasattr(sub_block.attn, "last_metrics"):
                        if sub_block.attn.last_metrics:
                            layer_metrics.append(sub_block.attn.last_metrics)
            elif hasattr(block, "attn") and hasattr(block.attn, "last_metrics"):
                if block.attn.last_metrics:
                    layer_metrics.append(block.attn.last_metrics)

        if not layer_metrics:
            return {}

        # Aggregate
        keys = layer_metrics[0].keys()
        for key in keys:
            values = [m[key] for m in layer_metrics]
            avg_val = sum(values) / len(values)
            results[f"avg_{key}"] = avg_val
            for i, val in enumerate(values):
                results[f"layer_{i:02d}_{key}"] = val

        return results

def vit_kope_small(
    patch_size: int = 16,
    num_register_tokens: int = 0,
    block_type: str = "base",
    **kwargs,
) -> DinoVisionTransformerKoPE:
    return DinoVisionTransformerKoPE(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        block_type=block_type,
        **kwargs,
    )


def vit_kope_base(
    patch_size: int = 16,
    num_register_tokens: int = 0,
    block_type: str = "base",
    **kwargs,
) -> DinoVisionTransformerKoPE:
    return DinoVisionTransformerKoPE(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        block_type=block_type,
        **kwargs,
    )


def vit_kope_large(
    patch_size: int = 16,
    num_register_tokens: int = 0,
    block_type: str = "base",
    **kwargs,
) -> DinoVisionTransformerKoPE:
    return DinoVisionTransformerKoPE(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        block_type=block_type,
        **kwargs,
    )


def vit_kope_giant(
    patch_size: int = 16,
    num_register_tokens: int = 0,
    block_type: str = "base",
    **kwargs,
) -> DinoVisionTransformerKoPE:
    return DinoVisionTransformerKoPE(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        block_type=block_type,
        **kwargs,
    )


if __name__ == "__main__":
    models = [
        ("vit_kope_small", vit_kope_small),
        ("vit_kope_base", vit_kope_base),
        ("vit_kope_large", vit_kope_large),
        ("vit_kope_giant", vit_kope_giant),
    ]
    try:
        from prettytable import PrettyTable  # type: ignore[import]

        table = PrettyTable(["Model", "Params", "Embed", "Depth", "Heads", "MLP Ratio"])
        for name, builder in models:
            model = builder()
            table.add_row(
                [
                    name,
                    sum(p.numel() for p in model.parameters()),
                    model.embed_dim,
                    model.n_blocks,
                    model.num_heads,
                    model.mlp_ratio,
                ]
            )
        table._rows.sort(key=lambda x: x[1])
        print(table)
    except ImportError:
        for name, builder in models:
            model = builder()
            print(
                f"Model: {name}, Params: {sum(p.numel() for p in model.parameters())}, "
                f"Embed: {model.embed_dim}, Depth: {model.n_blocks}, Heads: {model.num_heads}, MLP Ratio: {model.mlp_ratio}"
            )