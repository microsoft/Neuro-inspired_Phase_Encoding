""" KoPE Vision Transformer for CLIP.

Adapted from https://github.com/mlfoundations/open_clip (transformer.py).
Modifications: integrate Kuramoto Oscillatory Phase Encoding (KoPE).
"""
from collections import OrderedDict
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from xformers.ops import memory_efficient_attention, unbind
    XFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XFORMERS_AVAILABLE = False

from .utils import to_2tuple, feature_take_indices
from .pos_embed import get_2d_sincos_pos_embed


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(
            self,
            prob: float = 0.5,
            exclude_first_token: bool = True
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            scaled_cosine: bool = False,
            scale_heads: bool = False,
            inner_norm: bool = False,
            logit_scale_max: float = math.log(1. / 0.01),
            norm_layer: Type[nn.Module] = LayerNormFp32,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            # KoPE-related
            enable_kope: bool = False,
            kope_vo_rotation: bool = True,
            kope_mix: bool = True,
            kope_mix_init_gain: float = 0.1,
            kope_mix_phase_norm: bool = True,
            checkpoint_rotation: bool = False,
    ):
        super().__init__()
        assert not (scaled_cosine and qk_norm), "Cannot activate both scaled cosine and QK normalization"
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        self.use_fsdpa = hasattr(nn.functional, 'scaled_dot_product_attention')

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.empty((dim * 3, dim)))
        nn.init.xavier_uniform_(self.in_proj_weight)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        # QK normalization (with LN) from https://arxiv.org/abs/2106.04560 and related to other QK Norm ideas
        if qk_norm:
            self.ln_q = norm_layer(self.head_dim)
            self.ln_k = norm_layer(self.head_dim)
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()

        # Scaled cosine attention (from Swin Transformer V2, https://arxiv.org/abs/2111.09883)
        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None

        self.attn_drop = nn.Dropout(attn_drop)

        # Per-head attention logit scaling (from NormFormer, https://arxiv.org/abs/2110.09456)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None

        # Normalization of attention logits, before final projection.
        # Origin likely Sub-LN in (Foundation Transformers, https://arxiv.org/abs/2210.06423)
        if inner_norm:
            self.ln_inner = norm_layer(dim)
        else:
            self.ln_inner = nn.Identity()

        self.out_proj = nn.Linear(dim, dim)
        if self.out_proj.bias is not None:
             nn.init.zeros_(self.out_proj.bias)
        self.out_drop = nn.Dropout(proj_drop)

        # KoPE-related
        self.enable_kope = enable_kope
        self.kope_vo_rotation = kope_vo_rotation
        self.kope_mix = kope_mix
        self.kope_mix_init_gain = kope_mix_init_gain
        self.kope_mix_phase_norm = kope_mix_phase_norm
        self.checkpoint_rotation = checkpoint_rotation

        if self.enable_kope and self.kope_mix:
            rotary_dim = self.head_dim // 2
            if rotary_dim == 0:
                raise ValueError("KoPE mix requires head_dim // 2 > 0")
            self.kope_mix_coef = nn.Parameter(torch.empty(self.num_heads, rotary_dim, rotary_dim))
            self._reset_kope_mix_parameters()
        else:
            self.register_parameter("kope_mix_coef", None)

    def _reset_kope_mix_parameters(self) -> None:
        gain = float(self.kope_mix_init_gain)
        with torch.no_grad():
            nn.init.xavier_normal_(self.kope_mix_coef, gain=gain)
            rotary_dim = self.kope_mix_coef.shape[-1]
            eye = torch.eye(rotary_dim, device=self.kope_mix_coef.device, dtype=self.kope_mix_coef.dtype)
            self.kope_mix_coef.add_(eye.unsqueeze(0))

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None, phase_state: Optional[torch.Tensor] = None):
        N, L, C = x.shape

        if self.enable_kope:
            assert phase_state is not None
            # phase_state is (B, L, H, D//2, 2)

            def compute_qkv_rotated(x, phase_state):
                q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
                # N, L, H, D
                q = q.reshape(N, L, self.num_heads, -1)
                k = k.reshape(N, L, self.num_heads, -1)
                v = v.reshape(N, L, self.num_heads, -1)

                phase_state = phase_state.to(dtype=x.dtype)

                if self.kope_mix:
                    if not self.kope_mix_phase_norm:
                        mix = F.normalize(self.kope_mix_coef.to(dtype=phase_state.dtype), p=2, dim=-2, eps=1e-6)
                    else:
                        mix = self.kope_mix_coef.to(dtype=phase_state.dtype)

                    # phase_state: (B, L, H, D, 2)
                    # mix: (H, D, D)
                    # Contract on D dimension: d -> e
                    phase_state = torch.einsum('blhdi,hde->blhei', phase_state, mix)

                    if self.kope_mix_phase_norm:
                        sum_sq = phase_state.square().sum(dim=-1, keepdim=True)
                        inv = torch.rsqrt(torch.clamp(sum_sq, min=1e-5))
                        phase_state = phase_state * inv

                cos, sin = phase_state.unbind(-1)

                q = apply_rotary_pairs(q, cos, sin)
                k = apply_rotary_pairs(k, cos, sin)
                if self.kope_vo_rotation:
                    v = apply_rotary_pairs(v, cos, sin)

                # N, H, L, D
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                return q, k, v, phase_state

            if self.checkpoint_rotation and x.requires_grad:
                q, k, v, phase_state_final = torch.utils.checkpoint.checkpoint(
                    compute_qkv_rotated, x, phase_state, use_reentrant=False
                )
            else:
                q, k, v, phase_state_final = compute_qkv_rotated(x, phase_state)

            cos_heads, sin_heads = phase_state_final.unbind(-1)
        else:
            q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
            # N, H, L, D
            q = q.reshape(N, L, self.num_heads, -1).transpose(1, 2)
            k = k.reshape(N, L, self.num_heads, -1).transpose(1, 2)
            v = v.reshape(N, L, self.num_heads, -1).transpose(1, 2)

        if attn_mask is not None:
            if attn_mask.ndim == 3:
                # this module works with (L, L), or (N, num_heads, L, L) masks
                attn_mask = attn_mask.reshape(N, self.num_heads, L, L)
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            else:
                attn_mask = attn_mask.to(dtype=q.dtype)

        if self.logit_scale is not None:
            attn = torch.bmm(
                F.normalize(q, dim=-1),
                F.normalize(k, dim=-1).transpose(-1, -2)
            )
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn * logit_scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            q = self.ln_q(q)
            k = self.ln_k(k)
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        # N, num_heads, L, head_dim
        if self.head_scale is not None:
            x = x * self.head_scale
        if self.enable_kope and self.kope_vo_rotation:
            x = x.transpose(1, 2)
            if self.checkpoint_rotation:
                x = torch.utils.checkpoint.checkpoint(
                    apply_rotary_pairs,
                    x,
                    cos_heads,
                    -sin_heads,
                    use_reentrant=False,
                )
            else:
                x = apply_rotary_pairs(
                    x,
                    cos_heads,
                    -sin_heads,
                )
            x = x.reshape(N, L, C)
        else:
            x = x.transpose(1, 2).reshape(N, L, C)
        x = self.ln_inner(x)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


# === KoPE helpers ===

def apply_rotary_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.size(-1) % 2 != 0:
        raise ValueError("Rotary embedding requires even feature dimension")

    d = x.size(-1)
    half = d // 2
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)

    if cos.shape[-1] != half:
        raise ValueError(f"Mismatched rotary pair dimension: x={x.shape}, cos={cos.shape}")

    x_view = x.view(*x.shape[:-1], half, 2)
    real, imag = x_view.unbind(-1)

    updated_real = real * cos - imag * sin
    updated_imag = real * sin + imag * cos
    return torch.stack((updated_real, updated_imag), dim=-1).flatten(-2)


def build_2d_rope_from_coords(coords: torch.Tensor, inv_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # N, L, 2
    coords_fp32 = coords.to(dtype=torch.float32)
    # D//4
    inv_freq_fp32 = inv_freq.to(dtype=torch.float32)
    row_angles = coords_fp32[..., 0].unsqueeze(-1) * inv_freq_fp32
    col_angles = coords_fp32[..., 1].unsqueeze(-1) * inv_freq_fp32
    # N, L, D//2
    cos = torch.cat((torch.cos(row_angles), torch.cos(col_angles)), dim=-1)
    sin = torch.cat((torch.sin(row_angles), torch.sin(col_angles)), dim=-1)
    return cos, sin


@dataclass
class KoPEState:
    tokens: torch.Tensor
    phase_state: torch.Tensor


class PhaseStep(nn.Module):
    def __init__(self, init: float = 0.01, max_value: float = 1.0):
        super().__init__()
        self._raw = nn.Parameter(torch.log(torch.expm1(torch.tensor([init], dtype=torch.float32))))
        self.max_value = max_value

    def forward(self) -> torch.Tensor:
        return torch.clamp(F.softplus(self._raw), max=self.max_value)


class KoPECoupling(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        coupling_use_vo_proj: bool = False,
        coupling_use_tanh: bool = False,
        qknorm: bool = True,
        qknorm_learn: bool = True,
        coupling_qk_multilayer: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.coupling_use_vo_proj = coupling_use_vo_proj
        self.coupling_use_tanh = coupling_use_tanh
        self.qknorm = qknorm
        self.coupling_qk_multilayer = coupling_qk_multilayer

        if self.qknorm and qknorm_learn:
            self.qk_scale = nn.Parameter(torch.ones(1, 1, self.num_heads, self.head_dim))
        else:
            self.register_parameter("qk_scale", None)

        if self.coupling_qk_multilayer:
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

        if self.coupling_use_vo_proj:
            self.v_proj = nn.Linear(dim, dim, bias=proj_bias)
            self.o_proj = nn.Linear(dim, dim, bias=proj_bias)
        else:
            self.v_proj = None
            self.o_proj = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.output_dropout = nn.Dropout(proj_drop)

    def forward(
        self,
        tokens: torch.Tensor,
        phase_state: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        update_phase: bool = False,
        gamma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, head_count, half_dim, _ = phase_state.shape
        phase_head = phase_state.flatten(-2)

        q_full = self.q_proj(tokens).view(B, N, self.num_heads, self.head_dim)
        k_full = self.k_proj(tokens).view(B, N, self.num_heads, self.head_dim)

        if self.qknorm:
            q_full = F.rms_norm(q_full, (q_full.size(-1),))
            k_full = F.rms_norm(k_full, (k_full.size(-1),))
            if self.qk_scale is not None:
                q_full = q_full * self.qk_scale.to(dtype=q_full.dtype)

        if self.coupling_use_vo_proj:
            v_tokens = self.v_proj(phase_head.reshape(B, N, self.dim))
            v_full = v_tokens.view(B, N, self.num_heads, self.head_dim)
        else:
            v_full = phase_head.to(dtype=q_full.dtype)

        q = q_full
        k = k_full
        v = v_full

        if attn_bias is not None:
            if attn_bias.ndim == 3:
                attn_bias = attn_bias.reshape(B, self.num_heads, N, N)
            if attn_bias.dtype == torch.bool:
                new_attn_bias = torch.zeros_like(attn_bias, dtype=q.dtype)
                new_attn_bias.masked_fill_(attn_bias, float("-inf"))
                attn_bias = new_attn_bias
            else:
                attn_bias = attn_bias.to(dtype=q.dtype)

        use_xformers_attn = XFORMERS_AVAILABLE and not self.coupling_use_tanh
        use_fsdpa = hasattr(nn.functional, 'scaled_dot_product_attention') and not self.coupling_use_tanh

        if use_fsdpa:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            o_heads = attn_output.transpose(1, 2)
        elif use_xformers_attn:
            attn_output = memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.attn_drop.p if self.training else 0.0,
            )
            o_heads = attn_output
        else:
            q_perm = q.permute(0, 2, 1, 3)
            k_perm = k.permute(0, 2, 1, 3)
            attn_scores = (q_perm * self.scale) @ k_perm.transpose(-2, -1)
            if attn_bias is not None:
                attn_scores = attn_scores + attn_bias
            if self.coupling_use_tanh:
                attn_probs = torch.tanh(attn_scores) * self.scale
            else:
                attn_probs = attn_scores.softmax(dim=-1)
            attn_probs = self.attn_drop(attn_probs)
            v_perm = v.permute(0, 2, 1, 3)
            o_heads = torch.matmul(attn_probs, v_perm)
            o_heads = o_heads.permute(0, 2, 1, 3)

        o_tokens = o_heads.reshape([B, N, self.dim])
        if self.coupling_use_vo_proj:
            o_tokens = self.o_proj(o_tokens)
        o_tokens = self.output_dropout(o_tokens)

        if update_phase:
            assert gamma is not None, "Gamma scaling is required for phase update"
            delta = o_tokens.view(B, N, self.num_heads, self.head_dim // 2, 2)
            phase_head_view = phase_state
            alignment = (delta * phase_head_view).sum(dim=-1, keepdim=True)
            delta = delta.addcmul(alignment, phase_head_view, value=-1.0)
            phase_update = F.normalize(phase_head_view + gamma * delta, dim=-1, eps=1e-5)
            return phase_update

        o_full = (
            o_tokens.view(B, N, self.num_heads, self.head_dim // 2, 2)
            .contiguous()
        )

        return o_full


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_weight_dtype(self) -> torch.dtype:
        if hasattr(self.mlp.c_fc, 'int8_original_dtype'):
            return self.mlp.c_fc.int8_original_dtype
        return self.mlp.c_fc.weight.dtype

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x,
            need_weights=False,
            attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            qk_norm: bool = False,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn_inner: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            batch_first: bool = True,
            # KoPE-related
            enable_kope: bool = False,
            kope_vo_rotation: bool = True,
            kope_mix: bool = True,
            kope_mix_init_gain: float = 0.1,
            kope_mix_phase_norm: bool = True,
            checkpoint_rotation: bool = False,
            shared_phase_coupling_fn = None,
            kope_coupling_gamma: float = 0.05,
            kope_learn_gamma: bool = False,
            shared_gamma_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert batch_first, 'batch_first must be True for CustomResidualAttentionBlock'

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            qk_norm=qk_norm,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            inner_norm=scale_attn_inner,
            norm_layer=norm_layer,
            # KoPE-related
            enable_kope=enable_kope,
            kope_vo_rotation=kope_vo_rotation,
            kope_mix=kope_mix,
            kope_mix_init_gain=kope_mix_init_gain,
            kope_mix_phase_norm=kope_mix_phase_norm,
            checkpoint_rotation=checkpoint_rotation,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),  # from NormFormer / Foundation Transformers
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.enable_kope = enable_kope
        self.shared_phase_coupling_fn = shared_phase_coupling_fn
        self.kope_coupling_gamma = kope_coupling_gamma
        self.kope_learn_gamma = kope_learn_gamma
        self.shared_gamma_module = shared_gamma_module

    def get_weight_dtype(self) -> torch.dtype:
        if hasattr(self.mlp.c_fc, 'int8_original_dtype'):
            return self.mlp.c_fc.int8_original_dtype
        return self.mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, phase_state: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask, phase_state=phase_state)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        if self.enable_kope:
            assert self.shared_phase_coupling_fn is not None
            current_gamma = self.shared_gamma_module() if self.shared_gamma_module is not None else self.kope_coupling_gamma
            next_phase_state = self.shared_phase_coupling_fn(
                x, phase_state, attn_bias=attn_mask, update_phase=True, gamma=current_gamma
            )
            return x, next_phase_state
        return x


class CustomTransformer(nn.Module):
    """ A custom transformer that can use different block types. """
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            batch_first: bool = True,
            block_types: Union[str, List[str]] = 'CustomResidualAttentionBlock',
            # KoPE-related
            enable_kope: bool = False,
            kope_vo_rotation: bool = True,
            kope_mix: bool = True,
            kope_mix_init_gain: float = 0.1,
            kope_mix_phase_norm: bool = True,
            checkpoint_rotation: bool = False,
            kope_coupling_gamma: float = 0.05,
            kope_coupling_qknorm_learn: bool = True,
            kope_learn_gamma: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first  # run transformer stack in batch first (N, L, D)
        self.grad_checkpointing = False

        if isinstance(block_types, str):
            block_types = [block_types] * layers
        assert len(block_types) == layers

        self.enable_kope = enable_kope
        self.shared_gamma_module = None
        if enable_kope:
            self.shared_phase_coupling = KoPECoupling(
                dim=width,
                num_heads=heads,
                qknorm=True,
                qknorm_learn=kope_coupling_qknorm_learn,
            )
            if kope_learn_gamma:
                self.shared_gamma_module = PhaseStep(init=kope_coupling_gamma)

        def _create_block(bt: str):
            if bt == 'CustomResidualAttentionBlock':
                return CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                    # KoPE-related
                    enable_kope=enable_kope,
                    kope_vo_rotation=kope_vo_rotation,
                    kope_mix=kope_mix,
                    kope_mix_init_gain=kope_mix_init_gain,
                    kope_mix_phase_norm=kope_mix_phase_norm,
                    checkpoint_rotation=checkpoint_rotation,
                    shared_phase_coupling_fn=self.shared_phase_coupling.forward if enable_kope else None,
                    kope_coupling_gamma=kope_coupling_gamma,
                    kope_learn_gamma=kope_learn_gamma,
                    shared_gamma_module=self.shared_gamma_module,
                )
            else:
                assert False

        self.resblocks = nn.ModuleList([
            _create_block(bt)
            for bt in block_types
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].get_weight_dtype()

    def forward_intermediates(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            phase_state: Optional[torch.Tensor] = None,
    ):
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()  # NLD -> LND

        intermediates = []
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.resblocks
        else:
            blocks = self.resblocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                #x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)
                if self.enable_kope:
                    assert phase_state is not None
                    x, phase_state = checkpoint(blk, x, attn_mask=attn_mask, phase_state=phase_state, use_reentrant=False)
                else:
                    x = checkpoint(blk, x, attn_mask=attn_mask, use_reentrant=False)
            else:
                if self.enable_kope:
                    assert phase_state is not None
                    x, phase_state = blk(x, attn_mask=attn_mask, phase_state=phase_state)
                else:
                    x = blk(x, attn_mask=attn_mask)

            if i in take_indices:
                intermediates.append(x.transpose(0, 1) if not self.batch_first else x)

        if not self.batch_first:
            x = x.transpose(0, 1)  # LND -> NLD

        return x, intermediates

    def prune_intermediate_layers(self, indices: Union[int, List[int]] = 1):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[:max_index + 1]  # truncate blocks
        return take_indices

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, phase_state: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                #x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
                if self.enable_kope:
                    assert phase_state is not None
                    x, phase_state = checkpoint(r, x, attn_mask=attn_mask, phase_state=phase_state, use_reentrant=False)
                else:
                    x = checkpoint(r, x, attn_mask=attn_mask, use_reentrant=False)
            else:
                if self.enable_kope:
                    assert phase_state is not None
                    x, phase_state = r(x, attn_mask=attn_mask, phase_state=phase_state)
                else:
                    x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            batch_first: bool = True,
            block_type: Optional[str] = None,
            qk_norm: bool = False,
            scaled_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn_inner: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            # KoPE-related
            enable_kope: bool = False,
            kope_vo_rotation: bool = True,
            kope_mix: bool = True,
            kope_mix_init_gain: float = 0.1,
            kope_mix_phase_norm: bool = True,
            checkpoint_rotation: bool = False,
            kope_coupling_gamma: float = 0.05,
            kope_coupling_qknorm_learn: bool = True,
            kope_learn_gamma: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False

        # Auto-select custom block if any custom features are enabled
        if block_type is None:
            if any([qk_norm, scaled_cosine_attn, scale_heads, scale_attn_inner, scale_attn, scale_fc, enable_kope]):
                block_type = 'custom'
            else:
                block_type = 'default'

        self.enable_kope = enable_kope
        self.shared_gamma_module = None
        if enable_kope:
            self.shared_phase_coupling = KoPECoupling(
                dim=width,
                num_heads=heads,
                qknorm=True,
                qknorm_learn=kope_coupling_qknorm_learn,
            )
            if kope_learn_gamma:
                self.shared_gamma_module = PhaseStep(init=kope_coupling_gamma)

        if block_type == 'custom':
            self.resblocks = nn.ModuleList([
                CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    scale_cosine_attn=scaled_cosine_attn,
                    scale_heads=scale_heads,
                    scale_attn_inner=scale_attn_inner,
                    scale_attn=scale_attn,
                    scale_fc=scale_fc,
                    batch_first=batch_first,
                    # KoPE related
                    enable_kope=enable_kope,
                    kope_vo_rotation=kope_vo_rotation,
                    kope_mix=kope_mix,
                    kope_mix_init_gain=kope_mix_init_gain,
                    kope_mix_phase_norm=kope_mix_phase_norm,
                    checkpoint_rotation=checkpoint_rotation,
                    shared_phase_coupling_fn=self.shared_phase_coupling.forward if enable_kope else None,
                    kope_coupling_gamma=kope_coupling_gamma,
                    kope_learn_gamma=kope_learn_gamma,
                    shared_gamma_module=self.shared_gamma_module,
                )
                for _ in range(layers)
            ])
        else:
            self.resblocks = nn.ModuleList([
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                )
                for _ in range(layers)
            ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].get_weight_dtype()

    def forward_intermediates(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            phase_state: Optional[torch.Tensor] = None,
    ):
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND

        intermediates = []
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.resblocks
        else:
            blocks = self.resblocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                #x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)
                if self.enable_kope:
                    assert phase_state is not None
                    x, phase_state = checkpoint(blk, x, attn_mask=attn_mask, phase_state=phase_state, use_reentrant=False)
                else:
                    x = checkpoint(blk, x, attn_mask=attn_mask, use_reentrant=False)
            else:
                if self.enable_kope:
                    assert phase_state is not None
                    x, phase_state = blk(x, attn_mask=attn_mask, phase_state=phase_state)
                else:
                    x = blk(x, attn_mask=attn_mask)

            if i in take_indices:
                intermediates.append(x.transpose(0, 1) if not self.batch_first else x)

        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD

        return x, intermediates

    def prune_intermediate_layers(self, indices: Union[int, List[int]] = 1):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[:max_index + 1]  # truncate blocks
        return take_indices

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, phase_state: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                #x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
                if self.enable_kope:
                    assert phase_state is not None
                    x, phase_state = checkpoint(r, x, attn_mask=attn_mask, phase_state=phase_state, use_reentrant=False)
                else:
                    x = checkpoint(r, x, attn_mask=attn_mask, use_reentrant=False)
            else:
                if self.enable_kope:
                    assert phase_state is not None
                    x, phase_state = r(x, attn_mask=attn_mask, phase_state=phase_state)
                else:
                    x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            block_type: Optional[str] = None,
            qk_norm: bool = False,
            scaled_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn_inner: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            # KoPE related
            enable_kope: bool = False,
            kope_vo_rotation: bool = True,
            kope_mix: bool = True,
            kope_mix_init_gain: float = 0.1,
            kope_mix_phase_norm: bool = True,
            checkpoint_rotation: bool = True,
            kope_coupling_gamma: float = 0.05,
            kope_coupling_qknorm_learn: bool = True,
            kope_learn_gamma: bool = False,
            kope_base: int = 20,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim
        self.enable_kope = enable_kope
        self.heads = heads

        if self.enable_kope:
            head_dim = width // heads
            inv_freq = 1.0 / (
                kope_base ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32) / (head_dim // 2))
            )
            self.register_buffer("_phase_inv_freq", inv_freq, persistent=False)
            self._coords_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            block_type=block_type,
            qk_norm=qk_norm,
            scaled_cosine_attn=scaled_cosine_attn,
            scale_heads=scale_heads,
            scale_attn_inner=scale_attn_inner,
            scale_attn=scale_attn,
            scale_fc=scale_fc,
            # KoPE related
            enable_kope=enable_kope,
            kope_vo_rotation=kope_vo_rotation,
            kope_mix=kope_mix,
            kope_mix_init_gain=kope_mix_init_gain,
            kope_mix_phase_norm=kope_mix_phase_norm,
            checkpoint_rotation=checkpoint_rotation,
            kope_coupling_gamma=kope_coupling_gamma,
            kope_coupling_qknorm_learn=kope_coupling_qknorm_learn,
            kope_learn_gamma=kope_learn_gamma,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def _embeds(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # patch dropout (if active)
        x = self.patch_dropout(x)

        # apply norm before transformer
        x = self.ln_pre(x)
        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        return pooled, tokens

    def _get_patch_coords(self, height: int, width: int, device: torch.device):
        key = (height, width, device)
        if key in self._coords_cache:
            return self._coords_cache[key]

        grid_h = torch.arange(height, device=device)
        grid_w = torch.arange(width, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=-1).float()
        coords = grid.reshape(-1, 2)
        self._coords_cache[key] = coords
        return coords

    def _initialize_phase(self, coords: torch.Tensor, dtype: torch.dtype, device: torch.device):
        coords_fp32 = coords.to(device=device, dtype=torch.float32)
        inv_freq = self._phase_inv_freq
        cos_components, sin_components = build_2d_rope_from_coords(coords_fp32, inv_freq)

        base_cos = cos_components.to(dtype=dtype)
        base_sin = sin_components.to(dtype=dtype)
        half_dim = base_cos.shape[-1]

        # for CLS token
        identity_cos_base = torch.ones((1, 1, half_dim), dtype=dtype, device=device)
        identity_sin_base = torch.zeros_like(identity_cos_base)

        cos_base = torch.cat((identity_cos_base, base_cos), dim=1)
        sin_base = torch.cat((identity_sin_base, base_sin), dim=1)

        # 1, L, 1, D
        phase_cos = cos_base.unsqueeze(2)
        phase_sin = sin_base.unsqueeze(2)
        return phase_cos, phase_sin

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            output_fmt: str = 'NCHW',
            output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            intermediates_only: Only return intermediate features
            normalize_intermediates: Apply final norm layer to all intermediates
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both extra prefix class tokens
        Returns:

        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'

        # forward pass
        B, _, height, width = x.shape
        x = self._embeds(x)
        if self.enable_kope:
            feat_h = height // self.patch_size[0]
            feat_w = width // self.patch_size[1]
            patch_coords = self._get_patch_coords(feat_h, feat_w, x.device)
            coords_input = patch_coords.unsqueeze(0)
            phase_scale_cos, phase_scale_sin = self._initialize_phase(coords_input, x.dtype, x.device)
            # Combine to phase_state (1, L, 1, D//2, 2)
            phase_state = torch.stack([phase_scale_cos, phase_scale_sin], dim=-1)
            # Expand to (B, L, H, D//2, 2) - this is a zero-copy view
            phase_state = phase_state.expand(B, -1, self.heads, -1, -1)
            x, intermediates = self.transformer.forward_intermediates(
                x,
                indices=indices,
                stop_early=stop_early,
                phase_state=phase_state,
            )
        else:
            x, intermediates = self.transformer.forward_intermediates(
                x,
                indices=indices,
                stop_early=stop_early,
            )

        # process intermediates
        if normalize_intermediates:
            # apply final norm to all intermediates
            intermediates = [self.ln_post(xi) for xi in intermediates]
        num_prefix_tokens = 1  # one class token that's always there (as of now)
        if num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, num_prefix_tokens:] for y in intermediates]
        else:
            prefix_tokens = None
        if reshape:
            # reshape to BCHW output format
            H, W = height // self.patch_size[0], width // self.patch_size[1]
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        output = {'image_intermediates': intermediates}
        if prefix_tokens is not None and output_extra_tokens:
            output['image_intermediates_prefix'] = prefix_tokens

        if intermediates_only:
            return output

        pooled, _ = self._pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        output['image_features'] = pooled

        return output

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices = self.transformer.prune_intermediate_layers(indices)
        if prune_norm:
            self.ln_post = nn.Identity()
        if prune_head:
            self.proj = None
        return take_indices

    def forward(self, x: torch.Tensor):
        if not self.enable_kope:
            x = self._embeds(x)
            x = self.transformer(x)
        else:
            B, _, H, W = x.shape
            x = self._embeds(x)
            feat_h = H // self.patch_size[0]
            feat_w = W // self.patch_size[1]
            patch_coords = self._get_patch_coords(feat_h, feat_w, x.device)
            coords_input = patch_coords.unsqueeze(0)
            phase_scale_cos, phase_scale_sin = self._initialize_phase(coords_input, x.dtype, x.device)

            # Combine to phase_state (1, L, 1, D//2, 2)
            phase_state = torch.stack([phase_scale_cos, phase_scale_sin], dim=-1)
            # Expand to (B, L, H, D//2, 2) - this is a zero-copy view
            phase_state = phase_state.expand(B, -1, self.heads, -1, -1)

            x = self.transformer(x, phase_state=phase_state)

        pooled, tokens = self._pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled
