# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Tuple

from utils.pos_embed import VisionRotaryEmbeddingFast
import torch
from torch import nn

from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F

# used for KoPE
def apply_rotary_pairs(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # x: [B, N, H, D] -> [B, N, H, D/2, 2]
    x_view = x.reshape(*x.shape[:-1], -1, 2)
    real = x_view[..., 0]
    imag = x_view[..., 1]
    updated_real = real * cos - imag * sin
    updated_imag = real * sin + imag * cos
    return torch.stack((updated_real, updated_imag), dim=-1).view_as(x)

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

class PhaseStep(nn.Module):
    def __init__(self, init=0.01, max_value=1.0):
        super().__init__()
        # inverse softplus
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
        qknorm: bool = True,
        qknorm_learn: bool = True,
        coupling_qk_multilayer: bool = False,
        gamma: float = 0.05,
        learn_gamma: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
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

        self.attn_dropout = nn.Dropout(attn_drop)
        self.output_dropout = nn.Dropout(proj_drop)

        self.gamma = gamma
        self.learn_gamma = learn_gamma
        if self.learn_gamma:
            self.gamma_param = PhaseStep(init=gamma, max_value=1.0)
        else:
            self.register_parameter("gamma_param", None)

    def forward(
        self,
        tokens: torch.Tensor,
        phase_cos: torch.Tensor,
        phase_sin: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # tokens: [B, N, C], phase_*: [B, N, H, D/2]
        B, N, H, half_dim = phase_cos.shape

        q = self.q_proj(tokens).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(tokens).view(B, N, self.num_heads, self.head_dim)

        if self.qknorm:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            if self.qk_scale is not None:
                q = q * self.qk_scale.to(dtype=q.dtype)
                #k = k * self.qk_scale

        phase_head = torch.stack((phase_cos, phase_sin), dim=-1).reshape(B, N, self.num_heads, self.head_dim)
        v = phase_head

        use_fsdpa = hasattr(F, 'scaled_dot_product_attention')
        if use_fsdpa:
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = (~key_padding_mask).view(B, 1, 1, N)
            attn_output = F.scaled_dot_product_attention(
                q.transpose(1, 2),  # B, H, N, D
                k.transpose(1, 2),  # B, H, N, D
                v.transpose(1, 2),  # B, H, N, D
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )  # B, H, N, D
            attn_output = attn_output.transpose(1, 2)  # B, N, H, D
            o = attn_output
        else:
            q_perm = q.permute(0, 2, 1, 3)
            k_perm = k.permute(0, 2, 1, 3)
            attn_scores = (q_perm * self.scale) @ k_perm.transpose(-2, -1)
            attn_probs = attn_scores.softmax(dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            v_perm = v.permute(0, 2, 1, 3)
            o = torch.matmul(attn_probs, v_perm)
            o = o.permute(0, 2, 1, 3)

        o = o.reshape([B, N, self.dim])
        o = self.output_dropout(o)

        gamma = self.gamma_param().to(dtype=o.dtype) if self.learn_gamma else self.gamma
        delta = o.view(B, N, self.num_heads, self.head_dim // 2, 2)
        phase_head = phase_head.view(B, N, self.num_heads, self.head_dim // 2, 2)
        # projection
        alignment = (delta * phase_head).sum(dim=-1, keepdim=True)
        delta = delta.addcmul(alignment, phase_head, value=-1.0)
        phase_update = F.normalize(phase_head + gamma * delta, dim=-1, eps=1e-5)
        return phase_update[..., 0], phase_update[..., 1]


# Standard ViT
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        no_rope: int = 1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if self.head_dim % 2 != 0:
            raise ValueError("Rotary embeddings require the head dimension to be even")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        half_head_dim = embed_dim // num_heads // 2
        self.rotary = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=int(max_seq_len ** 0.5),
            no_rope=no_rope,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary(q)
        k = self.rotary(k)

        # Use FSDPA if available
        use_fsdpa = hasattr(F, 'scaled_dot_product_attention')
        if use_fsdpa:
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = (~key_padding_mask).view(batch_size, 1, 1, -1)
            attn_output = F.scaled_dot_product_attention(
                q,  # B, H, N, D
                k,  # B, H, N, D
                v,  # B, H, N, D
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )  # B, H, N, D
            attn_output = attn_output.transpose(1, 2)  # B, N, H, D
            attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
            attn_output = self.proj(attn_output)
            attn_output = self.proj_dropout(attn_output)
            return attn_output

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(
                mask,
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)
        return context


# KoPE
class MultiHeadSelfAttentionKoPE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        no_rope: int = 1,
        kope_vo_rotation: bool = True,
        kope_mix: bool = True,
        kope_mix_init_gain: float = 0.1,
        kope_mix_phase_norm: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if self.head_dim % 2 != 0:
            raise ValueError("Rotary embeddings require the head dimension to be even")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        half_head_dim = embed_dim // num_heads // 2

        self.kope_vo_rotation = kope_vo_rotation
        self.kope_mix = kope_mix
        self.kope_mix_init_gain = kope_mix_init_gain
        self.kope_mix_phase_norm = kope_mix_phase_norm

        if self.kope_mix:
            self.kope_mix_coef = nn.Parameter(torch.empty(self.num_heads, half_head_dim, half_head_dim))
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

    def forward(
        self,
        x: torch.Tensor,
        phase_cos: torch.Tensor,
        phase_sin: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        # B, N, H, D
        q, k, v = qkv.unbind(dim=2)

        cos = phase_cos.to(dtype=q.dtype)
        sin = phase_sin.to(dtype=q.dtype)
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

        use_fsdpa = hasattr(F, 'scaled_dot_product_attention')
        if use_fsdpa:
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = (~key_padding_mask).view(batch_size, 1, 1, -1)
            attn_output = F.scaled_dot_product_attention(
                q.transpose(1, 2),  # B, H, N, D
                k.transpose(1, 2),  # B, H, N, D
                v.transpose(1, 2),  # B, H, N, D
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )  # B, H, N, D
            attn_output = attn_output.transpose(1, 2)  # B, N, H, D
            if self.kope_vo_rotation:
                # Reverse the VO rotation on output
                attn_output = apply_rotary_pairs(attn_output, cos, -sin)
            attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
            attn_output = self.proj(attn_output)
            attn_output = self.proj_dropout(attn_output)
            return attn_output

        q = q.transpose(1, 2)  # B, H, N, D
        k = k.transpose(1, 2)  # B, H, N, D
        v = v.transpose(1, 2)  # B, H, N, D
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(
                mask,
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2)
        if self.kope_vo_rotation:
            context = apply_rotary_pairs(context, cos, -sin)
        context = context.reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)
        return context


# Standard ViT
class ARCTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = residual + self.dropout3(x)
        x = self.norm2(x)
        return x


# KoPE
class ARCTransformerEncoderLayerKoPE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 1,
        kope_vo_rotation: bool = True,
        kope_mix: bool = True,
        kope_mix_init_gain: float = 0.1,
        kope_mix_phase_norm: bool = True,
        shared_kope_coupling_fn = None,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttentionKoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
            kope_vo_rotation=kope_vo_rotation,
            kope_mix=kope_mix,
            kope_mix_init_gain=kope_mix_init_gain,
            kope_mix_phase_norm=kope_mix_phase_norm,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.shared_kope_coupling_fn = shared_kope_coupling_fn

    def forward(
        self,
        x: torch.Tensor,
        phase_cos: torch.Tensor,
        phase_sin: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, phase_cos=phase_cos, phase_sin=phase_sin, key_padding_mask=key_padding_mask)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = residual + self.dropout3(x)
        x = self.norm2(x)

        assert self.shared_kope_coupling_fn is not None, "shared_kope_coupling_fn must be provided for KoPE coupling"
        phase_cos, phase_sin = self.shared_kope_coupling_fn(
            x,
            phase_cos,
            phase_sin,
            key_padding_mask=key_padding_mask,
        )
        return x, phase_cos, phase_sin


# Standard ViT
class ARCTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ARCTransformerEncoderLayer(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                    max_seq_len=max_seq_len,
                    no_rope=no_rope,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


# KoPE
class ARCTransformerEncoderKoPE(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 0,
        kope_vo_rotation: bool = True,
        kope_mix: bool = True,
        kope_mix_init_gain: float = 0.1,
        kope_mix_phase_norm: bool = True,
        kope_gamma: float = 0.05,
        kope_learn_gamma: bool = True,
        kope_base: int = 20,
    ) -> None:
        super().__init__()
        self.shared_kope_coupling = KoPECoupling(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
            qknorm=True,
            qknorm_learn=True,
            coupling_qk_multilayer=False,
            gamma=kope_gamma,
            learn_gamma=kope_learn_gamma,
        )
        self.layers = nn.ModuleList(
            [
                ARCTransformerEncoderLayerKoPE(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                    max_seq_len=max_seq_len,
                    no_rope=no_rope,
                    kope_vo_rotation=kope_vo_rotation,
                    kope_mix=kope_mix,
                    kope_mix_init_gain=kope_mix_init_gain,
                    kope_mix_phase_norm=kope_mix_phase_norm,
                    shared_kope_coupling_fn=self.shared_kope_coupling.forward,
                )
                for _ in range(depth)
            ]
        )

        # initialization of phase
        head_dim = embed_dim // num_heads
        inv_freq = 1.0 / (kope_base ** (torch.arange(0, head_dim//2, 2, dtype=torch.float32) / (head_dim//2)))
        self.register_buffer("_phase_inv_freq", inv_freq, persistent=False)
        #self._coords_cache = {}
        self.num_task_tokens = no_rope
        self.num_heads = num_heads

    def _get_patch_coords(self, height: int, width: int, device: torch.device):
        #key = (height, width, device)
        #if key in self._coords_cache:
        #    return self._coords_cache[key].clone()

        grid_h = torch.arange(height, device=device)
        grid_w = torch.arange(width, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=-1).float()
        coords = grid.reshape(-1, 2)
        #self._coords_cache[key] = coords
        return coords.clone()

    def _initialize_phase(self, coords: torch.Tensor, dtype: torch.dtype, device: torch.device):
        coords_fp32 = coords.to(device=device, dtype=torch.float32)
        inv_freq = self._phase_inv_freq
        cos_components, sin_components = build_2d_rope_from_coords(coords_fp32, inv_freq)

        base_cos = cos_components.to(dtype=dtype)
        base_sin = sin_components.to(dtype=dtype)
        half_dim = base_cos.shape[-1]

        # for task tokens
        identity_cos_base = torch.ones((1, self.num_task_tokens, half_dim), dtype=dtype, device=device)
        identity_sin_base = torch.zeros_like(identity_cos_base)

        cos_base = torch.cat((identity_cos_base, base_cos), dim=1)
        sin_base = torch.cat((identity_sin_base, base_sin), dim=1)

        # 1, L, 1, D
        phase_cos = cos_base.unsqueeze(2)
        phase_sin = sin_base.unsqueeze(2)
        return phase_cos, phase_sin

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # initialize phase
        B, N, C = x.shape
        feat_h = int((N - self.num_task_tokens) ** 0.5)
        feat_w = int((N - self.num_task_tokens) ** 0.5)
        patch_coords = self._get_patch_coords(feat_h, feat_w, x.device)
        coords_input = patch_coords.unsqueeze(0)
        phase_cos, phase_sin = self._initialize_phase(coords_input, x.dtype, x.device)
        phase_cos = phase_cos.expand(B, -1, self.num_heads, -1).clone()
        phase_sin = phase_sin.expand(B, -1, self.num_heads, -1).clone()
        for layer in self.layers:
            x, phase_cos, phase_sin = layer(x, key_padding_mask=key_padding_mask, phase_cos=phase_cos, phase_sin=phase_sin)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        no_rope: int = 1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if self.head_dim % 2 != 0:
            raise ValueError("Rotary embeddings require the head dimension to be even")

        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        half_head_dim = embed_dim // num_heads // 2
        self.rotary_q = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            # For cross-attn, we assume max grid size.
            pt_seq_len=int(max_seq_len ** 0.5),
            no_rope=no_rope,
        )
        self.rotary_k = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=int(max_seq_len ** 0.5),
            no_rope=no_rope,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None, # For memory
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        _, mem_len, _ = memory.shape

        q = self.q(x)
        kv = self.kv(memory)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # B, H, N_q, D
        kv = kv.view(batch_size, mem_len, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # B, H, N_k, D

        q = self.rotary_q(q)
        k = self.rotary_k(k)

        # Use FSDPA if available
        use_fsdpa = hasattr(F, 'scaled_dot_product_attention')
        if use_fsdpa:
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = (~key_padding_mask).view(batch_size, 1, 1, -1)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            output = self.proj(attn_output)
            output = self.proj_dropout(output)
            return output

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(
                mask,
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.proj(context)
        output = self.proj_dropout(output)
        return output


class ARCTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
        )
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            no_rope=no_rope,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None, # For memory
    ) -> torch.Tensor:
        # Self Attention
        residual = x
        x = self.self_attn(x)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        # Cross Attention
        residual = x
        x = self.cross_attn(x, memory, key_padding_mask=key_padding_mask)
        x = residual + self.dropout2(x)
        x = self.norm2(x)

        # MLP
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        x = residual + self.dropout4(x)
        x = self.norm3(x)
        return x


class ARCTransformerDecoder(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ARCTransformerDecoderLayer(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                    max_seq_len=max_seq_len,
                    no_rope=no_rope,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, key_padding_mask=key_padding_mask)
        return x


class ARCTransformerDecoderLayerKoPESelf(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 1,
        kope_vo_rotation: bool = True,
        kope_mix: bool = True,
        kope_mix_init_gain: float = 0.1,
        kope_mix_phase_norm: bool = True,
        shared_kope_coupling_fn = None,
    ) -> None:
        super().__init__()
        # Self Attention uses KoPE
        self.self_attn = MultiHeadSelfAttentionKoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
            kope_vo_rotation=kope_vo_rotation,
            kope_mix=kope_mix,
            kope_mix_init_gain=kope_mix_init_gain,
            kope_mix_phase_norm=kope_mix_phase_norm,
        )
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            no_rope=no_rope,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.shared_kope_coupling_fn = shared_kope_coupling_fn

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        phase_cos: torch.Tensor,
        phase_sin: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None, # For memory
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Self Attention (KoPE)
        residual = x
        x = self.self_attn(x, phase_cos=phase_cos, phase_sin=phase_sin)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        # Cross Attention (Standard)
        residual = x
        x = self.cross_attn(x, memory, key_padding_mask=key_padding_mask)
        x = residual + self.dropout2(x)
        x = self.norm2(x)

        # MLP
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        x = residual + self.dropout4(x)
        x = self.norm3(x)

        # Update Phase internally
        assert self.shared_kope_coupling_fn is not None
        phase_cos, phase_sin = self.shared_kope_coupling_fn(
            x,
            phase_cos,
            phase_sin,
            key_padding_mask=None, # Decoder self-attn usually sees everything
        )
        return x, phase_cos, phase_sin


class ARCTransformerDecoderKoPE(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 0,
        # KoPE related args
        kope_vo_rotation: bool = True,
        kope_mix: bool = True,
        kope_mix_init_gain: float = 0.1,
        kope_mix_phase_norm: bool = True,
        kope_gamma: float = 0.05,
        kope_learn_gamma: bool = True,
        kope_base: int = 20,
    ) -> None:
        super().__init__()

        self.shared_kope_coupling = KoPECoupling(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
            qknorm=True,
            qknorm_learn=True,
            coupling_qk_multilayer=False,
            gamma=kope_gamma,
            learn_gamma=kope_learn_gamma,
        )
        # Helper for phase init
        head_dim = embed_dim // num_heads
        inv_freq = 1.0 / (kope_base ** (torch.arange(0, head_dim//2, 2, dtype=torch.float32) / (head_dim//2)))
        self.register_buffer("_phase_inv_freq", inv_freq, persistent=False)
        #self._coords_cache = {}
        self.num_task_tokens = no_rope
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(ARCTransformerDecoderLayerKoPESelf(
                embed_dim, num_heads, mlp_dim, dropout, max_seq_len, no_rope,
                kope_vo_rotation, kope_mix, kope_mix_init_gain, kope_mix_phase_norm,
                shared_kope_coupling_fn=self.shared_kope_coupling.forward
            ))

    def _get_patch_coords(self, height: int, width: int, device: torch.device):
        #key = (height, width, device)
        #if key in self._coords_cache:
        #    return self._coords_cache[key].clone()

        grid_h = torch.arange(height, device=device)
        grid_w = torch.arange(width, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=-1).float()
        coords = grid.reshape(-1, 2)
        #self._coords_cache[key] = coords
        return coords.clone()

    def _initialize_phase(self, coords: torch.Tensor, dtype: torch.dtype, device: torch.device):
        coords_fp32 = coords.to(device=device, dtype=torch.float32)
        inv_freq = self._phase_inv_freq
        cos_components, sin_components = build_2d_rope_from_coords(coords_fp32, inv_freq)

        base_cos = cos_components.to(dtype=dtype)
        base_sin = sin_components.to(dtype=dtype)
        half_dim = base_cos.shape[-1]

        # for task tokens
        if self.num_task_tokens > 0:
            identity_cos_base = torch.ones((1, self.num_task_tokens, half_dim), dtype=dtype, device=device)
            identity_sin_base = torch.zeros_like(identity_cos_base)

            cos_base = torch.cat((identity_cos_base, base_cos), dim=1)
            sin_base = torch.cat((identity_sin_base, base_sin), dim=1)
        else:
            cos_base = base_cos
            sin_base = base_sin

        # 1, L, 1, D
        phase_cos = cos_base.unsqueeze(2)
        phase_sin = sin_base.unsqueeze(2)
        return phase_cos, phase_sin

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Init phase for decoder
        B, N, C = x.shape
        feat_h = int((N - self.num_task_tokens) ** 0.5)
        feat_w = int((N - self.num_task_tokens) ** 0.5)
        patch_coords = self._get_patch_coords(feat_h, feat_w, x.device)
        coords_input = patch_coords.unsqueeze(0)
        phase_cos, phase_sin = self._initialize_phase(coords_input, x.dtype, x.device)
        phase_cos = phase_cos.expand(B, -1, self.num_heads, -1).clone()
        phase_sin = phase_sin.expand(B, -1, self.num_heads, -1).clone()

        for layer in self.layers:
            x, phase_cos, phase_sin = layer(x, memory, phase_cos, phase_sin, key_padding_mask=key_padding_mask)
        return x


# ViTKoPE Encoder + ViT Decoder
class ARCViTKoPEEncDec(nn.Module):
    """Vision Transformer tailored for ARC tasks.

    Each ARC task gets a dedicated learnable token that is prepended to the
    sequence of flattened pixel embeddings. Pixels are represented by a
    discrete color vocabulary of size ``num_colors``.
    """

    def __init__(
        self,
        num_tasks: int,
        image_size: int = 30,
        num_colors: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        depth_decoder: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        num_task_tokens: int = 1,
        patch_size: int = 2,
        kope_vo_rotation: bool = True,
        kope_mix: bool = True,
        kope_mix_init_gain: float = 0.1,
        kope_mix_phase_norm: bool = True,
        kope_gamma: float = 0.05,
        kope_learn_gamma: bool = True,
        kope_base: int = 20,
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("`image_size` must be > 0.")
        if num_colors <= 0:
            raise ValueError("`num_colors` must be > 0.")
        if num_tasks <= 0:
            raise ValueError("`num_tasks` must be > 0.")

        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        if patch_size is None:
            self.seq_length = image_size * image_size
        else:
            self.seq_length = (image_size//patch_size)**2
        self.patch_size = patch_size
        print(f"Patch size: {self.patch_size}, sequence length: {self.seq_length}")
        self.num_task_tokens = num_task_tokens
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim * self.num_task_tokens)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)

        total_seq_len = self.num_task_tokens + self.seq_length
        self.positional_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))

        # Decoder params
        self.decoder_query_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))

        self.encoder = ARCTransformerEncoderKoPE(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=total_seq_len,
            no_rope=num_task_tokens,
            kope_vo_rotation=kope_vo_rotation,
            kope_mix=kope_mix,
            kope_mix_init_gain=kope_mix_init_gain,
            kope_mix_phase_norm=kope_mix_phase_norm,
            kope_gamma=kope_gamma,
            kope_learn_gamma=kope_learn_gamma,
            kope_base=kope_base,
        )
        self.decoder = ARCTransformerDecoder(
            depth=depth_decoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=total_seq_len,
            no_rope=num_task_tokens,
            )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_colors * (1 if patch_size is None else patch_size)**2)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_query_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        if pixel_values.dim() != 3:
            raise ValueError("`pixel_values` must be (batch, height, width).")
        if pixel_values.size(1) != self.image_size or pixel_values.size(2) != self.image_size:
            raise ValueError(
                "`pixel_values` height/width must match configured image_size="
                f"{self.image_size}. Received {pixel_values.shape[1:]}"
            )

        batch_size = pixel_values.size(0)
        device = pixel_values.device

        tokens = self.color_embed(pixel_values.long())
        tokens = self.patch_embed(tokens.permute((0, 3, 1, 2)))
        tokens = tokens + self.positional_embed[:, : tokens.size(1), :]

        task_tokens = self.task_token_embed(task_ids.long())
        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)
        hidden_states = torch.cat([task_tokens, tokens], dim=1)
        hidden_states = self.dropout(hidden_states)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, self.image_size, self.image_size):
                raise ValueError(
                    "`attention_mask` must match pixel grid size."
                )
            if self.patch_size is not None:
                attention_mask = attention_mask.reshape(batch_size, self.image_size//self.patch_size, self.patch_size, self.image_size//self.patch_size, self.patch_size)
                attention_mask = torch.max(torch.max(attention_mask, dim=2)[0], dim=3)[0]
            flat_mask = attention_mask.view(batch_size, self.seq_length)
            pad_mask = ~flat_mask.bool()
            pad_mask = torch.cat(
                [torch.zeros(batch_size, self.num_task_tokens, device=device, dtype=torch.bool), pad_mask],
                dim=1,
            )
            key_padding_mask = pad_mask

        # ViTKoPE Encoder
        memory = self.encoder(hidden_states, key_padding_mask=key_padding_mask)

        # ViT Decoder
        # Prepare decoder input
        # Query tokens (learnable) + Pos Embed
        decoder_tokens = self.decoder_query_embed.expand(batch_size, -1, -1)
        decoder_tokens = decoder_tokens + self.decoder_pos_embed[:, :decoder_tokens.size(1), :]

        # Concatenate task tokens (reusing from input as they define the task)
        # Residual connection for task tokens: Combine raw task embedding with encoder's contextualized output
        encoder_task_tokens = memory[:, :self.num_task_tokens, :]
        task_tokens_combined = task_tokens + encoder_task_tokens

        decoder_input = torch.cat([task_tokens_combined, decoder_tokens], dim=1)
        decoder_input = self.dropout(decoder_input)

        # In cross attention, memory is the encoder output.
        # key_padding_mask applies to memory (encoder output), so we reuse it.
        # We don't have a specific padding mask for decoder queries (assuming all valid or we treat pad as task).
        decoded = self.decoder(decoder_input, memory=memory, key_padding_mask=key_padding_mask)

        decoded = self.norm(decoded)
        pixel_states = decoded[:, self.num_task_tokens:, :]

        logits = self.head(pixel_states)
        logits = logits.reshape((-1, self.image_size//self.patch_size, self.image_size//self.patch_size, self.patch_size, self.patch_size, self.num_colors))
        logits = logits.permute((0, 1, 3, 2, 4, 5))
        logits = logits.reshape(batch_size, self.image_size, self.image_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)
        return logits


# ViTKoPE encoder + ViTKoPE decoder
class ARCViTKoPE2EncDec(nn.Module):
    """Vision Transformer tailored for ARC tasks with KoPE in Encoder AND Decoder (Self-Attn only).
    """

    def __init__(
        self,
        num_tasks: int,
        image_size: int = 30,
        num_colors: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        depth_decoder: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        num_task_tokens: int = 1,
        patch_size: int = 2,
        kope_vo_rotation: bool = True,
        kope_mix: bool = True,
        kope_mix_init_gain: float = 0.1,
        kope_mix_phase_norm: bool = True,
        kope_gamma: float = 0.05,
        kope_learn_gamma: bool = True,
        kope_base: int = 20,
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("`image_size` must be > 0.")
        if num_colors <= 0:
            raise ValueError("`num_colors` must be > 0.")
        if num_tasks <= 0:
             raise ValueError("`num_tasks` must be > 0.")

        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        if patch_size is None:
            self.seq_length = image_size * image_size
        else:
            self.seq_length = (image_size//patch_size)**2
        self.patch_size = patch_size
        # print(f"Patch size: {self.patch_size}, sequence length: {self.seq_length}")
        self.num_task_tokens = num_task_tokens
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim * self.num_task_tokens)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)

        total_seq_len = self.num_task_tokens + self.seq_length
        self.positional_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))

        # Decoder params
        self.decoder_query_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))

        self.encoder = ARCTransformerEncoderKoPE(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=total_seq_len,
            no_rope=num_task_tokens,
            kope_vo_rotation=kope_vo_rotation,
            kope_mix=kope_mix,
            kope_mix_init_gain=kope_mix_init_gain,
            kope_mix_phase_norm=kope_mix_phase_norm,
            kope_gamma=kope_gamma,
            kope_learn_gamma=kope_learn_gamma,
            kope_base=kope_base,
        )
        self.decoder = ARCTransformerDecoderKoPE(
            depth=depth_decoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=total_seq_len,
            no_rope=num_task_tokens,
            kope_vo_rotation=kope_vo_rotation,
            kope_mix=kope_mix,
            kope_mix_init_gain=kope_mix_init_gain,
            kope_mix_phase_norm=kope_mix_phase_norm,
            kope_gamma=kope_gamma,
            kope_learn_gamma=kope_learn_gamma,
            kope_base=kope_base,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_colors * (1 if patch_size is None else patch_size)**2)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_query_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        if pixel_values.dim() != 3:
            raise ValueError("`pixel_values` must be (batch, height, width).")
        if pixel_values.size(1) != self.image_size or pixel_values.size(2) != self.image_size:
            raise ValueError(
                "`pixel_values` height/width must match configured image_size="
                f"{self.image_size}. Received {pixel_values.shape[1:]}"
            )

        batch_size = pixel_values.size(0)
        device = pixel_values.device

        tokens = self.color_embed(pixel_values.long())
        tokens = self.patch_embed(tokens.permute((0, 3, 1, 2)))
        tokens = tokens + self.positional_embed[:, : tokens.size(1), :]

        task_tokens = self.task_token_embed(task_ids.long())
        # task_tokens shape: (batch_size, num_task_tokens * embed_dim)
        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)
        hidden_states = torch.cat([task_tokens, tokens], dim=1)
        hidden_states = self.dropout(hidden_states)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, self.image_size, self.image_size):
                raise ValueError(
                    "`attention_mask` must match pixel grid size."
                )
            if self.patch_size is not None:
                attention_mask = attention_mask.reshape(batch_size, self.image_size//self.patch_size, self.patch_size, self.image_size//self.patch_size, self.patch_size)
                attention_mask = torch.max(torch.max(attention_mask, dim=2)[0], dim=3)[0]
            flat_mask = attention_mask.view(batch_size, self.seq_length)
            pad_mask = ~flat_mask.bool()
            pad_mask = torch.cat(
                [torch.zeros(batch_size, self.num_task_tokens, device=device, dtype=torch.bool), pad_mask],
                dim=1,
            )
            key_padding_mask = pad_mask

        # ViTKoPE Encoder
        memory = self.encoder(hidden_states, key_padding_mask=key_padding_mask)

        # ViT Decoder (with KoPE)
        decoder_tokens = self.decoder_query_embed.expand(batch_size, -1, -1)
        decoder_tokens = decoder_tokens + self.decoder_pos_embed[:, :decoder_tokens.size(1), :]

        # Residual connection for task tokens: Combine raw task embedding with encoder's contextualized output
        encoder_task_tokens = memory[:, :self.num_task_tokens, :]
        task_tokens_combined = task_tokens + encoder_task_tokens

        decoder_input = torch.cat([task_tokens_combined, decoder_tokens], dim=1)
        decoder_input = self.dropout(decoder_input)

        decoded = self.decoder(decoder_input, memory=memory, key_padding_mask=key_padding_mask)

        decoded = self.norm(decoded)
        pixel_states = decoded[:, self.num_task_tokens:, :]

        logits = self.head(pixel_states)
        logits = logits.reshape((-1, self.image_size//self.patch_size, self.image_size//self.patch_size, self.patch_size, self.patch_size, self.num_colors))
        logits = logits.permute((0, 1, 3, 2, 4, 5))
        logits = logits.reshape(batch_size, self.image_size, self.image_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)
        return logits