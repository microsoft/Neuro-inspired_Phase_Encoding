# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import math

import torch
from torch.nn import functional as F

import simdinov2.layers.attention as attention_ops


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            #But we also experiments output_kd.size(0)
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss




class PhaseLossOld(torch.nn.Module):
    """Encourages KoPE phase vectors to locally synchronize via angle-based two-attractor interpolation."""

    def __init__(
        self,
        phase_attractor_n: int = 32,
    ):
        super().__init__()
        if phase_attractor_n <= 0:
            raise ValueError("phase_attractor_n must be positive")
        self.phase_attractor_n = phase_attractor_n
        self._eps = 1e-6

    def forward(self, phases):
        """
        Args:
            phases: shape (B, S, H, D//2, 2)
        """
        if phases.numel() == 0:
            return torch.tensor(0.0, device=phases.device, dtype=phases.dtype)

        b, s, h, d_half, two = phases.shape
        assert two == 2, "The last dimension must contain (cos, sin)."

        # (B, S, H, D, 2) -> (B, H, D, S, 2) for per-(B,H,D) grouping over sequence tokens
        grouped = phases.permute(0, 2, 3, 1, 4).reshape(-1, s, 2)
        grouped = F.normalize(grouped, dim=-1)

        n_attr = self.phase_attractor_n
        theta = torch.atan2(grouped[..., 1], grouped[..., 0])
        theta = torch.remainder(theta, 2 * math.pi)
        continuous_idx = theta * (n_attr / (2 * math.pi))
        idx0 = torch.floor(continuous_idx).long()
        idx0 = torch.clamp(idx0, max=n_attr - 1)
        frac = continuous_idx - idx0.float()
        idx1 = (idx0 + 1) % n_attr
        w0 = 1.0 - frac
        w1 = frac

        G = grouped.size(0)
        cluster_weights = torch.zeros(G, n_attr, device=grouped.device, dtype=grouped.dtype)
        cluster_sums = torch.zeros(G, n_attr, 2, device=grouped.device, dtype=grouped.dtype)

        flat_idx0 = idx0.reshape(G, -1)
        flat_idx1 = idx1.reshape(G, -1)
        cluster_weights.scatter_add_(1, flat_idx0, w0.reshape(G, -1))
        cluster_weights.scatter_add_(1, flat_idx1, w1.reshape(G, -1))

        grouped_w0 = grouped * w0.unsqueeze(-1)
        grouped_w1 = grouped * w1.unsqueeze(-1)
        cluster_sums.scatter_add_(
            1,
            idx0.unsqueeze(-1).expand_as(grouped),
            grouped_w0,
        )
        cluster_sums.scatter_add_(
            1,
            idx1.unsqueeze(-1).expand_as(grouped),
            grouped_w1,
        )

        cluster_means = cluster_sums / torch.clamp(cluster_weights.unsqueeze(-1), min=self._eps)
        cluster_means = F.normalize(cluster_means, dim=-1)

        means0 = cluster_means.gather(1, idx0.unsqueeze(-1).expand_as(grouped))
        means1 = cluster_means.gather(1, idx1.unsqueeze(-1).expand_as(grouped))
        align0 = (grouped * means0).sum(dim=-1)
        align1 = (grouped * means1).sum(dim=-1)

        loss_token = w0 * (1 - align0) + w1 * (1 - align1)
        return loss_token.mean()


class PhaseLossOld2(torch.nn.Module):
    """Encourages KoPE phase vectors to locally synchronize via phase-only self-attention.

    Args:
        local_sync_deg: approximate angular bandwidth (in degrees) within which another token
            should retain ~50% of the weight compared to a perfectly aligned neighbor.
            Smaller values push the loss to enforce tighter, more local consensus.
    """

    def __init__(self, local_sync_deg: float = 15.0):
        super().__init__()
        if local_sync_deg <= 0:
            raise ValueError("local_sync_deg must be positive")

        self.local_sync_deg = float(local_sync_deg)
        self.local_sync_rad = math.radians(self.local_sync_deg)
        self._phase_dim = 2.0  # (cos, sin)
        self._kernel_scale = self._compute_kernel_scale(self.local_sync_rad)
        # Match xFormers' implicit 1/sqrt(d) scaling by pre-multiplying with sqrt(phase_dim)
        self._query_scale = self._kernel_scale * math.sqrt(self._phase_dim)
        self._eps = 1e-6

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        if phases.numel() == 0:
            return torch.tensor(0.0, device=phases.device, dtype=phases.dtype)

        b, s, h, d_half, two = phases.shape
        if two != 2:
            raise ValueError("The last dimension of phases must contain (cos, sin)")

        grouped = phases.permute(0, 3, 1, 2, 4).contiguous()  # (B, D//2, S, H, 2)
        grouped = grouped.view(-1, s, h, 2)

        local_mean = self._mean_shift(grouped)
        mean_norm = torch.linalg.vector_norm(local_mean, dim=-1, keepdim=True).clamp_min(self._eps)
        consensus_dir = local_mean / mean_norm
        alignment = (grouped * consensus_dir).sum(dim=-1).clamp(-1.0, 1.0)
        # Weight tokens by how concentrated their local consensus already is.
        weighted_misalignment = (1.0 - alignment) * mean_norm.squeeze(-1)
        return weighted_misalignment.mean()

    def _mean_shift(self, grouped: torch.Tensor) -> torch.Tensor:
        if attention_ops.XFORMERS_AVAILABLE and grouped.is_cuda:
            return self._mean_shift_xformers(grouped)
        flat = grouped.reshape(-1, grouped.size(1), grouped.size(3))
        dense_mean = self._mean_shift_dense(flat)
        return dense_mean.view_as(grouped)

    def _mean_shift_xformers(self, grouped: torch.Tensor) -> torch.Tensor:
        batch_groups, seq_len, heads, feat_dim = grouped.shape

        original_dtype = grouped.dtype
        target_dim = max(16, ((feat_dim + 3) // 4) * 4)

        attn_input = grouped
        if original_dtype not in (torch.float16, torch.bfloat16):
            attn_input = attn_input.to(torch.float16)

        if feat_dim < target_dim:
            attn_input = F.pad(attn_input, (0, target_dim - feat_dim))

        scale_correction = math.sqrt(feat_dim / target_dim)

        q = (attn_input * (self._query_scale * scale_correction)).contiguous()
        k = attn_input.contiguous()
        v = attn_input.contiguous()
        attn = attention_ops.memory_efficient_attention(q, k, v, attn_bias=None, p=0.0)

        attn = attn[..., :feat_dim]
        return attn.to(dtype=original_dtype)

    def _mean_shift_dense(self, grouped: torch.Tensor) -> torch.Tensor:
        scaled_q = grouped * self._query_scale
        scores = torch.matmul(scaled_q, grouped.transpose(-2, -1)) / math.sqrt(grouped.shape[-1])
        weights = scores.softmax(dim=-1)
        return torch.matmul(weights, grouped)

    @staticmethod
    def _compute_kernel_scale(bandwidth_rad: float, half_weight_ratio: float = 0.5) -> float:
        clamped_ratio = min(0.99, max(1e-3, half_weight_ratio))
        clamped_angle = max(1e-3, min(math.pi, bandwidth_rad))
        cos_delta = math.cos(clamped_angle)
        denom = cos_delta - 1.0
        denom = min(-1e-6, denom)
        # Resulting positive scale ensures exp(scale * cos Δθ) drops to half at the requested angle
        return math.log(clamped_ratio) / denom


class PhaseLoss(torch.nn.Module):
    """Encourages KoPE phase vectors to locally synchronize via phase-only self-attention.

    Args:
        consensus_sharpness: temperature-like logit scale applied to the per-head cosine
            similarities before the softmax. Larger values make the loss focus on very
            tight phase agreement, smaller values treat partially aligned tokens more
            uniformly.
    """

    def __init__(self, consensus_sharpness: float = 10.0):
        super().__init__()
        if consensus_sharpness <= 0:
            raise ValueError("consensus_sharpness must be positive")

        self.consensus_sharpness = float(consensus_sharpness)
        self._kernel_scale = self.consensus_sharpness
        self._eps = 1e-6

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        b, s, h, d_half, two = phases.shape
        if two != 2:
            raise ValueError("The last dimension of phases must contain (cos, sin)")

        grouped = phases.reshape(b, s, h, d_half*two)
        # normalize
        grouped  = F.normalize(grouped, dim=-1)

        local_mean = self._mean_shift(grouped)
        mean_norm = torch.linalg.vector_norm(local_mean, dim=-1, keepdim=True).clamp_min(self._eps)
        consensus_dir = local_mean / mean_norm
        alignment = (grouped * consensus_dir).sum(dim=-1).clamp(-1.0, 1.0)
        # Weight tokens by how concentrated their local consensus already is.
        weighted_misalignment = (1.0 - alignment) * mean_norm.squeeze(-1)
        return weighted_misalignment.mean()

    def _mean_shift(self, grouped: torch.Tensor) -> torch.Tensor:
        if attention_ops.XFORMERS_AVAILABLE and grouped.is_cuda:
            return self._mean_shift_xformers(grouped)

        grouped_ = grouped.transpose(1,2).contiguous()  # (B, H, S, D)
        dense_mean = self._mean_shift_dense(grouped_)
        return dense_mean.transpose(1,2).contiguous()

    def _mean_shift_xformers(self, grouped: torch.Tensor) -> torch.Tensor:
        batch_groups, seq_len, heads, feat_dim = grouped.shape

        original_dtype = grouped.dtype

        attn_input = grouped
        if original_dtype not in (torch.float16, torch.bfloat16):
            attn_input = attn_input.to(torch.float16)

        q = attn_input.contiguous()
        k = attn_input.contiguous()
        v = attn_input.contiguous()
        attn = attention_ops.memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, scale=self._kernel_scale)

        return attn.to(dtype=original_dtype)

    def _mean_shift_dense(self, grouped: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(grouped, grouped.transpose(-2, -1)) * self._kernel_scale
        weights = scores.softmax(dim=-1)
        return torch.matmul(weights, grouped)
