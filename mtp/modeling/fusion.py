"""Blocks that fuse EEG and IMU features into one feature map.

Each block takes the EEG and IMU features, both of shape (batch, latent_dim, n_times),
and returns fused features of the same shape. Baseline 3 uses `ConcatFusion`, Baseline
6 `BidirectionalCrossModalAttentionFusion`, Baseline 7 `NormalMultiHeadAttentionFusion`
and Baseline 5 `LagAwareFusion`.
"""

import math

import torch
from torch import nn

from mtp.modeling.encoders import TCNBlock


class ConcatFusion(nn.Module):
    """Fuse EEG and IMU features by concatenation, a 1x1 convolution and a TCN block.

    Parameters
    ----------
    latent_dim : int
        Number of features in each input and in the output.
    dropout : float
        Dropout rate.
    """

    def __init__(self, latent_dim: int, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim * 2, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
        )

    def forward(self, h_eeg, h_imu):
        """Fuse two feature maps of shape (batch, latent_dim, n_times) into one."""
        return self.net(torch.cat([h_eeg, h_imu], dim=1))


class BidirectionalCrossModalAttentionFusion(nn.Module):
    """Fuse EEG and IMU features with attention in both directions (Baseline 6).

    The EEG features attend to the IMU features and the IMU features attend to the EEG
    features. The original and attended features of both are concatenated, projected
    back to `latent_dim` and added to the EEG features, and a TCN block follows.

    The EEG features are added back because MI fine-tuning and testing use an all-zero
    IMU input, so the model must not come to rely on the IMU entirely.

    Parameters
    ----------
    latent_dim : int
        Number of features. Must be divisible by `num_heads`.
    num_heads : int
        Number of attention heads in each direction.
    dropout : float
        Dropout rate.
    """

    def __init__(self, latent_dim: int, num_heads: int = 4, dropout: float = 0.20):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.eeg_to_imu = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.imu_to_eeg = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.eeg_norm = nn.LayerNorm(latent_dim)
        self.imu_norm = nn.LayerNorm(latent_dim)
        self.eeg_cross_norm = nn.LayerNorm(latent_dim)
        self.imu_cross_norm = nn.LayerNorm(latent_dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.fusion_proj = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.out = nn.Sequential(
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
        )

    def forward(self, h_eeg, h_imu):
        """Fuse two feature maps of shape (batch, latent_dim, n_times) into one."""
        # nn.MultiheadAttention works on (batch, n_times, latent_dim).
        eeg_tokens = h_eeg.permute(0, 2, 1).contiguous()
        imu_tokens = h_imu.permute(0, 2, 1).contiguous()

        eeg_q = self.eeg_norm(eeg_tokens)
        imu_q = self.imu_norm(imu_tokens)

        eeg_from_imu, _ = self.eeg_to_imu(
            query=eeg_q,
            key=imu_q,
            value=imu_q,
            need_weights=False,
        )
        eeg_from_imu = self.eeg_cross_norm(eeg_tokens + self.attn_dropout(eeg_from_imu))

        imu_from_eeg, _ = self.imu_to_eeg(
            query=imu_q,
            key=eeg_q,
            value=eeg_q,
            need_weights=False,
        )
        imu_from_eeg = self.imu_cross_norm(imu_tokens + self.attn_dropout(imu_from_eeg))

        fused_tokens = torch.cat(
            [eeg_tokens, imu_tokens, eeg_from_imu, imu_from_eeg],
            dim=-1,
        )
        fused_tokens = self.fusion_proj(fused_tokens)

        fused_tokens = fused_tokens + eeg_tokens

        fused = fused_tokens.permute(0, 2, 1).contiguous()
        return self.out(fused)


class NormalMultiHeadAttentionFusion(nn.Module):
    """Fuse EEG and IMU features with self-attention (Baseline 7).

    The EEG and IMU features are concatenated and projected into one feature map.
    Multi-head self-attention and a small feed-forward layer refine it over time, the
    EEG features are added back, and a TCN block follows.

    Unlike `BidirectionalCrossModalAttentionFusion`, the attention runs over the already
    fused features rather than between the two signals. The EEG features are added back
    for the same reason: MI uses an all-zero IMU input.

    Parameters
    ----------
    latent_dim : int
        Number of features. Must be divisible by `num_heads`.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(self, latent_dim: int, num_heads: int = 4, dropout: float = 0.20):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.input_proj = nn.Sequential(
            nn.Conv1d(latent_dim * 2, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(latent_dim)
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Dropout(dropout),
        )

        self.out = nn.Sequential(
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
        )

    def forward(self, h_eeg, h_imu):
        """Fuse two feature maps of shape (batch, latent_dim, n_times) into one."""
        h0 = self.input_proj(torch.cat([h_eeg, h_imu], dim=1))

        # nn.MultiheadAttention works on (batch, n_times, latent_dim).
        tokens = h0.permute(0, 2, 1).contiguous()

        attn_in = self.attn_norm(tokens)
        attn_out, _ = self.self_attn(
            query=attn_in,
            key=attn_in,
            value=attn_in,
            need_weights=False,
        )
        tokens = tokens + self.attn_dropout(attn_out)

        ffn_in = self.ffn_norm(tokens)
        tokens = tokens + self.ffn(ffn_in)

        eeg_tokens = h_eeg.permute(0, 2, 1).contiguous()
        tokens = tokens + eeg_tokens

        fused = tokens.permute(0, 2, 1).contiguous()
        return self.out(fused)


FUSION_BLOCKS = ("concat", "cross_mha", "self_mha")


def make_fusion_block(fusion: str, latent_dim: int, num_heads=4, dropout: float = 0.20):
    """Create the fusion block named `fusion`: "concat", "cross_mha" or "self_mha".

    `num_heads` is only used by the attention blocks and may be None for "concat".
    """
    if fusion == "concat":
        return ConcatFusion(latent_dim, dropout)
    if fusion == "cross_mha":
        return BidirectionalCrossModalAttentionFusion(latent_dim, num_heads, dropout)
    if fusion == "self_mha":
        return NormalMultiHeadAttentionFusion(latent_dim, num_heads, dropout)
    raise ValueError(f"Unknown fusion {fusion!r}; expected one of {FUSION_BLOCKS}")


class LagAwareFusion(nn.Module):
    """Fuse EEG and IMU features while allowing for a time lag between them (Baseline 5).

    At each time step, the EEG features are compared with the IMU features shifted by
    -lag_radius to +lag_radius steps. A softmax over these lags weights the shifted IMU
    features into one context vector. A learned gate decides how much of the context is
    added to the EEG features, and a TCN block follows.

    Parameters
    ----------
    latent_dim : int
        Number of features.
    lag_radius : int
        Largest shift in time steps. At 30 Hz, 3 steps is 100 ms.
    dropout : float
        Dropout rate.
    """

    def __init__(self, latent_dim: int, lag_radius: int = 3, dropout: float = 0.20):
        super().__init__()
        self.latent_dim = latent_dim
        self.lag_radius = lag_radius
        self.lags = list(range(-lag_radius, lag_radius + 1))

        self.q_proj = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)
        self.k_proj = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)
        self.v_proj = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)

        self.context_proj = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Sequential(
            nn.Conv1d(latent_dim * 3, latent_dim, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(latent_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
        )

    @staticmethod
    def shift_time(x, lag: int):
        """Shift x along its last (time) axis by `lag` steps, filling the gap with zeros.

        A positive lag moves later values earlier: ``out[..., t] = x[..., t + lag]``.
        """
        if lag == 0:
            return x
        out = torch.zeros_like(x)
        if lag > 0:
            out[:, :, :-lag] = x[:, :, lag:]
        else:
            lag_abs = -lag
            out[:, :, lag_abs:] = x[:, :, :-lag_abs]
        return out

    def forward(self, h_eeg, h_imu):
        """Fuse two feature maps of shape (batch, latent_dim, n_times) into one."""
        D = h_eeg.shape[1]
        q = self.q_proj(h_eeg)
        k = self.k_proj(h_imu)
        v = self.v_proj(h_imu)

        k_stack = torch.stack([self.shift_time(k, lag) for lag in self.lags], dim=1)
        v_stack = torch.stack([self.shift_time(v, lag) for lag in self.lags], dim=1)

        scores = (q.unsqueeze(1) * k_stack).sum(dim=2) / math.sqrt(D)
        alpha = torch.softmax(scores, dim=1)
        context = (alpha.unsqueeze(2) * v_stack).sum(dim=1)
        context = self.context_proj(context)

        gate_in = torch.cat([h_eeg, h_imu, context], dim=1)
        g = self.gate(gate_in)
        h_fused = h_eeg + g * context
        return self.out(h_fused)
