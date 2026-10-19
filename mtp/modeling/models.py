"""Encoders, fusion blocks and the models of the deep baselines.

Inside the models, sequence features have shape (batch, latent_dim, n_times), where
n_times is 150 to match the 30 Hz velocity targets. Velocity outputs have shape
(batch, 150, n_outputs).
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from mtp.config import IMU_DIM


def linear_resample_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """Return a matrix that linearly resamples the last axis of a tensor.

    ``x @ W`` gives the same result as
    ``F.interpolate(x, size=n_out, mode="linear", align_corners=False)``. W is made by
    interpolating an identity matrix, so it uses PyTorch's own weights. The EEG encoder
    multiplies by W instead of calling F.interpolate because F.interpolate has no
    deterministic backward pass on the GPU.

    Parameters
    ----------
    n_in, n_out : int
        Number of points before and after resampling.

    Returns
    -------
    Tensor, shape (n_in, n_out)
    """
    with torch.no_grad():
        identity = torch.eye(n_in).unsqueeze(0)
        return F.interpolate(identity, size=n_out, mode="linear", align_corners=False)[0]


class TCNBlock(nn.Module):
    """Two dilated 1-D convolutions with a residual connection.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    kernel_size : int
        Convolution kernel size.
    dilation : int
        Spacing between kernel taps. Larger values look further back and ahead in time.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Apply the block to x of shape (batch, channels, n_times); the shape is kept."""
        return x + self.net(x)


class EEGEncoder(nn.Module):
    """Turn filter-bank EEG into features at the velocity sampling rate.

    Each band first goes through its own temporal convolutions and pooling. The bands
    are then stacked, a spatial convolution combines all channels, and four dilated TCN
    blocks model the time course. With 5 s of EEG at 512 Hz the pooling leaves 320 time
    steps, which are resampled linearly to `target_len`.

    Parameters
    ----------
    n_bands : int
        Number of filter-bank bands.
    n_channels : int
        Number of EEG channels.
    target_len : int
        Number of time steps in the output.
    temporal_filters : int
        Temporal filters per band.
    spatial_filters : int
        Spatial filters across channels.
    latent_dim : int
        Number of output features.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_bands: int,
        n_channels: int,
        target_len: int = 150,
        temporal_filters: int = 12,
        spatial_filters: int = 32,
        latent_dim: int = 64,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.target_len = target_len
        # Built on the first forward pass, when the input length is known.
        self._resample = None

        self.band_temporal = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        1, temporal_filters, kernel_size=(1, 31), padding=(0, 15), bias=False
                    ),
                    nn.BatchNorm2d(temporal_filters),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
                    nn.Dropout(dropout),
                    nn.Conv2d(
                        temporal_filters,
                        temporal_filters,
                        kernel_size=(1, 15),
                        padding=(0, 7),
                        bias=False,
                    ),
                    nn.BatchNorm2d(temporal_filters),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
                    nn.Dropout(dropout),
                )
                for _ in range(n_bands)
            ]
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=temporal_filters * n_bands,
                out_channels=spatial_filters,
                kernel_size=(n_channels, 1),
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.proj = nn.Sequential(
            nn.Conv1d(spatial_filters, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.tcn = nn.Sequential(
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=2, dropout=dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=4, dropout=dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=8, dropout=dropout),
        )

    def forward(self, x):
        """Encode filter-bank EEG.

        Parameters
        ----------
        x : Tensor, shape (batch, n_bands, n_channels, n_samples)
            EEG from `make_eeg_input`.

        Returns
        -------
        Tensor, shape (batch, latent_dim, target_len)
        """
        band_feats = []
        for b in range(self.n_bands):
            xb = x[:, b : b + 1, :, :]
            fb = self.band_temporal[b](xb)
            band_feats.append(fb)

        h = torch.cat(band_feats, dim=1)
        h = self.spatial(h)
        h = h.squeeze(2)
        h = self.proj(h)
        h = self.tcn(h)
        return h @ self._resample_matrix(h)

    def _resample_matrix(self, h):
        W = self._resample
        if W is None or W.shape[0] != h.shape[-1] or W.device != h.device or W.dtype != h.dtype:
            W = linear_resample_matrix(h.shape[-1], self.target_len).to(
                device=h.device, dtype=h.dtype
            )
            self._resample = W
        return W


class IMUEncoder(nn.Module):
    """Turn the IMU signals into features.

    Parameters
    ----------
    in_dim : int
        Number of IMU signals.
    latent_dim : int
        Number of output features.
    dropout : float
        Dropout rate.
    """

    def __init__(self, in_dim: int = 6, latent_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, latent_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=2, dropout=dropout),
        )

    def forward(self, imu):
        """Encode the IMU signals.

        Parameters
        ----------
        imu : Tensor, shape (batch, n_times, in_dim)

        Returns
        -------
        Tensor, shape (batch, latent_dim, n_times)
        """
        x = imu.permute(0, 2, 1).contiguous()
        return self.net(x)


class SeqHead(nn.Module):
    """Map features to velocities with two 1x1 convolutions.

    Parameters
    ----------
    latent_dim : int
        Number of input features.
    out_dim : int
        Number of velocity outputs.
    dropout : float
        Dropout rate.
    """

    def __init__(self, latent_dim: int, out_dim: int, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, 64, kernel_size=1),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, out_dim, kernel_size=1),
        )

    def forward(self, h):
        """Map features to velocities.

        Parameters
        ----------
        h : Tensor, shape (batch, latent_dim, n_times)

        Returns
        -------
        Tensor, shape (batch, n_times, out_dim)
        """
        y = self.net(h)
        return y.permute(0, 2, 1).contiguous()


class EEGOnlyModel(nn.Module):
    """EEG encoder followed by a head, used by Baselines 1 and 2.

    Parameters
    ----------
    n_bands, n_channels : int
        Number of filter-bank bands and EEG channels in the input.
    out_dim : int
        Number of velocity outputs.
    target_len : int
        Number of output time steps.
    latent_dim : int
        Number of feature channels.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_bands, n_channels, out_dim, target_len=150, latent_dim=64, dropout=0.20):
        super().__init__()
        self.eeg_encoder = EEGEncoder(
            n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout
        )
        self.head = SeqHead(latent_dim, out_dim, dropout)

    def forward(self, eeg):
        """Predict velocities, shape (batch, target_len, out_dim), from filter-bank EEG."""
        h = self.eeg_encoder(eeg)
        y = self.head(h)
        return y


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


def make_fusion_block(fusion: str, latent_dim: int, num_heads: int = 4, dropout: float = 0.20):
    """Create the fusion block named `fusion`: "concat", "cross_mha" or "self_mha"."""
    if fusion == "concat":
        return ConcatFusion(latent_dim, dropout)
    if fusion == "cross_mha":
        return BidirectionalCrossModalAttentionFusion(latent_dim, num_heads, dropout)
    if fusion == "self_mha":
        return NormalMultiHeadAttentionFusion(latent_dim, num_heads, dropout)
    raise ValueError(f"Unknown fusion {fusion!r}; expected one of {FUSION_BLOCKS}")


class FusionModel(nn.Module):
    """EEG and IMU model of Baselines 3, 6 and 7.

    An EEG encoder and an IMU encoder feed a fusion block, and a head turns the fused
    features into velocities. ME trials use the recorded IMU; MI trials use the same
    model with an all-zero IMU input.

    Parameters
    ----------
    fusion : {"concat", "cross_mha", "self_mha"}
        Fusion block of Baseline 3, 6 or 7.
    n_bands, n_channels : int
        Number of filter-bank bands and EEG channels in the input.
    out_dim : int
        Number of velocity outputs.
    target_len : int
        Number of output time steps.
    latent_dim : int
        Number of feature channels.
    num_heads : int
        Number of attention heads. Not used by "concat".
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        fusion,
        n_bands,
        n_channels,
        out_dim,
        target_len=150,
        latent_dim=64,
        num_heads=4,
        dropout=0.20,
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder(
            n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout
        )
        self.imu_encoder = IMUEncoder(IMU_DIM, latent_dim, dropout)
        self.fusion = make_fusion_block(fusion, latent_dim, num_heads, dropout)
        self.head = SeqHead(latent_dim, out_dim, dropout)

    def forward(self, eeg, imu):
        """Predict velocities from EEG and IMU.

        Parameters
        ----------
        eeg : Tensor, shape (batch, n_bands, n_channels, n_samples)
        imu : Tensor, shape (batch, target_len, 6)

        Returns
        -------
        Tensor, shape (batch, target_len, out_dim)
        """
        h_eeg = self.eeg_encoder(eeg)
        h_imu = self.imu_encoder(imu)
        h = self.fusion(h_eeg, h_imu)
        return self.head(h)


class DistillationSimpleModel(nn.Module):
    """Teacher-student model of Baseline 4.

    One EEG encoder feeds two heads. The teacher head sees the EEG features fused with
    the IMU features by `ConcatFusion`; the student head sees only the EEG features.

    Parameters
    ----------
    n_bands, n_channels : int
        Number of filter-bank bands and EEG channels in the input.
    out_dim : int
        Number of velocity outputs.
    target_len : int
        Number of output time steps.
    latent_dim : int
        Number of feature channels.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_bands, n_channels, out_dim, target_len=150, latent_dim=64, dropout=0.20):
        super().__init__()
        self.eeg_encoder = EEGEncoder(
            n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout
        )
        self.imu_encoder = IMUEncoder(IMU_DIM, latent_dim, dropout)
        self.fusion = ConcatFusion(latent_dim, dropout)
        self.teacher_head = SeqHead(latent_dim, out_dim, dropout)
        self.student_head = SeqHead(latent_dim, out_dim, dropout)

    def forward_teacher_student(self, eeg, imu):
        """Predict with both heads.

        Returns
        -------
        y_teacher, y_student : Tensor, shape (batch, target_len, out_dim)
        """
        h_eeg = self.eeg_encoder(eeg)
        h_imu = self.imu_encoder(imu)
        h_teacher = self.fusion(h_eeg, h_imu)
        y_teacher = self.teacher_head(h_teacher)
        y_student = self.student_head(h_eeg)
        return y_teacher, y_student

    def forward_student(self, eeg):
        """Predict from the EEG alone with the student head."""
        h_eeg = self.eeg_encoder(eeg)
        return self.student_head(h_eeg)


class SubjectFiLM(nn.Module):
    """Scale and shift EEG features with values learned for each subject.

    The scale starts at 1 and the shift at 0, so the layer starts as an identity.

    Parameters
    ----------
    n_subjects : int
        Number of subjects with their own scale and shift.
    latent_dim : int
        Number of features.
    """

    def __init__(self, n_subjects: int, latent_dim: int):
        super().__init__()
        self.gamma = nn.Embedding(n_subjects, latent_dim)
        self.beta = nn.Embedding(n_subjects, latent_dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, h, subject_id):
        """Apply each trial's subject scale and shift.

        Parameters
        ----------
        h : Tensor, shape (batch, latent_dim, n_times)
            EEG features.
        subject_id : Tensor of int, shape (batch,)
            Subject index of each trial.
        """
        gamma = self.gamma(subject_id).unsqueeze(-1)
        beta = self.beta(subject_id).unsqueeze(-1)
        return gamma * h + beta


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


class FullPrivilegedModel(nn.Module):
    """Teacher-student model of Baseline 5.

    Like `DistillationSimpleModel`, but the EEG features first get a per-subject scale
    and shift (`SubjectFiLM`), and the teacher fuses EEG and IMU with `LagAwareFusion`.

    Parameters
    ----------
    n_bands, n_channels : int
        Number of filter-bank bands and EEG channels in the input.
    out_dim : int
        Number of velocity outputs.
    n_subjects : int
        Number of subjects for `SubjectFiLM`.
    target_len : int
        Number of output time steps.
    latent_dim : int
        Number of feature channels.
    lag_radius : int
        Largest IMU shift in `LagAwareFusion`, in time steps.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_bands: int,
        n_channels: int,
        out_dim: int,
        n_subjects: int = 1,
        target_len: int = 150,
        latent_dim: int = 64,
        lag_radius: int = 3,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder(
            n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout
        )
        self.subject_film = SubjectFiLM(n_subjects, latent_dim)
        self.imu_encoder = IMUEncoder(IMU_DIM, latent_dim, dropout)
        self.fusion = LagAwareFusion(latent_dim, lag_radius, dropout)
        self.teacher_head = SeqHead(latent_dim, out_dim, dropout)
        self.student_head = SeqHead(latent_dim, out_dim, dropout)

    def encode_eeg_student(self, eeg, subject_id):
        """Encode the EEG and apply the subject's scale and shift."""
        h = self.eeg_encoder(eeg)
        h = self.subject_film(h, subject_id)
        return h

    def forward_teacher_student(self, eeg, imu, subject_id):
        """Predict with both heads.

        Returns
        -------
        y_teacher, y_student : Tensor, shape (batch, target_len, out_dim)
            Predictions of the teacher and the student.
        h_eeg, h_teacher : Tensor, shape (batch, latent_dim, target_len)
            The student's EEG features and the teacher's fused features.
        """
        h_eeg = self.encode_eeg_student(eeg, subject_id)
        h_imu = self.imu_encoder(imu)
        h_teacher = self.fusion(h_eeg, h_imu)
        y_teacher = self.teacher_head(h_teacher)
        y_student = self.student_head(h_eeg)
        return y_teacher, y_student, h_eeg, h_teacher

    def forward_student(self, eeg, subject_id, return_latent: bool = False):
        """Predict from the EEG alone with the student head.

        If `return_latent` is True, the EEG features, of shape
        (batch, latent_dim, target_len), are returned too.
        """
        h_eeg = self.encode_eeg_student(eeg, subject_id)
        y_student = self.student_head(h_eeg)
        if return_latent:
            return y_student, h_eeg
        return y_student


def set_full_student_finetune_mode(model: FullPrivilegedModel):
    """Freeze everything except the EEG encoder, the SubjectFiLM layer and the student head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.eeg_encoder.parameters():
        p.requires_grad = True
    for p in model.subject_film.parameters():
        p.requires_grad = True
    for p in model.student_head.parameters():
        p.requires_grad = True


def set_distill_student_finetune_mode(model: DistillationSimpleModel):
    """Freeze everything except the EEG encoder and the student head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.eeg_encoder.parameters():
        p.requires_grad = True
    for p in model.student_head.parameters():
        p.requires_grad = True
