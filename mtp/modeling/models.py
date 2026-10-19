"""Encoders, fusion blocks and the models used by the baselines.

All sequence features are (batch, latent_dim, time) with time = TARGET_LEN (150).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtp.config import IMU_DIM


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15):
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
        return x + self.net(x)


class EEGEncoder(nn.Module):
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

        self.band_temporal = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, temporal_filters, kernel_size=(1, 31), padding=(0, 15), bias=False),
                nn.BatchNorm2d(temporal_filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(temporal_filters, temporal_filters, kernel_size=(1, 15), padding=(0, 7), bias=False),
                nn.BatchNorm2d(temporal_filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(dropout),
            )
            for _ in range(n_bands)
        ])

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
        band_feats = []
        for b in range(self.n_bands):
            xb = x[:, b:b + 1, :, :]
            fb = self.band_temporal[b](xb)
            band_feats.append(fb)

        h = torch.cat(band_feats, dim=1)
        h = self.spatial(h)
        h = h.squeeze(2)
        h = self.proj(h)
        h = self.tcn(h)
        h = F.interpolate(h, size=self.target_len, mode="linear", align_corners=False)
        return h


class IMUEncoder(nn.Module):
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
        x = imu.permute(0, 2, 1).contiguous()
        return self.net(x)


class SeqHead(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, 64, kernel_size=1),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, out_dim, kernel_size=1),
        )

    def forward(self, h):
        y = self.net(h)
        return y.permute(0, 2, 1).contiguous()


class EEGOnlyModel(nn.Module):
    def __init__(self, n_bands, n_channels, out_dim, target_len=150, latent_dim=64, dropout=0.20):
        super().__init__()
        self.eeg_encoder = EEGEncoder(n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout)
        self.head = SeqHead(latent_dim, out_dim, dropout)

    def forward(self, eeg):
        h = self.eeg_encoder(eeg)
        y = self.head(h)
        return y


class ConcatFusion(nn.Module):
    """Simple fusion: concatenate EEG and IMU features, then a 1x1 conv and a TCN block."""
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
        return self.net(torch.cat([h_eeg, h_imu], dim=1))


class BidirectionalCrossModalAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Modal Multi-Head Attention fusion for EEG and IMU features.

    Input tensors:
        h_eeg: (batch, latent_dim, time)
        h_imu: (batch, latent_dim, time)

    Fusion logic:
        1) EEG queries IMU: Q=EEG, K/V=IMU
        2) IMU queries EEG: Q=IMU, K/V=EEG
        3) Concatenate original and cross-attended features
        4) Project back to latent_dim and keep an EEG residual path

    The residual EEG path is important for Baseline 6 because MI fine-tuning/testing
    uses zero IMU, so the model should not become completely dependent on IMU.
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
        # Convert from Conv1D format (B, D, T) to token format (B, T, D).
        eeg_tokens = h_eeg.permute(0, 2, 1).contiguous()
        imu_tokens = h_imu.permute(0, 2, 1).contiguous()

        eeg_q = self.eeg_norm(eeg_tokens)
        imu_q = self.imu_norm(imu_tokens)

        # EEG attends to IMU: use EEG as query and IMU as key/value.
        eeg_from_imu, _ = self.eeg_to_imu(
            query=eeg_q,
            key=imu_q,
            value=imu_q,
            need_weights=False,
        )
        eeg_from_imu = self.eeg_cross_norm(eeg_tokens + self.attn_dropout(eeg_from_imu))

        # IMU attends to EEG: use IMU as query and EEG as key/value.
        imu_from_eeg, _ = self.imu_to_eeg(
            query=imu_q,
            key=eeg_q,
            value=eeg_q,
            need_weights=False,
        )
        imu_from_eeg = self.imu_cross_norm(imu_tokens + self.attn_dropout(imu_from_eeg))

        # Fuse both original and cross-modal context features.
        fused_tokens = torch.cat(
            [eeg_tokens, imu_tokens, eeg_from_imu, imu_from_eeg],
            dim=-1,
        )
        fused_tokens = self.fusion_proj(fused_tokens)

        # EEG residual path helps preserve EEG-only decoding ability for MI with zero IMU.
        fused_tokens = fused_tokens + eeg_tokens

        # Convert back to Conv1D format (B, D, T), then refine temporally.
        fused = fused_tokens.permute(0, 2, 1).contiguous()
        return self.out(fused)


class NormalMultiHeadAttentionFusion(nn.Module):
    """
    Normal Multi-Head Attention fusion for EEG and IMU features.

    Input tensors:
        h_eeg: (batch, latent_dim, time)
        h_imu: (batch, latent_dim, time)

    Fusion logic:
        1) Concatenate EEG and IMU latent features along the channel dimension
        2) Project concatenated features back to latent_dim
        3) Apply normal multi-head self-attention with Q=K=V=fused tokens
        4) Refine with a small feed-forward block and TCNBlock

    Difference from Baseline 6:
        - Baseline 6 uses cross-modal attention: EEG queries IMU and IMU queries EEG.
        - Baseline 7 uses normal self-attention after initial EEG+IMU feature fusion.

    The EEG residual path helps preserve an EEG-only route when MI uses zero IMU.
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
        # First make one fused latent stream in Conv1D format: (B, D, T).
        h0 = self.input_proj(torch.cat([h_eeg, h_imu], dim=1))

        # Convert to token format for PyTorch MultiheadAttention: (B, T, D).
        tokens = h0.permute(0, 2, 1).contiguous()

        # Normal MHA: Q, K, and V all come from the same fused EEG+IMU tokens.
        attn_in = self.attn_norm(tokens)
        attn_out, _ = self.self_attn(
            query=attn_in,
            key=attn_in,
            value=attn_in,
            need_weights=False,
        )
        tokens = tokens + self.attn_dropout(attn_out)

        # Small Transformer-style feed-forward refinement.
        ffn_in = self.ffn_norm(tokens)
        tokens = tokens + self.ffn(ffn_in)

        # EEG residual path helps MI fine-tuning/testing with zero IMU.
        eeg_tokens = h_eeg.permute(0, 2, 1).contiguous()
        tokens = tokens + eeg_tokens

        # Convert back to Conv1D format and refine temporally.
        fused = tokens.permute(0, 2, 1).contiguous()
        return self.out(fused)


FUSION_BLOCKS = ("concat", "cross_mha", "self_mha")


def make_fusion_block(fusion: str, latent_dim: int, num_heads: int = 4, dropout: float = 0.20):
    if fusion == "concat":
        return ConcatFusion(latent_dim, dropout)
    if fusion == "cross_mha":
        return BidirectionalCrossModalAttentionFusion(latent_dim, num_heads, dropout)
    if fusion == "self_mha":
        return NormalMultiHeadAttentionFusion(latent_dim, num_heads, dropout)
    raise ValueError(f"Unknown fusion {fusion!r}; expected one of {FUSION_BLOCKS}")


class FusionModel(nn.Module):
    """
    Baselines 3, 6 and 7: EEG + IMU encoders -> fusion block -> trajectory head.

    ME uses the real IMU; MI uses the same model with an all-zero IMU input.
    fusion: "concat" (Baseline 3), "cross_mha" (Baseline 6) or "self_mha" (Baseline 7).
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
        self.eeg_encoder = EEGEncoder(n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout)
        self.imu_encoder = IMUEncoder(IMU_DIM, latent_dim, dropout)
        self.fusion = make_fusion_block(fusion, latent_dim, num_heads, dropout)
        self.head = SeqHead(latent_dim, out_dim, dropout)

    def forward(self, eeg, imu):
        h_eeg = self.eeg_encoder(eeg)
        h_imu = self.imu_encoder(imu)
        h = self.fusion(h_eeg, h_imu)
        return self.head(h)


class DistillationSimpleModel(nn.Module):
    """
    Baseline 4:
    Teacher: EEG+IMU simple fusion
    Student: EEG only
    """
    def __init__(self, n_bands, n_channels, out_dim, target_len=150, latent_dim=64, dropout=0.20):
        super().__init__()
        self.eeg_encoder = EEGEncoder(n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout)
        self.imu_encoder = IMUEncoder(IMU_DIM, latent_dim, dropout)
        self.fusion = ConcatFusion(latent_dim, dropout)
        self.teacher_head = SeqHead(latent_dim, out_dim, dropout)
        self.student_head = SeqHead(latent_dim, out_dim, dropout)

    def forward_teacher_student(self, eeg, imu):
        h_eeg = self.eeg_encoder(eeg)
        h_imu = self.imu_encoder(imu)
        h_teacher = self.fusion(h_eeg, h_imu)
        y_teacher = self.teacher_head(h_teacher)
        y_student = self.student_head(h_eeg)
        return y_teacher, y_student

    def forward_student(self, eeg):
        h_eeg = self.eeg_encoder(eeg)
        return self.student_head(h_eeg)


class SubjectFiLM(nn.Module):
    def __init__(self, n_subjects: int, latent_dim: int):
        super().__init__()
        self.gamma = nn.Embedding(n_subjects, latent_dim)
        self.beta = nn.Embedding(n_subjects, latent_dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, h, subject_id):
        gamma = self.gamma(subject_id).unsqueeze(-1)
        beta = self.beta(subject_id).unsqueeze(-1)
        return gamma * h + beta


class LagAwareFusion(nn.Module):
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
        self.eeg_encoder = EEGEncoder(n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout)
        self.subject_film = SubjectFiLM(n_subjects, latent_dim)
        self.imu_encoder = IMUEncoder(IMU_DIM, latent_dim, dropout)
        self.fusion = LagAwareFusion(latent_dim, lag_radius, dropout)
        self.teacher_head = SeqHead(latent_dim, out_dim, dropout)
        self.student_head = SeqHead(latent_dim, out_dim, dropout)

    def encode_eeg_student(self, eeg, subject_id):
        h = self.eeg_encoder(eeg)
        h = self.subject_film(h, subject_id)
        return h

    def forward_teacher_student(self, eeg, imu, subject_id):
        h_eeg = self.encode_eeg_student(eeg, subject_id)
        h_imu = self.imu_encoder(imu)
        h_teacher = self.fusion(h_eeg, h_imu)
        y_teacher = self.teacher_head(h_teacher)
        y_student = self.student_head(h_eeg)
        return y_teacher, y_student, h_eeg, h_teacher

    def forward_student(self, eeg, subject_id, return_latent: bool = False):
        h_eeg = self.encode_eeg_student(eeg, subject_id)
        y_student = self.student_head(h_eeg)
        if return_latent:
            return y_student, h_eeg
        return y_student


def set_full_student_finetune_mode(model: FullPrivilegedModel):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.eeg_encoder.parameters():
        p.requires_grad = True
    for p in model.subject_film.parameters():
        p.requires_grad = True
    for p in model.student_head.parameters():
        p.requires_grad = True


def set_distill_student_finetune_mode(model: DistillationSimpleModel):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.eeg_encoder.parameters():
        p.requires_grad = True
    for p in model.student_head.parameters():
        p.requires_grad = True
