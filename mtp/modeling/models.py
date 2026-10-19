"""The models of the deep baselines.

Each model combines the encoders in `mtp.modeling.encoders`, the fusion blocks in
`mtp.modeling.fusion` and a `SeqHead` that turns features into velocities of shape
(batch, 150, n_outputs).

- `EEGOnlyModel`: Baselines 1 and 2.
- `FusionModel`: Baselines 3, 6 and 7.
- `DistillationSimpleModel`: Baseline 4.
- `FullPrivilegedModel`: Baseline 5.
"""

from torch import nn

from mtp.config import IMU_DIM
from mtp.modeling.encoders import EEGEncoder, IMUEncoder
from mtp.modeling.fusion import ConcatFusion, LagAwareFusion, make_fusion_block


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
    num_heads : int or None
        Number of attention heads. Not used by "concat", which accepts None.
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
    It has the same methods as `FullPrivilegedModel`, so the same training code works for
    both; the subject IDs are accepted but not used.

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

    def forward_teacher_student(self, eeg, imu, subject_id=None):
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

    def forward_student(self, eeg, subject_id=None):
        """Predict from the EEG alone with the student head."""
        h_eeg = self.eeg_encoder(eeg)
        return self.student_head(h_eeg)


class SubjectFiLM(nn.Module):
    """Scale and shift EEG features with values learned for each subject.

    The scale starts at 1 and the shift at 0, so the layer starts as an identity. Each
    run currently trains on one subject, so in practice this is a single learned scale
    and shift for all trials. It is kept so the model can be trained on several subjects
    at once later.

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
        """
        h_eeg = self.encode_eeg_student(eeg, subject_id)
        h_imu = self.imu_encoder(imu)
        h_teacher = self.fusion(h_eeg, h_imu)
        y_teacher = self.teacher_head(h_teacher)
        y_student = self.student_head(h_eeg)
        return y_teacher, y_student

    def forward_student(self, eeg, subject_id):
        """Predict from the EEG alone with the student head."""
        return self.student_head(self.encode_eeg_student(eeg, subject_id))

    def forward_student_with_features(self, eeg, subject_id):
        """Predict with the student head and also return the EEG features it used.

        Returns
        -------
        y_student : Tensor, shape (batch, target_len, out_dim)
        h_eeg : Tensor, shape (batch, latent_dim, target_len)
        """
        h_eeg = self.encode_eeg_student(eeg, subject_id)
        return self.student_head(h_eeg), h_eeg


def freeze_all_except(model: nn.Module, *modules: nn.Module):
    """Freeze every parameter of `model` except those of the given submodules."""
    for p in model.parameters():
        p.requires_grad = False
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True
