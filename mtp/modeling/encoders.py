"""The EEG and IMU encoders, and the TCN block they are built from.

The encoders return features of shape (batch, latent_dim, n_times), where n_times is
150 to match the 30 Hz velocity targets.
"""

import torch
from torch import nn
import torch.nn.functional as F


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
        self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.20
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
        return h @ linear_resample_matrix(h.shape[-1], self.target_len).to(h)


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
