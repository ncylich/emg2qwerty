from collections.abc import Sequence

import torch
from torch import nn
import torchaudio
import numpy as np
from typing import Optional
import math
import torch.nn.functional as F

from emg2qwerty.modules import TDSConv2dBlock

"""
CONFORMER_MODULES: Shared modules for the conformer-based models
"""


class DctLogSpectrogram(nn.Module):
    """
    Computes log spectrogram features from input signals by applying a DCT
    along the frequency axis. Expects inputs of shape (T, N, bands, C, F)
    and returns outputs of shape (T, N, bands, C, freq), where it is assumed
    that bands==2 and C==16.

    Args:
        n_dct (int): Window size for unfolding along frequency axis.
                     Also the size of the DCT window.
        hop_length (int): Step size for unfolding.
        log_offset (float): Small constant added for numerical stability.
    """

    def __init__(
            self,
            n_dct: int = 64,
            hop_length: int = 16,
            log_offset: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_dct = n_dct
        self.hop_length = hop_length
        self.log_offset = log_offset

        self._initialize_combined_dct_hann_matrix()

    def _initialize_combined_dct_hann_matrix(self) -> None:
        n = self.n_dct
        dct_mat = torch.empty(n, n)
        for k in range(n):
            for i in range(n):
                dct_mat[k, i] = math.cos(math.pi * k * (2 * i + 1) / (2 * n))
        dct_mat[0] *= 1.0 / math.sqrt(n)
        dct_mat[1:] *= math.sqrt(2.0 / n)
        combined_dct_hann = dct_mat * torch.hann_window(n).unsqueeze(1)  # Same as diag(window) @ dct_mat
        self.register_buffer("combined_dct_hann", combined_dct_hann)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Tensor of shape (T, N, bands, C, F)
        Returns:
            torch.Tensor: Tensor of shape (T, N, bands, C, freq)
        """
        permuted_inputs = inputs.movedim(0, -1)
        x_unfold = permuted_inputs.unfold(dimension=-1, size=self.n_dct, step=self.hop_length)

        # Combined windowing & DCT mult
        dct_coeffs = torch.matmul(x_unfold, self.combined_dct_hann)

        # Energy + log
        spec = dct_coeffs.pow(2)
        logspec = torch.log10(spec + self.log_offset)

        permuted_logspec = logspec.permute(-2, 0, 1, 2, -1)
        return permuted_logspec  # (T, N, bands, C, freq)


class SpecAugment(nn.Module):
    """
    Applies time and frequency masking as described in
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    (https://arxiv.org/abs/1904.08779).

    This module expects inputs of shape (T, N, bands, C, freq) and returns
    outputs of the same shape.

    Args:
        n_time_masks (int): Maximum number of time masks to apply (default: 0).
        time_mask_param (int): Maximum possible width of each time mask (default: 0).
        iid_time_masks (bool): Whether to apply different time masks for each channel (default: True).
        n_freq_masks (int): Maximum number of frequency masks to apply (default: 0).
        freq_mask_param (int): Maximum possible width of each frequency mask (default: 0).
        iid_freq_masks (bool): Whether to apply different frequency masks for each channel (default: True).
        mask_value (float): The value to fill in the masked areas (default: 0.0).
    """
    def __init__(
        self,
        n_time_masks: int = 0,
        time_mask_param: int = 0,
        iid_time_masks: bool = True,
        n_freq_masks: int = 0,
        freq_mask_param: int = 0,
        iid_freq_masks: bool = True,
        mask_value: float = 0.0,
    ):
        super().__init__()
        self.n_time_masks = n_time_masks
        self.time_mask_param = time_mask_param
        self.iid_time_masks = iid_time_masks
        self.n_freq_masks = n_freq_masks
        self.freq_mask_param = freq_mask_param
        self.iid_freq_masks = iid_freq_masks
        self.mask_value = mask_value

        # Initialize torchaudio masking transforms
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param, iid_masks=iid_time_masks)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param, iid_masks=iid_freq_masks)

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            specgram (torch.Tensor): Input tensor of shape (T, N, bands, C, freq)
        Returns:
            torch.Tensor: Output tensor of shape (T, N, bands, C, freq)
        """
        # Move the time dimension to the end so that we can mask over time
        # Input shape: (T, N, bands, C, freq) -> (N, bands, C, freq, T)
        x = specgram.movedim(0, -1)

        # Determine the number of time masks to apply using torch.randint
        n_t_masks = torch.randint(0, self.n_time_masks + 1, (1,)).item()
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Determine the number of frequency masks to apply
        n_f_masks = torch.randint(0, self.n_freq_masks + 1, (1,)).item()
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # Return the tensor to original shape: (N, bands, C, freq, T) -> (T, N, bands, C, freq)
        return x.movedim(-1, 0)


# Positional Encoding for transformer inputs
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# Swish activation function
class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


# Convolution module for the Conformer block
class ConformerConvModule(nn.Module):
    def __init__(
            self,
            d_model: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Pointwise Conv + GLU
        self.pointwise_conv1 = nn.Conv1d(
            d_model,
            d_model * expansion_factor * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.glu = nn.GLU(dim=1)

        # 1D Depthwise Conv
        self.depthwise_conv = nn.Conv1d(
            d_model * expansion_factor,
            d_model * expansion_factor,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model * expansion_factor,
            bias=True
        )

        # Batch norm
        self.batch_norm = nn.BatchNorm1d(d_model * expansion_factor)

        # Activation
        self.swish = Swish()

        # Pointwise Conv
        self.pointwise_conv2 = nn.Conv1d(
            d_model * expansion_factor,
            d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (T, N, d_model)
        Returns:
            Tensor with shape (T, N, d_model)
        """
        # Convert to (N, d_model, T) for convolution
        x = x.permute(1, 2, 0)

        # LayerNorm -> GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)

        # Pointwise Conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # Convert back to (T, N, d_model)
        x = x.permute(2, 0, 1)

        return x


# Feed Forward module for the Conformer block
class FeedForwardModule(nn.Module):
    def __init__(
            self,
            d_model: int,
            expansion_factor: int = 4,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (T, N, d_model)
        Returns:
            Output tensor with shape (T, N, d_model)
        """
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x * 0.5 + residual


# Multi-head self-attention module with relative positional encoding
class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (T, N, d_model)
            attn_mask: Attention mask with shape (T, T)
        Returns:
            Output tensor with shape (T, N, d_model)
        """
        residual = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.dropout(x)

        return x + residual


# Conformer block combining all modules
class ConformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            conv_kernel_size: int = 31,
            ff_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        # First feed-forward module (half-step)
        self.ff_module1 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            dropout=dropout
        )

        # Multi-head self-attention module
        self.mhsa_module = MultiHeadSelfAttentionModule(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Convolution module
        self.conv_module = ConformerConvModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout=dropout
        )

        # Second feed-forward module (half-step)
        self.ff_module2 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (T, N, d_model)
            attn_mask: Attention mask with shape (T, T)
        Returns:
            Output tensor with shape (T, N, d_model)
        """
        x = self.ff_module1(x)
        x = self.mhsa_module(x, attn_mask)
        x = self.conv_module(x)
        x = self.ff_module2(x)
        x = self.layer_norm(x)

        return x


class SubsampleConvModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm1 = nn.BatchNorm2d(channels * 2)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm2 = nn.BatchNorm2d(channels * 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, N, _, Freq, Bins = x.shape  # (T, N, 2, Freq, Bins)
        # Conv input (N, 2, T, -1)
        x = x.permute(1, 2, 0, 3, 4).reshape(N, 2, T, Freq * Bins)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        # reshape back to (T, N, 2, Freq, Bins)
        x = x.reshape(N, 2, -1, Freq, Bins).permute(2, 0, 1, 3, 4)
        return x

class TdsSubsampleConvModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = TDSConv2dBlock(channels, channels, kernel_size, stride=stride, padding=padding)
        self.conv2 = TDSConv2dBlock(channels, channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, N, _, Freq = x.shape  # (T, N, 2, Freq)
        x = x.reshape(T, N, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(N, 2, -1, Freq).permute(2, 0, 1, 3)
        return x
