# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, ClassVar, Optional

import math
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


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


class ConformerCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
            self,
            in_features: int,
            mlp_features: Sequence[int],
            d_model: int,
            nhead: int,
            num_layers: int,
            conv_kernel_size: int = 31,
            ff_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            dropout: float = 0.1,
            ctc_weight: float = 0.7,  # Weight for CTC loss
            ce_weight: float = 0.3,   # Weight for CE loss
            optimizer: DictConfig = None,
            lr_scheduler: DictConfig = None,
            decoder: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Store loss weights
        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight

        # Embedding for EMG data
        # Input shape: (T, N, bands=2, electrode_channels=16, freq)
        self.embedding = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # Flatten over the bands and channel dimensions
            nn.Flatten(start_dim=2),
            # Project to model dimension
            nn.Linear(mlp_features[-1] * self.NUM_BANDS, d_model),
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Stack of Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=nhead,
                conv_kernel_size=conv_kernel_size,
                ff_expansion_factor=ff_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Final classifier
        self.fc = nn.Linear(d_model, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Loss functions
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=charset().null_class)

        # Decoder from Hydra config
        self.decoder = instantiate(decoder)

        # Character error rate metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Conformer model.

        Args:
            inputs: Input tensor with shape (T, N, bands, electrode_channels, freq)

        Returns:
            Log probabilities with shape (T, N, num_classes)
        """
        # Embed inputs
        x = self.embedding(inputs)  # (T, N, d_model)
        x = self.positional_encoding(x)

        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)

        # Project to output classes and apply log softmax
        x = self.fc(x)
        x = self.log_softmax(x)

        return x

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)  # (T, N, C)

        # CTC loss
        ctc_loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (N, T_target)
            input_lengths=input_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        ce_loss = torch.zeros().to(emissions.device)
        # CE loss - fix the shape mismatch
        # First transpose to (N, T, C)
        emissions_t = emissions.transpose(0, 1)

        # # Create a mask for valid positions based on input lengths
        # mask = torch.arange(emissions.size(0)).unsqueeze(0).to(inputs.device) < input_lengths.unsqueeze(1)
        #
        # # Flatten only valid positions for both logits and targets
        # ce_logits = []
        # ce_targets = []
        #
        # # TODO: optimize loop & fix padding
        # for i in range(N):
        #     valid_len = min(input_lengths[i], targets.size(0))
        #     ce_logits.append(emissions_t[i, :valid_len])
        #     ce_targets.append(targets[:valid_len, i])
        #
        # ce_logits = torch.cat(ce_logits, dim=0)  # Shape: (sum(valid_lengths), C)
        # ce_targets = torch.cat(ce_targets, dim=0)  # Shape: (sum(valid_lengths))
        #
        # ce_loss = self.ce_loss(ce_logits, ce_targets)

        # Combined loss
        loss = self.ctc_weight * ctc_loss + self.ce_weight * ce_loss

        # Log individual losses
        self.log(f"{phase}/ctc_loss", ctc_loss, batch_size=N, sync_dist=True)
        self.log(f"{phase}/ce_loss", ce_loss, batch_size=N, sync_dist=True)
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True, prog_bar=True)

        # Decode emissions for metrics
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=input_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/CER", metrics.compute()[f"{phase}/CER"], batch_size=N, sync_dist=True, prog_bar=True)
        return loss

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), logger=True, sync_dist=True)
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
