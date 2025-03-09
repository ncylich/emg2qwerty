from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
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
import math
from emg2qwerty.transforms import Transform
from emg2qwerty.custom_modules import SubsampleConvModule, PositionalEncoding, ConformerBlock


class ConformerCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
            self,
            in_features: int,
            sub_sample_conv: bool,
            mlp_features: Sequence[int],
            d_model: int,
            nhead: int,
            num_layers: int,
            conv_kernel_size: int = 31,
            ff_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            dropout: float = 0.1,
            optimizer: DictConfig = None,
            lr_scheduler: DictConfig = None,
            decoder: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Embedding for EMG data
        # Input shape: (T, N, bands=2, electrode_channels=16, freq)
        embeddings = [
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),  # (T, N, 2, 16, 6)
            SubsampleConvModule(channels=2, kernel_size=3, dropout=dropout, stride=1) if sub_sample_conv else None,
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),  # (T, N, 2, mlp_features)
            # Flatten over the bands and channel dimensions
            nn.Flatten(start_dim=2),
            # Project to model dimension
            nn.Linear(mlp_features[-1] * self.NUM_BANDS, d_model),
        ]
        embeddings = [layer for layer in embeddings if layer]
        self.embedding = nn.Sequential(*embeddings)

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
        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (N, T_target)
            input_lengths=input_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

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
