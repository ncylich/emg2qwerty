# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Optional

import math
import numpy as np
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from numpy.ma.core import zeros_like
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection, MeanMetric

from emg2qwerty import utils
from emg2qwerty.ce_charset import charset
from emg2qwerty.ce_data import LabelData, WindowedEMGDataset
from emg2qwerty.custom_modules import SubsampleConvModule, PositionalEncoding, ConformerBlock, Swish, DctLogSpectrogram, SpecAugment
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    PermuteReshape,
    TDSConv2dBlock,
)
from emg2qwerty.transforms import Transform, LogFreqBinsSpectrogram


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Self-attention with causal masking
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

        # Cross-attention to encoder outputs
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence (T_tgt, N, d_model)
            memory: Encoder outputs (T_src, N, d_model)
            tgt_mask: Attention mask for self-attention (T_tgt, T_tgt)
            tgt_key_padding_mask: Key padding mask for target (N, T_tgt)
            memory_key_padding_mask: Key padding mask for memory (N, T_src)
        """
        # Self-attention block
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            query=tgt2,
            key=tgt2,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)

        # Cross-attention block
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.cross_attn(
            query=tgt2,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)

        # Feed-forward block
        tgt2 = self.norm3(tgt)
        tgt2 = self.feed_forward(tgt2)
        tgt = tgt + tgt2

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence (T_tgt, N, d_model)
            memory: Encoder outputs (T_src, N, d_model)
            tgt_mask: Self-attention mask (T_tgt, T_tgt)
            tgt_key_padding_mask: Target padding mask (N, T_tgt)
            memory_key_padding_mask: Memory padding mask (N, T_src)
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return self.norm(output)


class ConformerDecoder(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
            self,
            in_features: int,
            mlp_features: Sequence[int],
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            conv_kernel_size: int = 31,
            ff_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            dropout: float = 0.1,
            ctc_loss_weight: float = 0.1,
            l1_loss_weight: float = 0.0,
            tds_conv_encoder_block_channels: Sequence[int] = (16, 16),
            tds_conv_encoder_kernel_width: int = 15,
            # hop_length: int: = 16,
            use_dct: bool = False,
            sos_token_id: int = None,  # Add SOS token ID parameter
            eos_token_id: int = None,
            optimizer: DictConfig = None,
            decoder: DictConfig = None,
            lr_scheduler: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.sos_token_id = sos_token_id if sos_token_id is not None else charset().sos_class
        self.eos_token_id = eos_token_id if eos_token_id is not None else charset().eos_class

        self.use_dct = use_dct
        if use_dct:
            self.spectrogram = DctLogSpectrogram(n_dct=64, hop_length=16)
            self.spec_augment = SpecAugment(n_time_masks=3, time_mask_param=25, n_freq_masks=2, freq_mask_param=4)

        # Embedding for EMG data
        self.embedding = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),  # (T, N, 2, 16, 6)
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            # TDSConv2dBlock(channels=self.NUM_BANDS, width=mlp_features[-1], kernel_width=3, stride=2, padding=1, dropout=dropout),
            TDSConvEncoder(num_features=mlp_features[-1] * self.NUM_BANDS, dropout=dropout,
                           block_channels=tds_conv_encoder_block_channels, kernel_width=tds_conv_encoder_kernel_width),
        )

        self.embedding_to_encoder = nn.Linear(mlp_features[-1] * self.NUM_BANDS, d_model)
        self.embedding_to_ctc = nn.Linear(mlp_features[-1] * self.NUM_BANDS, charset().num_classes)

        # Special token embeddings (learned)
        self.sos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        # # Band embedding
        # self.band_embedding = nn.Embedding(self.NUM_BANDS, d_model)

        # Initialize special embeddings with Xavier/Glorot
        nn.init.xavier_normal_(self.sos_embedding)

        # Positional encoding for encoder
        self.encoder_pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Conformer encoder blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=nhead,
                conv_kernel_size=conv_kernel_size,
                ff_expansion_factor=ff_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                dropout=dropout
            ) for _ in range(num_encoder_layers)
        ])

        # Decoder embedding and positional encoding
        self.decoder_embedding = nn.Embedding(charset().num_classes, d_model)
        self.decoder_pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Transformer decoder
        self.transformer_decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=d_model * ff_expansion_factor,
            dropout=dropout
        )

        self.output_proj = nn.Linear(d_model, charset().num_classes)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=charset().null_class)
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.ce_weight = 1.0 - ctc_loss_weight
        self.ctc_weight = ctc_loss_weight

        self.l1_loss_weight = self.hparams.l1_loss_weight

        # Beam search decoder
        self.beam_decoder = instantiate(decoder)

        # Character error rate metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )
        self.loss_metrics = nn.ModuleDict(
            {f"{loss}_loss": MeanMetric() for loss in ["ce", "ctc", "l1"]}
        )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask where upper triangle is True (masked out, not including diagonal)"""
        return torch.tril(torch.ones(sz, sz)) == 0

    def l1_loss(self):
        if self.l1_loss_weight == 0.0:
            return 0.0
        l1_loss = sum(torch.norm(param, p=1) for param in self.parameters() if param.requires_grad)
        return self.l1_loss_weight * l1_loss

    def log_loss_metrics(self, ce_loss, ctc_loss, l1_loss):
        self.loss_metrics["ce_loss"].update(ce_loss)
        self.log(f"ce_loss", self.loss_metrics["ce_loss"].compute(), sync_dist=True, prog_bar=True)
        self.loss_metrics["ctc_loss"].update(ctc_loss)
        self.log(f"ctc_loss", self.loss_metrics["ctc_loss"].compute(), sync_dist=True, prog_bar=True)
        if self.l1_loss_weight > 0:
            self.loss_metrics["l1_loss"].update(l1_loss)
            self.log(f"l1_loss", self.loss_metrics["l1_loss"].compute(), sync_dist=True, prog_bar=True)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor]:
        """Encode input EMG data with the conformer encoder"""
        # Prepare Inputs (if using DCT)
        if self.use_dct:
            inputs = self.spectrogram(inputs)
            if self.training:
                inputs = self.spec_augment(inputs)

        # Embed Inputs
        x = self.embedding(inputs)  # (T, N, 2, d_model)
        ctc_logits = self.embedding_to_ctc(x)  # (T, N, num_classes)
        x = self.embedding_to_encoder(x)
        x = self.encoder_pos_encoding(x)
        # band1 = self.encoder_pos_encoding(x[:, :, 0, :])  # (T, N, d_model)
        # band2 = self.encoder_pos_encoding(x[:, :, 1, :])  # (T, N, d_model)
        # band1 = band1 + self.band_embedding(torch.zeros(1, 1, dtype=torch.long, device=inputs.device))
        # band2 = band2 + self.band_embedding(torch.ones(1, 1, dtype=torch.long, device=inputs.device))
        # x = torch.cat([band1, band2], dim=0)  # (T * 2, N, d_model)

        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)

        return x, ctc_logits

    def forward(
            self,
            inputs: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            target_lengths: Optional[torch.Tensor] = None,
            teacher_forcing_ratio: float = 1.0
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the encoder-decoder model.

        Args:
            inputs: Input tensor with shape (T, N, bands, electrode_channels, freq)
            targets: Target tensor with shape (T_target, N) (optional, for training)
            target_lengths: Lengths of targets (optional, for training)
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            Dictionary with encoder and decoder outputs
        """
        device = inputs.device
        encoder_outputs, ctc_logits = self.encode(inputs)  # (T, N, d_model)

        # Handle decoder outputs based on training or inference
        if self.training and targets is not None:
            # Add EOS tokens to targets for training
            batch_size = encoder_outputs.size(1)

            # Start with the learned SOS embedding
            sos_tokens = self.sos_embedding.expand(-1, batch_size, -1)

            # Convert target tokens to embeddings
            if targets.size(0) > 1:
                target_emb = self.decoder_embedding(targets[:-1])
                decoder_inputs = torch.cat([sos_tokens, target_emb], dim=0)
            else:
                decoder_inputs = sos_tokens

            # Add positional encoding
            decoder_inputs = self.decoder_pos_encoding(decoder_inputs)

            # Create causal mask
            tgt_mask = self._generate_square_subsequent_mask(decoder_inputs.size(0)).to(device)

            # Create key padding mask using target_lengths
            # The decoder_inputs has shape (T_tgt, N) where T_tgt = targets[:-1] concatenated with SOS, so total length equals targets.size(0)
            T_tgt = decoder_inputs.size(0)
            tgt_key_padding_mask = torch.arange(T_tgt, device=device).unsqueeze(0).expand(batch_size, T_tgt) > target_lengths.unsqueeze(1)

            # Run decoder with both causal mask and key padding mask
            decoder_outputs = self.transformer_decoder(
                tgt=decoder_inputs,
                memory=encoder_outputs,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            # Project to vocabulary
            decoder_logits = self.output_proj(decoder_outputs)  # (T_tgt, N, num_classes)
            decoder_log_probs = nn.functional.log_softmax(decoder_logits, dim=-1)

        else:
            # Inference mode
            batch_size = encoder_outputs.size(1)
            max_len = 50

            # Start with the learned SOS embedding
            decoder_input = torch.full((1, batch_size), self.sos_token_id,
                                       dtype=torch.long, device=device)
            decoder_outputs = []

            for i in range(max_len):
                # For SOS token, use special embedding
                if i == 0:
                    emb = self.sos_embedding.expand(-1, batch_size, -1)
                else:
                    # For normal tokens, use standard embedding
                    emb = self.decoder_embedding(decoder_input)

                emb = self.decoder_pos_encoding(emb)

                # Create causal mask
                tgt_mask = self._generate_square_subsequent_mask(emb.size(0)).to(device)

                # Decode one step
                decoder_output = self.transformer_decoder(
                    tgt=emb,
                    memory=encoder_outputs,
                    tgt_mask=tgt_mask
                )

                # Get prediction for last position
                last_output = decoder_output[-1:]  # (1, N, d_model)
                logits = self.output_proj(last_output)  # (1, N, num_classes)

                # Store output
                decoder_outputs.append(logits)

                # Sample next token (greedy)
                next_token = torch.argmax(logits, dim=-1)  # (1, N)

                # Append to input for next iteration
                decoder_input = torch.cat([decoder_input, next_token], dim=0)

                # Early stopping if all sequences predicted EOS
                if ((next_token == self.eos_token_id).all()).item():
                    break

            # Combine all decoder outputs
            decoder_logits = torch.cat(decoder_outputs, dim=0)  # (T_out, N, num_classes)
            decoder_log_probs = nn.functional.log_softmax(decoder_logits, dim=-1)

        return {
            "ctc_logits": ctc_logits,
            "decoder_logits": decoder_logits,
            "decoder_log_probs": decoder_log_probs
        }

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        input_lengths = batch["input_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        N = len(target_lengths)

        # Forward pass (for non-training, inference branch is used automatically)
        outputs = self.forward(inputs, targets, target_lengths)

        if phase == "train":
            # Compute cross-entropy loss (teacher-forcing mode)
            # decoder_logits shape: (T, N, C) -> transpose to (N, T, C)
            decoder_logits = outputs["decoder_logits"].transpose(0, 1)  # (N, T, C)
            targets_t = targets.transpose(0, 1)  # (N, T)

            # Use decoder outputs from index 1 onward to predict the next token
            ce_logits = decoder_logits[:, 1:, :]  # (N, T-1, C)
            ce_targets = targets_t[:, 1:]  # (N, T-1)

            # Flatten for cross-entropy
            ce_logits = ce_logits.reshape(-1, charset().num_classes)
            ce_targets = ce_targets.reshape(-1)

            ce_loss = self.ce_loss(ce_logits, ce_targets)

            ctc_loss = outputs["ctc_logits"]
            emissions = F.log_softmax(ctc_loss, dim=-1)
            T_diff = inputs.shape[0] - emissions.shape[0]
            emission_lengths = input_lengths - T_diff
            ctc_loss = self.ctc_loss(
                log_probs=emissions,  # (T, N, num_classes)
                targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
                input_lengths=emission_lengths,  # (N,)
                target_lengths=target_lengths,  # (N,)
            )

            l1_loss = self.l1_loss()

            self.log_loss_metrics(ce_loss=ce_loss, ctc_loss=ctc_loss, l1_loss=l1_loss)
            loss = ce_loss + ctc_loss + l1_loss
        else:
            # For validation (or test), skip cross-entropy loss computation due to variable output lengths
            loss = torch.tensor(0.0, device=inputs.device)

        # Greedy decoding for metrics
        # Obtain predicted tokens from decoder_log_probs (shape: T x N)
        greedy_preds = torch.argmax(outputs["decoder_log_probs"], dim=-1)  # (T, N)
        greedy_preds = greedy_preds.detach().cpu().numpy()

        # For each sample, trim the prediction at the first occurrence of the EOS token
        predictions = []
        eos_token = self.eos_token_id
        T, N = greedy_preds.shape
        for i in range(N):
            pred_seq = greedy_preds[:, i].tolist()
            if eos_token in pred_seq:
                eos_index = pred_seq.index(eos_token)
                pred_seq = pred_seq[:eos_index + 1]
            predictions.append(pred_seq)

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Extract the raw target sequence as a list
            raw_target = targets_np[:target_lengths_np[i], i].tolist()

            # Filter out the special tokens: null, SOS, and EOS
            filtered_target = [t for t in raw_target if
                               t not in [charset().null_class, self.sos_token_id, self.eos_token_id]]
            filtered_prediction = [t for t in predictions[i] if t
                                   not in [charset().null_class, self.sos_token_id, self.eos_token_id]]

            target = LabelData.from_labels(filtered_target)
            prediction = LabelData.from_labels(filtered_prediction)
            # print(f"Target: {target.text}\tPrediction: {prediction.text}")
            metrics.update(prediction=prediction, target=target)

        self.log(f"{phase}/CER", metrics.compute()[f"{phase}/CER"], sync_dist=True, prog_bar=True)
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
        for metric in self.loss_metrics.values():
            metric.reset()

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