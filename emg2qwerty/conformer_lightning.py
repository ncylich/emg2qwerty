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
from numpy.ma.core import zeros_like
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
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
            sos_token_id: int = None,  # Add SOS token ID parameter
            eos_token_id: int = None,
            optimizer: DictConfig = None,
            lr_scheduler: DictConfig = None,
            decoder: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.sos_token_id = sos_token_id if sos_token_id is not None else charset().sos_class
        self.eos_token_id = eos_token_id if eos_token_id is not None else charset().eos_class

        # Embedding for EMG data
        self.embedding = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            SubsampleConvModule(channels=2, kernel_size=3, stride=2, dropout=dropout),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            nn.Linear(mlp_features[-1] * self.NUM_BANDS, d_model),
        )

        # Special token embeddings (learned)
        self.sos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

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

        # CTC projection layer (for auxiliary CTC loss)
        self.ctc_proj = nn.Linear(d_model, charset().num_classes)

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

        # Output projection
        self.output_proj = nn.Linear(d_model, charset().num_classes)

        # Loss functions
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=charset().null_class)

        # Beam search decoder (currently disabled; using greedy search instead)
        # self.beam_decoder = instantiate(decoder)

        # Character error rate metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode input EMG data with the conformer encoder"""
        # Embed inputs
        x = self.embedding(inputs)  # (T, N, d_model)
        x = self.encoder_pos_encoding(x)

        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)

        return x

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
        encoder_outputs = self.encode(inputs)  # (T, N, d_model)

        # CTC outputs (auxiliary)
        encoder_logits = self.ctc_proj(encoder_outputs)  # (T, N, num_classes)
        ctc_log_probs = nn.functional.log_softmax(encoder_logits, dim=-1)

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
            "ctc_logits": encoder_logits,
            "ctc_log_probs": ctc_log_probs,
            "decoder_logits": decoder_logits,
            "decoder_log_probs": decoder_log_probs
        }

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
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
            loss = self.ce_weight * ce_loss

            self.log(f"{phase}/ce_loss", ce_loss, batch_size=N, sync_dist=True)
        else:
            # For validation (or test), skip cross-entropy loss computation due to variable output lengths
            loss = torch.tensor(0.0, device=inputs.device)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True, prog_bar=True)

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

            target = LabelData.from_labels(filtered_target)
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