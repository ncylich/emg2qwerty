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
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.ce_charset import charset
from emg2qwerty.ce_data import LabelData, WindowedEMGDataset
from emg2qwerty.conformer_modules import PositionalEncoding
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform, LogFreqBinsSpectrogram

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequence (T_src, N, d_model)
            src_mask: Attention mask (T_src, T_src)
            src_key_padding_mask: Key padding mask (N, T_src)
        """
        # Self-attention block
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            query=src2,
            key=src2,
            value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout(src2)

        # Feed-forward block
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + src2

        return src


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequence (T_src, N, d_model)
            src_mask: Attention mask (T_src, T_src)
            src_key_padding_mask: Key padding mask (N, T_src)
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )

        return self.norm(output)


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
            nn.ReLU(),  # Using ReLU instead of Swish for simplicity
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


class TransformerEncoderDecoder(pl.LightningModule):
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
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            sos_token_id: int = None,
            eos_token_id: int = None,
            optimizer: DictConfig = None,
            decoder: DictConfig = None,
            lr_scheduler: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.sos_token_id = sos_token_id if sos_token_id is not None else charset().sos_class
        self.eos_token_id = eos_token_id if eos_token_id is not None else charset().eos_class

        # Embedding for EMG data
        self.embedding = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            nn.Linear(mlp_features[-1] * self.NUM_BANDS, d_model),
        )

        # Special token embeddings
        self.sos_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.xavier_normal_(self.sos_embedding)

        # Positional encoding for encoder and decoder
        self.encoder_pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.decoder_pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # CTC projection layer (auxiliary)
        self.ctc_proj = nn.Linear(d_model, charset().num_classes)

        # Decoder embedding
        self.decoder_embedding = nn.Embedding(charset().num_classes, d_model)

        # Transformer decoder
        self.transformer_decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, charset().num_classes)

        # Loss functions
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=charset().null_class)

        # Character error rate metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask where upper triangle is True (masked out, not including diagonal)"""
        return torch.tril(torch.ones(sz, sz)) == 0

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode input EMG data with the transformer encoder"""
        # Embed inputs
        x = self.embedding(inputs)  # (T, N, d_model)
        x = self.encoder_pos_encoding(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

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

        # Forward pass
        outputs = self.forward(inputs, targets, target_lengths)

        if phase == "train":
            # Compute cross-entropy loss (teacher-forcing mode)
            decoder_logits = outputs["decoder_logits"].transpose(0, 1)  # (N, T, C)
            targets_t = targets.transpose(0, 1)  # (N, T)

            # Use decoder outputs from index 1 onward to predict the next token
            ce_logits = decoder_logits[:, 1:, :]  # (N, T-1, C)
            ce_targets = targets_t[:, 1:]  # (N, T-1)

            # Flatten for cross-entropy
            ce_logits = ce_logits.reshape(-1, charset().num_classes)
            ce_targets = ce_targets.reshape(-1)

            loss = self.ce_loss(ce_logits, ce_targets)
        else:
            # For validation/test
            loss = torch.tensor(0.0, device=inputs.device)

        if phase == "train":
            self.log(f"train/loss", loss, batch_size=N, sync_dist=True, prog_bar=True)

        # Greedy decoding for metrics
        greedy_preds = torch.argmax(outputs["decoder_log_probs"], dim=-1)  # (T, N)
        greedy_preds = greedy_preds.detach().cpu().numpy()

        # Process predictions
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
            raw_target = targets_np[:target_lengths_np[i], i].tolist()

            # Filter out special tokens
            filtered_target = [t for t in raw_target if
                              t not in [charset().null_class, self.sos_token_id, self.eos_token_id]]
            filtered_prediction = [t for t in predictions[i] if t
                                  not in [charset().null_class, self.sos_token_id, self.eos_token_id]]

            target = LabelData.from_labels(filtered_target)
            prediction = LabelData.from_labels(filtered_prediction)
            metrics.update(prediction=prediction, target=target)

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