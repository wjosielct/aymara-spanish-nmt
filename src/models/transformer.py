# ===========================
# ./src/models/transformer.py
# ===========================

import math
import torch
import torch.nn as nn
from typing import Tuple


class SinusoidalPositionalEncoding(nn.Module):

    """
    Implements standard sinusoidal positional encoding of Vaswani et al. (2017).
    """

    def __init__(self, d_model: int, max_len: int = 1000):

        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.pe[:, :x.size(1), :]


class TranslationTransformer(nn.Module):

    """
    Sequence-to-Sequence Transformer architecture with weight tying and sinusoidal embeddings.
    """
    
    def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            n_heads: int = 8,
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 4,
            dim_feedforward: int = 1024,
            dropout: float = 0.2,
            pad_id: int = 0
    ):
        
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.output_layer.weight = self.embedding.weight
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:

        return (src == self.pad_id)

    def make_tgt_mask(self, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        tgt_pad_mask = (tgt == self.pad_id)
        seq_len = tgt.size(1)

        tgt_causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device), diagonal=1)

        return tgt_pad_mask, tgt_causal_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:

        src_pad_mask = self.make_src_mask(src)
        tgt_pad_mask, tgt_causal_mask = self.make_tgt_mask(tgt)

        src_emb = self.dropout(self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model)))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_causal_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        
        return self.output_layer(out)

