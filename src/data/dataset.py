# ./src/data/dataset.py

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ParallelTranslationDataset(Dataset):

    """
    PyTorch Dataset for parallel NMT tokenized TSV files.
    Parses pre-tokenized whitespace-separated token ID strings.
    """

    def __init__(
            self,
            tsv_path: str | Path,
            src_col: str = "aym_ids",
            tgt_col: str = "spa_ids"
    ):
        
        self.df = pd.read_csv(tsv_path, sep="\t", dtype=str)
        
        self.src_ids = [[int(token) for token in str(text).split()] for text in self.df[src_col]]
        self.tgt_ids = [[int(token) for token in str(text).split()] for text in self.df[tgt_col]]

        assert len(self.src_ids) == len(self.tgt_ids), "Source and Target lengths must match."

    def __len__(self) -> int:

        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        return (torch.tensor(self.src_ids[idx], dtype=torch.long), torch.tensor(self.tgt_ids[idx], dtype=torch.long))


def collate_fn_pad(
        batch: List[Tuple[torch.Tensor, torch.Tensor]], 
        pad_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Dynamically pads batches to the maximum sequence length in the current batch.
    """

    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    return src_padded, tgt_padded