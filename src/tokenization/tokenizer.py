# ./src/tokenization/tokenizer.py

from pathlib import Path
from typing import Iterable, Iterator
import sentencepiece as spm
import pandas as pd


def train_sentencepiece_tokenizer(
        sentences: Iterable[str] | Iterator[str],
        vocab_size: int = 8000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        tokenizer_dir: str | Path = "tokenizer"
) -> None:

    """
    Trains a SentencePiece subword tokenizer directly from an iterable of strings.

    Special Token IDs:
        0: <pad>
        1: <s>   (bos)
        2: </s>  (eos)
        3: <unk> (unknown)
    """

    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    sp_prefix = tokenizer_dir / "SentencePiece"

    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences,
        model_prefix=str(sp_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        shuffle_input_sentence=True,
        normalization_rule_name="nmt_nfkc",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3
    )

    print(f"☑ Done! tokenizer saved to --> {tokenizer_dir}")


def encode_dataframe(
        df: pd.DataFrame,
        sp_model: spm.SentencePieceProcessor,
        source_col: str = "aym_text",
        target_col: str = "spa_text"
) -> pd.DataFrame:

    """
    Tokenizes parallel text columns in a DataFrame using a SentencePiece model.

    Args:
        df: Input DataFrame containing parallel text columns.
        sp_model: SentencePiece model.
        source_col: Name of the source text column.
        target_col: Name of the target text column.

    Returns:
        pd.DataFrame: A copy of the DataFrame with added 'aym_ids' and 'spa_ids' columns.
    """

    bos_id = sp_model.bos_id()
    eos_id = sp_model.eos_id()

    def _encode_sequence(text: str) -> str:

        if not text.strip():
            ids = []
        else:
            ids = sp_model.encode(text.strip(), out_type=int)

        ids = [bos_id] + ids + [eos_id]

        return " ".join(map(str, ids))

    df_encoded = df.copy()

    src_target_name = f"{source_col.split('_')[0]}_ids"
    tgt_target_name = f"{target_col.split('_')[0]}_ids"

    print(f" --> Encoding column: '{source_col}'...")
    df_encoded[src_target_name] = df_encoded[source_col].map(_encode_sequence)

    print(f" --> Encoding column: '{target_col}'...")
    df_encoded[tgt_target_name] = df_encoded[target_col].map(_encode_sequence)

    return df_encoded