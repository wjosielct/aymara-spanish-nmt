# ====================
# ./train_tokenizer.py
# ====================

from itertools import chain
from pathlib import Path
import pandas as pd
from src.tokenization.tokenizer import train_sentencepiece_tokenizer
from src.utils.config import load_config


if __name__ == "__main__":

    cfg = load_config("config.yaml")

    splits_dir = Path(cfg["splits_dir"])

    # Load only bible train dataset to build vocabulary
    train_tsv_path = splits_dir / "train_bible.tsv"
    train_df = pd.read_csv(train_tsv_path, sep="\t", dtype=str)
    print(f"[Loaded] training data from: {train_tsv_path} (pairs: {len(train_df)})")

    # Train SentencePiece tokenizer
    training_sentences = chain(train_df["aym_text"].astype(str), train_df["spa_text"].astype(str))

    print(f"[Training] SentencePiece tokenizer...")
    train_sentencepiece_tokenizer(
        sentences=training_sentences,
        vocab_size=cfg["vocab_size"],
        model_type=cfg["model_type"],
        character_coverage=cfg["character_coverage"]
    )

