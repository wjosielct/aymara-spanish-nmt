# =====================
# ./preprocess_bible.py
# =====================

from pathlib import Path
import pandas as pd
from src.utils.config import load_config
from src.data.preprocessor import preprocess_text, remove_empty_pairs, remove_duplicate_pairs, filter_by_length_and_ratio


if __name__ == "__main__":

    cfg = load_config("config.yaml")

    input_path = Path(cfg["interim_data_dir"]) / "verses_aligned_raw.tsv"
    output_dir = Path(cfg["clean_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "verses_aligned_clean.tsv"

    df = pd.read_csv(input_path, sep="\t", dtype=str)
    print(f"▪ Total raw pairs: {len(df):,}")

    print(f"-> Cleaning column: 'aym_text'...")
    df["aym_text"] = df["aym_text"].astype(str).map(lambda text: preprocess_text(text))

    print(f"-> Cleaning column: 'spa_text'...")
    df["spa_text"] = df["spa_text"].astype(str).map(lambda text: preprocess_text(text))

    # Remove empty pairs
    df = remove_empty_pairs(df=df, source_col="aym_text", target_col="spa_text")

    # Remove duplicate pairs
    df = remove_duplicate_pairs(df=df, source_col="aym_text", target_col="spa_text")

    # Filter extreme lengths and abnormal ratios
    df = filter_by_length_and_ratio(
        df=df,
        source_col="aym_text",
        target_col="spa_text",
        min_words=cfg["min_words_bible"],
        max_words=cfg["max_words"],
        max_ratio=cfg["max_ratio"]
    )

    print(f"▪ Total clean pairs: {len(df):,}")

    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

    print(f"[Saved] Clean Aligned Verses exported to --> {output_path}")

