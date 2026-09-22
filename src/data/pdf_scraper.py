# =========================
# ./src/data/pdf_scraper.py
# =========================

import re
from pathlib import Path
import pdfplumber
import pandas as pd


def scrap_pdf(
        pdf_path: str | Path, 
        start_page: int = 13, 
        end_page: int = 146
) -> pd.DataFrame:

    """
    Extracts an Aymara-Spanish parallel dialogue corpus from a PDF.

    Parses conversation turns matching the 'SPEAKER) Aymara / Spanish' format across the specified 
    page range. Handles multi-page line wrapping, filters out structural metadata (headers, 
    page numbers, author credits), and standardizes whitespace.

    Args:
        pdf_path: Path to the source PDF document.
        start_page: 1-based index of the starting page. Defaults to 13.
        end_page: 1-based index of the ending page. Defaults to 146.

    Returns:
        pd.DataFrame: DataFrame containing ['aymara', 'spanish'] parallel text columns.
    """

    print("\n▪ Starting PDF scraping...")

    raw_lines = []

    # Extract text line by line across the specified page range
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(start_page - 1, end_page):
            page = pdf.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                raw_lines.extend(page_text.split("\n"))

    # Filter out static headers and standalone page numbers
    cleaned_lines = []
    for line in raw_lines:
        trimmed_line = line.strip()

        if not trimmed_line:
            continue
        # Skip static page headers
        if trimmed_line in ["Conversaciones en aimara", "Aymara Aruskipawinaka"]:
            continue
        # Skip standalone page numbers
        if re.match(r"^\d+$", trimmed_line):
            continue

        cleaned_lines.append(trimmed_line)

    # Accumulate blocks strictly starting with speaker pattern "LETTER) "
    regex_speaker = re.compile(r"^[A-Z]\)\s+")
    speaker_blocks = []
    current_block = []

    for line in cleaned_lines:
        # Check if line is metadata (chapter title, author section, or divider)
        is_metadata = (
            bool(re.search(r"\bARUSKIPAWI\b", line, re.IGNORECASE))
            or line.startswith("Qilqirinaka:")
            or bool(re.match(r"^-+&*-+$", line))
        )

        # If metadata is reached, finalize current speaker block and reset
        if is_metadata:
            if current_block:
                speaker_blocks.append(" ".join(current_block))
                current_block = []
            continue

        has_speaker = bool(regex_speaker.match(line))
        has_slash = " / " in line or " /" in line or "/ " in line

        if has_speaker:
            # Save the previous block and start a new speaker block
            if current_block:
                speaker_blocks.append(" ".join(current_block))
            current_block = [line]

        elif current_block:
            current_has_slash = any("/" in l for l in current_block)

            # If current block already has '/' and line has '/' without speaker letter,
            # it is a standalone narrative line. Close current block and discard line.
            if current_has_slash and has_slash:
                speaker_blocks.append(" ".join(current_block))
                current_block = []
            else:
                # Append text wrapping / continuation across lines or pages
                current_block.append(line)

    # Append the remaining block if exists
    if current_block:
        speaker_blocks.append(" ".join(current_block))

    # Parse accumulated speaker blocks into structured dataset
    regex_parser = re.compile(r"^([A-Z])\)\s*(.*?)\s*/\s*(.*)$")
    book_corpus = []

    for block in speaker_blocks:
        if "/" not in block:
            continue

        match = regex_parser.match(block)
        if match:
            aym_text = re.sub(r"\s+", " ", match.group(2)).strip()
            spa_text = re.sub(r"\s+", " ", match.group(3)).strip()

            if aym_text and spa_text:
                book_corpus.append({"souce": "book", "aym_text": aym_text, "spa_text": spa_text})


    print("¡Process Completed!")
    print(f"Total number of raw pairs: {len(book_corpus):,}")

    return pd.DataFrame(book_corpus)

