"""
Ingestion: load raw documents from disk and split them into retrievable chunks.

Implements (per Final Project Brief, Section 2):
  Req 1 - Document ingestion: loads .txt (and .pdf if pypdf is installed)
  Req 2 - Chunking: sentence-aware chunking with word-budget + overlap
          (defensible strategy: chunks never cut a sentence in half, so every
          retrieved chunk is readable and citable on its own)

Each chunk carries metadata (source filename, title, chunk index) so the
generation layer can cite exactly where an answer came from.
"""

import os
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    source_file: str
    text: str


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_pdf(path: str) -> str:
    """PDF support is optional; requires `pip install pypdf`."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def load_documents(folder: str) -> List[dict]:
    """Load every .txt/.pdf file in `folder` into {"title", "text", "file"} dicts."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Document folder not found: {folder}")

    docs = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".txt":
            text = _load_txt(path)
        elif ext == ".pdf":
            text = _load_pdf(path)
        else:
            continue
        if not text:
            continue
        title = os.path.splitext(filename)[0].replace("_", " ").title()
        docs.append({"title": title, "text": text, "file": filename})
    return docs


# ---------------------------------------------------------------------------
# Chunking (sentence-aware)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no external NLP dependency)."""
    # Normalise whitespace first so paragraph breaks don't create empty items.
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENTENCE_SPLIT.split(flat) if s.strip()]


def chunk_text(text: str, chunk_size: int = 120, overlap_sentences: int = 1) -> List[str]:
    """
    Sentence-aware chunking: greedily pack whole sentences into a chunk until
    the word budget (`chunk_size`) is reached, then start the next chunk,
    carrying over the last `overlap_sentences` sentences for context continuity.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        n_words = len(sentence.split())
        if current and current_words + n_words > chunk_size:
            chunks.append(" ".join(current))
            # overlap: keep the tail sentences as the start of the next chunk
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_words = sum(len(s.split()) for s in current)
        current.append(sentence)
        current_words += n_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunk_records(docs: List[dict], chunk_size: int = 120,
                        overlap_sentences: int = 1) -> List[Chunk]:
    """Turn loaded documents into a flat list of Chunk records ready for embedding."""
    records: List[Chunk] = []
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size,
                            overlap_sentences=overlap_sentences)
        for i, piece in enumerate(pieces):
            records.append(Chunk(
                chunk_id=f"{doc['file']}::{i}",
                doc_title=doc["title"],
                source_file=doc["file"],
                text=piece,
            ))
    return records
