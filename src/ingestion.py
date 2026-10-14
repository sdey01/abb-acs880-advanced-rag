"""PDF loading, source metadata normalization and chunking."""
from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(f"{CHUNK_SIZE}:{CHUNK_OVERLAP}".encode())
    return digest.hexdigest()


def load_chunks(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Manual PDF not found: {path}")
    pages = PyPDFLoader(str(path)).load()
    if not pages or not any(doc.page_content.strip() for doc in pages):
        raise ValueError("The PDF has no extractable text.")
    for doc in pages:
        doc.metadata["source"] = path.name
        raw_page = doc.metadata.get("page")
        doc.metadata["page_number"] = int(raw_page) + 1 if raw_page is not None else None
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    ).split_documents(pages)
    if not chunks:
        raise ValueError("No text chunks could be created from the PDF.")
    return chunks
