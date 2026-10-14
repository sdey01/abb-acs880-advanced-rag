"""Central configuration for the ABB ACS880 manual assistant."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
PDF_PATH = Path(os.getenv("ACS880_PDF_PATH", str(ROOT / "EN_ACS880_Primary_FW_manual_V_A4.pdf")))
if not PDF_PATH.is_absolute():
    PDF_PATH = (ROOT / PDF_PATH).resolve()
CHROMA_PATH = Path(os.getenv("ACS880_CHROMA_PATH", str(ROOT / "chroma_db")))
if not CHROMA_PATH.is_absolute():
    CHROMA_PATH = (ROOT / CHROMA_PATH).resolve()
COLLECTION_NAME = "abb_acs880_firmware_manual"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
CHUNK_SIZE, CHUNK_OVERLAP = 1024, 32
FACTUAL_CANDIDATES, BROADER_CANDIDATES = 5, 20
EXPANSION_COUNT, FINAL_CONTEXT_LIMIT = 3, 8
CACHE_TTL_SECONDS, CACHE_MAX_ENTRIES, CACHE_THRESHOLD = 3600, 200, 0.90
