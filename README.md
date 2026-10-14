# ABB ACS880 Firmware Assistant

A local Streamlit RAG proof of concept for asking questions about the supplied ABB ACS880 firmware manual. Answers use only retrieved PDF excerpts and include source/page citations. The manual and qualified engineering review remain the source of truth.

## Requirements and setup

- Windows, Python 3.11, and the already-created `virtual_environment` in this project directory.
- An OpenAI API key with access to `gpt-5.4-mini`.
- The supplied `EN_ACS880_Primary_FW_manual_V_A4.pdf` (or configure `ACS880_PDF_PATH`).

The virtual environment in this workspace is `virtual_environment`; the verified interpreter is Python 3.11.9. The installed Streamlit, LangChain, ChromaDB, OpenAI integration, dotenv, and CPU PyTorch packages were reused. Only missing packages should be installed:

```powershell
.\virtual_environment\Scripts\python.exe -m pip install -r requirements.txt
```

CPU PyTorch was installed from the official CPU wheel index in the existing venv. If setting up a fresh environment and pip selects a GPU build, install CPU PyTorch with:

```powershell
.\virtual_environment\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Add `OPENAI_API_KEY=...` to the existing `.env`. Do not replace or share that file. `.env.example` contains placeholders only. The application never displays the key. `OPENAI_MODEL` may optionally override the default `gpt-5.4-mini`; it does not silently fall back to another model.

Place the manual in the project root under its current filename or configure `ACS880_PDF_PATH` in `.env`. Do not commit licensed/restricted manuals.

## Run

```powershell
.\virtual_environment\Scripts\python.exe run_app.py
```

The launcher initializes the PyTorch CPU runtime before Streamlit. This ordering is required in this Windows environment because importing Streamlit first can prevent `torch`'s native `c10.dll` from initializing. First startup loads the CPU embedding model and indexes the PDF. The embedding model is `sentence-transformers/all-MiniLM-L6-v2`; the CPU cross-encoder reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Models are downloaded/cached through Hugging Face's normal cache when not present. The persisted Chroma index is stored in `chroma_db/`, reused when the PDF fingerprint and embedding model match, and rebuilt using **Re-index manual**.

## Behavior and configuration

- Classification routes narrow technical lookups to `FACTUAL` (5 initial candidates) and explanation, multi-part, comparison, relationship, or procedure questions to `BROADER` (20).
- A cache check happens before classification. Semantic cache similarity defaults to a conservative cosine threshold of `0.90`; technical identifiers, numeric tokens, and recognized units must match exactly. The answer cache is bounded to 200 entries with an individual one-hour TTL. These values live in `config.py`.
- Query expansion asks for three alternative searches and preserves the original question. If expansion fails, retrieval continues with the original.
- Reranking is an independent CPU cross-encoder stage. On reranker failure, the retrieved order is retained.
- The answer cache is in-memory, shared only within the local Streamlit process, and resets on restart. The document index persists independently to `chroma_db/`.
- Session history is kept in Streamlit session state. Cache activity and contents are visible in the sidebar.
- Model API errors, unavailable models, missing keys, or insufficient evidence are not represented as successful cached answers.

## Tests

```powershell
.\virtual_environment\Scripts\python.exe -m pytest
```

Tests use lightweight fakes and do not call OpenAI or download models.

## Troubleshooting

- **Missing `OPENAI_API_KEY`:** add it to the project `.env`; do not put it in source or shell history.
- **Model not available:** confirm account access to `gpt-5.4-mini`; optionally set `OPENAI_MODEL` to an account-accessible model.
- **PDF not found / no text:** check `ACS880_PDF_PATH` or confirm the PDF is in the project root and readable.
- **Slow first startup:** initial CPU model download/indexing can take time. Later runs reuse the on-disk vector index.
- **Index errors:** use the UI's re-index action; check available disk space and local file permissions.
- **Page numbering:** source page metadata is converted to human-readable 1-based numbering; a missing page is displayed as unavailable, never guessed.
