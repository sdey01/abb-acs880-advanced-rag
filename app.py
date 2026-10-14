"""Streamlit UI for the local ABB ACS880 firmware manual assistant."""
from __future__ import annotations

import logging
import os
from datetime import datetime

# Initialize PyTorch's Windows native libraries before Streamlit loads its own
# numerical stack; loading Torch after Streamlit can fail to initialize c10.dll.
import torch  # noqa: F401
import streamlit as st

from config import (CACHE_MAX_ENTRIES, CACHE_THRESHOLD, CACHE_TTL_SECONDS,
                    EMBEDDING_MODEL, PDF_PATH, RERANKER_MODEL)
from src.embeddings import LocalEmbeddings
from src.rag_pipeline import RagPipeline
from src.reranker import CrossEncoderReranker
from src.semantic_cache import SemanticCache
from src.vector_store import ManualIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

st.set_page_config(
    page_title="ABB ACS880 Firmware Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1500px;}
    [data-testid="stSidebar"] {background: #f4f7fa;}
    .hero {padding: 1.25rem 1.5rem; border-radius: 14px; color: white;
           background: linear-gradient(110deg,#132f4c,#167b85); margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .4rem 0 0; opacity: .88;}
    .status-hit {background:#e8f6ed;color:#146c36;padding:.55rem .8rem;
                 border-radius:8px;font-weight:700;}
    .status-miss {background:#fff4df;color:#815500;padding:.55rem .8rem;
                  border-radius:8px;font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_services() -> tuple[LocalEmbeddings, ManualIndex, SemanticCache, RagPipeline]:
    embeddings = LocalEmbeddings()
    index = ManualIndex(PDF_PATH, embeddings)
    cache = SemanticCache(
        embeddings, ttl=CACHE_TTL_SECONDS, capacity=CACHE_MAX_ENTRIES,
        threshold=CACHE_THRESHOLD,
    )
    pipeline = RagPipeline(index, cache, CrossEncoderReranker())
    return embeddings, index, cache, pipeline


def citation(chunk) -> str:
    page = f"p. {chunk.page}" if chunk.page is not None else "page unavailable"
    return f"{chunk.source}, {page}"


def clear_history_selection() -> None:
    st.session_state.selected_history = None
    st.session_state.history_selection = None


def show_result(result) -> None:
    status_class = "status-hit" if result.cache_status == "CACHE HIT" else "status-miss"
    st.markdown(f'<div class="{status_class}">{result.cache_status}</div>', unsafe_allow_html=True)
    st.markdown("### Answer")
    st.markdown(result.answer)
    if result.sources:
        st.markdown("**Sources:** " + " · ".join(
            f"[{citation(chunk)}](#evidence-{chunk.chunk_id[:8]})"
            for chunk in result.sources
        ))
    metrics = st.columns(4)
    metrics[0].metric("Response time", f"{result.latency_seconds:.2f} s")
    metrics[1].metric("Category", result.category or "—")
    metrics[2].metric("Evidence chunks", len(result.sources))
    metrics[3].metric("Request ID", result.request_id or "—")
    if result.cache_status == "CACHE HIT":
        st.caption(
            f"Matched: “{result.cache_match_question}” · similarity "
            f"{result.cache_similarity:.3f} · {result.cache_ttl_seconds:.0f}s remaining. "
            "No classification, retrieval, reranking, or generation was run."
        )
    if result.reason:
        st.caption(f"Routing: {result.reason}")
    if result.fallbacks:
        for note in result.fallbacks:
            st.warning(note)
    if result.cache_status == "CACHE MISS" and result.expires_at:
        expiry = datetime.fromisoformat(result.expires_at).astimezone()
        st.caption(f"Cached in memory until {expiry:%Y-%m-%d %H:%M:%S %Z}.")

    with st.expander("Retrieved evidence and ranking", expanded=False):
        if result.cache_status == "CACHE HIT":
            st.info("Evidence below is stored with the cached answer; no new retrieval was performed.")
        candidates_by_id = {chunk.chunk_id: chunk for chunk in result.candidates}
        if result.candidates:
            st.markdown("**Initial candidates and reranked order**")
            for item in sorted(result.candidates, key=lambda chunk: chunk.reranked_rank or 999):
                score = (
                    f"{item.retrieval_score:.3f}"
                    if item.retrieval_score is not None else "n/a"
                )
                st.markdown(
                    f'<div id="evidence-{item.chunk_id[:8]}"></div>'
                    f"**Reranked #{item.reranked_rank or '—'}** · initial "
                    f"#{item.initial_rank or '—'} · {citation(item)} · retrieval "
                    f"{score}",
                    unsafe_allow_html=True,
                )
                st.write(item.text)
                st.divider()
        elif result.sources:
            for item in result.sources:
                st.markdown(f"**{citation(item)}**")
                st.write(item.text)
                st.divider()
        else:
            st.info("No evidence was returned.")
        if result.expanded_queries:
            st.markdown("**Query expansion**")
            st.write("Original: " + result.question)
            for query in result.expanded_queries:
                st.write(f"- {query}")
        if result.timings:
            st.markdown("**Stage timings (seconds)**")
            st.json({name: round(seconds, 3) for name, seconds in result.timings.items()})
        if result.candidate_budget:
            st.caption(
                f"Configured initial retrieval budget: {result.candidate_budget}; "
                f"unique candidates returned after multi-query deduplication: {len(candidates_by_id)}."
            )


st.markdown(
    '<div class="hero"><h1>ABB ACS880 Firmware Assistant</h1>'
    '<p>Answers grounded in the supplied firmware manual, with page-level evidence.</p></div>',
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "selected_history" not in st.session_state:
    st.session_state.selected_history = None

with st.sidebar:
    st.header("Session history")
    if st.button("Clear history", disabled=not st.session_state.history, use_container_width=True):
        st.session_state.history = []
        clear_history_selection()
        st.rerun()
    if st.session_state.history:
        if (st.session_state.selected_history is not None
                and st.button("Ask a new question", use_container_width=True)):
            clear_history_selection()
            st.rerun()
        choices = list(range(len(st.session_state.history)))
        selected = st.radio(
            "Review a previous answer",
            choices,
            index=(
                st.session_state.selected_history
                if st.session_state.selected_history in choices else None
            ),
            format_func=lambda idx: st.session_state.history[idx]["question"][:64],
            label_visibility="collapsed",
            key="history_selection",
        )
        st.session_state.selected_history = selected
    else:
        st.caption("Questions from this browser session will appear here.")

    st.divider()
    st.header("Manual status")
    st.write(f"PDF: **{'Found' if PDF_PATH.is_file() else 'Missing'}**")
    st.caption(str(PDF_PATH))

if not PDF_PATH.is_file():
    st.error(f"Manual not found. Set `ACS880_PDF_PATH` or place the PDF at `{PDF_PATH}`.")
    st.stop()

try:
    with st.spinner("Loading CPU models and validating the local manual index…"):
        embeddings, index, cache, pipeline = get_services()
    st.success(f"Index ready · {index.chunk_count:,} chunks · CPU embeddings")
except Exception as exc:
    logging.exception("Application initialization failed: %s", type(exc).__name__)
    st.error(
        "Could not initialize the local index/models. Check that required packages are installed, "
        "the PDF is readable, and the embedding models can be downloaded. "
        f"Technical error: {type(exc).__name__}: {exc}"
    )
    st.stop()

with st.sidebar:
    st.caption(f"Embedding: `{EMBEDDING_MODEL}` · reranker: `{RERANKER_MODEL}` (CPU)")
    if st.button("Re-index manual", use_container_width=True):
        try:
            with st.spinner("Rebuilding the local vector index…"):
                count = index.reindex()
            st.success(f"Index rebuilt with {count:,} chunks.")
        except Exception as exc:
            logging.exception("Manual re-index failed: %s", type(exc).__name__)
            st.error(f"Re-index failed ({type(exc).__name__}): {exc}")
    st.divider()
    st.header("Semantic cache")
    cache_stats = cache.stats()
    st.metric("Entries", f"{cache_stats['entry_count']} / {cache_stats['max_entries']}")
    st.caption(
        f"TTL: {cache_stats['ttl_seconds'] // 60} minutes · similarity threshold: "
        f"{CACHE_THRESHOLD:.2f} · expired removed: {cache_stats['expired_removed']}"
    )
    st.info("In-memory, shared by this local app process; resets when it restarts.")
    with st.expander("Recent cache activity"):
        for item in cache_stats["activity"]:
            st.write(f"{item['status']} · {item['question']}")
    with st.expander("Current cache entries"):
        for item in cache_stats["entries"]:
            st.write(f"**{item['question']}**")
            st.caption(f"Expires {item['expires_at']}")

if st.session_state.selected_history is not None:
    show_result(st.session_state.history[st.session_state.selected_history]["result"])
else:
    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "Ask a question about the ABB ACS880 manual",
            placeholder="For example: What does parameter 99.04 control?",
            height=100,
        )
        submitted = st.form_submit_button(
            "Ask the manual", type="primary", use_container_width=True,
            on_click=clear_history_selection,
        )
    if submitted:
        if not question.strip():
            st.warning("Enter a question before submitting.")
        else:
            try:
                with st.spinner("Checking cache and preparing an evidence-grounded answer…"):
                    result = pipeline.ask(question)
                st.session_state.history.insert(0, {"question": question.strip(), "result": result})
                st.session_state.selected_history = None
                show_result(result)
            except Exception as exc:
                logging.exception("Question request failed: %s", type(exc).__name__)
                if not os.getenv("OPENAI_API_KEY"):
                    st.error("OpenAI API key is missing. Add `OPENAI_API_KEY` to the existing project `.env`.")
                else:
                    st.error(
                        f"Could not complete this request ({type(exc).__name__}). "
                        "Check OpenAI model access, network connection, and local setup. "
                        "Credentials are not shown in diagnostics."
                    )
    else:
        st.caption(
            "The assistant searches the supplied PDF only. Review cited pages before using "
            "information for engineering decisions."
        )
