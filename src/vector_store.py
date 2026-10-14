"""Persistent Chroma index and multi-query candidate retrieval."""
from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb

from config import CHROMA_PATH, COLLECTION_NAME
from src.ingestion import fingerprint, load_chunks
from src.schemas import Chunk


class ManualIndex:
    def __init__(self, path: Path, embeddings) -> None:
        self.path, self.embeddings = path, embeddings
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = None
        self.chunk_count = 0
        self._open_or_build()

    def _open_or_build(self, force: bool = False) -> None:
        source_id = fingerprint(self.path)
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            collection = None
        if collection is not None and not force:
            metadata = collection.metadata or {}
            if (metadata.get("source_fingerprint") == source_id and
                    metadata.get("embedding_model") == self.embeddings.model_name and collection.count()):
                self.collection = collection
                self.chunk_count = collection.count()
                return
        if collection is not None:
            self.client.delete_collection(COLLECTION_NAME)
        docs = load_chunks(self.path)
        self.collection = self.client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "source_fingerprint": source_id,
                      "embedding_model": self.embeddings.model_name},
        )
        for start in range(0, len(docs), 128):
            batch = docs[start:start + 128]
            texts = [doc.page_content for doc in batch]
            ids = [hashlib.sha256(f"{source_id}:{start+i}:{text}".encode()).hexdigest()
                   for i, text in enumerate(texts)]
            metadata = [{
                "source": str(doc.metadata.get("source", self.path.name)),
                "page_number": int(doc.metadata["page_number"])
                if doc.metadata.get("page_number") is not None else -1,
            } for doc in batch]
            self.collection.add(ids=ids, documents=texts, metadatas=metadata,
                                embeddings=self.embeddings.embed_documents(texts))
        self.chunk_count = self.collection.count()

    def reindex(self) -> int:
        self._open_or_build(force=True)
        return self.chunk_count

    def retrieve(self, queries: list[str], budget: int) -> list[Chunk]:
        if not self.collection or not self.chunk_count:
            return []
        found: dict[str, Chunk] = {}
        for query in queries:
            result = self.collection.query(
                query_embeddings=[self.embeddings.embed_query(query)],
                n_results=min(budget, self.chunk_count),
                include=["documents", "metadatas", "distances"],
            )
            for rank, (identifier, text, metadata, distance) in enumerate(zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0],
                result["distances"][0]), 1):
                score = 1 - float(distance)
                chunk = Chunk(
                    chunk_id=identifier, text=text,
                    source=str(metadata.get("source", self.path.name)),
                    page=int(metadata["page_number"]) if int(metadata.get("page_number", -1)) > 0 else None,
                    retrieval_score=score, initial_rank=rank,
                )
                if identifier not in found or score > (found[identifier].retrieval_score or -1):
                    found[identifier] = chunk
        return sorted(found.values(), key=lambda item: item.retrieval_score or 0, reverse=True)[:budget]
