from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import DEFAULT_CHROMA_PATH, REGULATIONS_DIR, Database, now_iso
from .document_loader import SUPPORTED_EXTENSIONS, chunk_text, classify_document, load_document
from .embeddings import OllamaEmbeddingClient


os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@dataclass(slots=True)
class SourceChunk:
    id: str
    document_name: str
    path: str
    score: float
    snippet: str
    paragraph: str = ""
    type: str = ""


class KnowledgeBase:
    def __init__(
        self,
        db: Database,
        embedding_client: Any | None = None,
        chroma_path: str | Path = DEFAULT_CHROMA_PATH,
        collection_name: str = "hok_ai_documents",
    ) -> None:
        self.db = db
        self.embedding_client = embedding_client or OllamaEmbeddingClient()
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self.collection = self._open_collection()

    def _open_collection(self) -> Any:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError("A ChromaDB nincs telepitve. Futtasd: pip install -r requirements.txt") from exc

        self.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_file(self, path: str | Path, *, force: bool = False) -> dict[str, Any]:
        path = Path(path)
        loaded = load_document(path)
        existing = self.db.document_by_path(path)
        if existing and existing.get("file_hash") == loaded.metadata["file_hash"] and not force:
            return {"indexed": 0, "chunks": existing.get("chunks", 0), "skipped": True, "document": existing}

        classification = classify_document(path)
        document = self.db.upsert_document(
            {
                "id": existing.get("id") if existing else None,
                "path": path,
                "name": path.name,
                "type": classification["type"],
                "source_folder": str(path.parent),
                "status": "folyamatban",
                "file_hash": loaded.metadata["file_hash"],
                "chunks": 0,
                "is_regulation": classification["is_regulation"],
                "regulation_name": classification["regulation_name"],
            }
        )
        old_ids = self.db.delete_document_chunks(document["id"])
        if old_ids:
            try:
                self.collection.delete(ids=old_ids)
            except Exception:
                pass

        chunks = chunk_text(loaded.text)
        if not chunks:
            document = self.db.upsert_document(
                {
                    **document,
                    "path": path,
                    "status": "hiba",
                    "error": "Nem nyerheto ki szoveg a dokumentumbol.",
                    "chunks": 0,
                }
            )
            return {"indexed": 0, "chunks": 0, "skipped": False, "document": document}

        vector_ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            vector_id = f"{document['id']}:{chunk.index}"
            metadata = {
                "document_id": document["id"],
                "document_name": path.name,
                "path": str(path),
                "type": classification["type"],
                "chunk_index": chunk.index,
                "paragraph": chunk.paragraph_hint,
                "is_regulation": bool(classification["is_regulation"]),
                "regulation_name": classification["regulation_name"],
            }
            vector_ids.append(vector_id)
            documents.append(chunk.text)
            metadatas.append(metadata)
            embeddings.append(self.embedding_client.embed(chunk.text))
            self.db.add_document_chunk(document["id"], chunk.index, vector_id, chunk.text, metadata)

        self.collection.add(ids=vector_ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        paragraph_hint = next((chunk.paragraph_hint for chunk in chunks if chunk.paragraph_hint), "")
        document = self.db.upsert_document(
            {
                **document,
                "path": path,
                "status": "indexelt",
                "indexed_at": now_iso(),
                "error": "",
                "chunks": len(chunks),
                "paragraph_hint": paragraph_hint,
            }
        )
        self.db.log("info", "Dokumentum indexelve", {"path": str(path), "chunks": len(chunks)})
        return {"indexed": 1, "chunks": len(chunks), "skipped": False, "document": document}

    def index_path(self, path: str | Path, *, force: bool = False) -> dict[str, Any]:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"A megadott mappa vagy fajl nem talalhato: {root}")
        files = [root] if root.is_file() else [item for item in root.rglob("*") if item.suffix.lower() in SUPPORTED_EXTENSIONS]
        indexed = 0
        skipped = 0
        errors: list[dict[str, str]] = []
        chunks = 0
        for file_path in files:
            try:
                result = self.index_file(file_path, force=force)
                indexed += int(result.get("indexed", 0))
                skipped += 1 if result.get("skipped") else 0
                chunks += int(result.get("chunks", 0) or 0)
            except Exception as exc:
                errors.append({"path": str(file_path), "error": str(exc)})
                self.db.upsert_document(
                    {
                        "path": file_path,
                        "name": file_path.name,
                        "type": classify_document(file_path)["type"],
                        "source_folder": str(file_path.parent),
                        "status": "hiba",
                        "error": str(exc),
                        "chunks": 0,
                    }
                )
        return {"indexed": indexed, "skipped": skipped, "chunks": chunks, "errors": errors}

    def index_regulations(self) -> dict[str, Any]:
        REGULATIONS_DIR.mkdir(parents=True, exist_ok=True)
        return self.index_path(REGULATIONS_DIR)

    def search(self, query: str, *, limit: int = 5, min_score: float = 0.65) -> list[SourceChunk]:
        if not query.strip():
            return []
        hits: list[SourceChunk] = []
        seen: set[str] = set()
        try:
            query_embedding = self.embedding_client.embed(query)
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            ids = (result.get("ids") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]

            for item_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
                score = max(0.0, min(1.0, 1.0 - float(distance)))
                if score <= min_score:
                    continue
                seen.add(item_id)
                hits.append(
                    SourceChunk(
                        id=item_id,
                        document_name=metadata.get("document_name", ""),
                        path=metadata.get("path", ""),
                        score=round(score, 4),
                        snippet=(text or "")[:900],
                        paragraph=metadata.get("paragraph", ""),
                        type=metadata.get("type", ""),
                    )
                )
        except Exception as exc:
            self.db.log("warning", "Vektoros kereses sikertelen, keyword fallback indul", {"error": str(exc)})

        if len(hits) < limit:
            for item in self.db.search_chunks_keyword(query, limit=limit * 3):
                item_id = item.get("id", "")
                if item_id in seen:
                    continue
                hits.append(
                    SourceChunk(
                        id=item_id,
                        document_name=item.get("document_name", ""),
                        path=item.get("path", ""),
                        score=0.7,
                        snippet=(item.get("text") or "")[:900],
                        type=item.get("type", ""),
                    )
                )
                seen.add(item_id)
                if len(hits) >= limit:
                    break
        return hits

    def documents(self, keyword: str = "", doc_type: str = "") -> list[dict[str, Any]]:
        return self.db.documents(keyword=keyword, doc_type=doc_type)

    def stats(self) -> dict[str, int]:
        documents = self.db.documents(limit=100000)
        return {
            "documents": len(documents),
            "chunks": sum(int(item.get("chunks") or 0) for item in documents),
            "errors": len([item for item in documents if item.get("status") == "hiba"]),
        }


def source_chunks_to_json(chunks: list[SourceChunk]) -> list[dict[str, Any]]:
    return [
        {
            "id": chunk.id,
            "document_name": chunk.document_name,
            "path": chunk.path,
            "score": chunk.score,
            "snippet": chunk.snippet,
            "paragraph": chunk.paragraph,
            "type": chunk.type,
        }
        for chunk in chunks
    ]
