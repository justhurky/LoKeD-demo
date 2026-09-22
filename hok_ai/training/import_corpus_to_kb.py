from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from hok_ai.core.db import DEFAULT_CHROMA_PATH, DEFAULT_DB_PATH, Database, new_id, now_iso
from hok_ai.core.embeddings import OllamaEmbeddingClient
from hok_ai.core.knowledge_base import KnowledgeBase


def ollama_embed_many(client: OllamaEmbeddingClient, texts: list[str]) -> list[list[float]]:
    payload = {"model": client.model, "input": texts}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        client._url("/api/embed"),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(client.timeout, 300)) as response:
            body = json.loads(response.read().decode("utf-8"))
        embeddings = body.get("embeddings") or []
        if len(embeddings) == len(texts):
            return [[float(value) for value in embedding] for embedding in embeddings]
    except (urllib.error.URLError, json.JSONDecodeError):
        pass
    return [client.embed(text) for text in texts]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def import_corpus(
    corpus_dir: Path,
    *,
    db_path: Path,
    batch_size: int = 48,
    reset_knowledge_base: bool = False,
    chunks_only: bool = False,
    skip_existing_chunks: bool = False,
    sqlite_only: bool = False,
) -> dict[str, Any]:
    documents_path = corpus_dir / "documents.jsonl"
    chunks_path = corpus_dir / "chunks.jsonl"
    if not documents_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Hianyzik a documents.jsonl vagy chunks.jsonl.")

    db = Database(db_path)
    db.init()
    if reset_knowledge_base and not chunks_only:
        reset_documents_and_vectors(db)
    settings = db.get_settings()
    embeddings = OllamaEmbeddingClient(
        host=settings.get("ollama_host", "http://127.0.0.1:11434"),
        model=settings.get("embedding_model", "nomic-embed-text"),
        timeout=120,
    )
    kb = KnowledgeBase(db=db, embedding_client=embeddings)

    docs_seen = 0
    chunks_seen = 0
    chunks_by_doc: Counter[str] = Counter()
    doc_paths: dict[str, str] = {}
    doc_id_map: dict[str, str] = {}
    old_vector_ids: list[str] = []

    for doc in read_jsonl(documents_path):
        docs_seen += 1
        doc_id = doc["id"]
        existing = db.document_by_path(doc["path"])
        if chunks_only:
            if not existing:
                raise RuntimeError(f"Chunks-only importnal hianyzik a dokumentum a DB-bol: {doc['path']}")
            document = existing
        else:
            document = db.upsert_document(
                {
                    "id": existing.get("id") if existing else doc_id,
                    "path": doc["path"],
                    "name": doc.get("name", Path(doc["path"]).name),
                    "type": doc.get("type", "egyeb"),
                    "source_folder": str(Path(doc["path"]).parent),
                    "status": "folyamatban",
                    "file_hash": doc.get("sha256", ""),
                    "chunks": 0,
                    "is_regulation": int(bool(doc.get("is_regulation"))),
                    "regulation_name": doc.get("regulation_name", ""),
                    "paragraph_hint": "",
                }
            )
            removed = db.delete_document_chunks(document["id"])
            old_vector_ids.extend(removed)
        doc_id_map[doc_id] = document["id"]
        doc_paths[document["id"]] = doc["path"]

    if old_vector_ids and not reset_knowledge_base:
        for batch_ids in batched(old_vector_ids, 500):
            try:
                kb.collection.delete(ids=batch_ids)
            except Exception:
                pass

    existing_vector_ids = load_existing_vector_ids(db) if skip_existing_chunks else set()
    skipped_chunks = 0
    batch: list[dict[str, Any]] = []
    for chunk in read_jsonl(chunks_path):
        if chunk["id"] in existing_vector_ids:
            skipped_chunks += 1
            continue
        batch.append(chunk)
        if len(batch) >= batch_size:
            chunks_seen += import_chunk_batch(db, kb, embeddings, batch, chunks_by_doc, doc_id_map, sqlite_only=sqlite_only)
            if chunks_seen % max(batch_size * 20, 1) == 0:
                print(json.dumps({"progress": "chunks", "imported": chunks_seen}, ensure_ascii=False), flush=True)
            batch = []
    if batch:
        chunks_seen += import_chunk_batch(db, kb, embeddings, batch, chunks_by_doc, doc_id_map, sqlite_only=sqlite_only)

    finalize_document_counts(db)

    db.log(
        "info",
        "JSON korpusz importalva a tudastarba",
        {"corpus_dir": str(corpus_dir), "documents": docs_seen, "chunks": chunks_seen, "skipped_chunks": skipped_chunks},
    )
    return {
        "documents": docs_seen,
        "chunks": chunks_seen,
        "skipped_chunks": skipped_chunks,
        "batch_size": batch_size,
        "knowledge": kb.stats(),
    }


def reset_documents_and_vectors(db: Database) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM document_chunks")
        conn.execute("DELETE FROM documents")
    if DEFAULT_CHROMA_PATH.exists():
        shutil.rmtree(DEFAULT_CHROMA_PATH)


def load_existing_vector_ids(db: Database) -> set[str]:
    with db.connect() as conn:
        rows = conn.execute("SELECT vector_id FROM document_chunks").fetchall()
    return {row["vector_id"] for row in rows}


def finalize_document_counts(db: Database) -> None:
    now = now_iso()
    with db.connect() as conn:
        rows = conn.execute("SELECT document_id, COUNT(*) AS chunk_count FROM document_chunks GROUP BY document_id").fetchall()
        conn.execute(
            """
            UPDATE documents
            SET chunks = 0,
                status = 'indexelt',
                indexed_at = ?,
                updated_at = ?,
                error = ''
            """,
            (now, now),
        )
        conn.executemany(
            "UPDATE documents SET chunks = ? WHERE id = ?",
            [(row["chunk_count"], row["document_id"]) for row in rows],
        )


def import_chunk_batch(
    db: Database,
    kb: KnowledgeBase,
    embeddings: OllamaEmbeddingClient,
    batch: list[dict[str, Any]],
    chunks_by_doc: Counter[str],
    doc_id_map: dict[str, str],
    *,
    sqlite_only: bool = False,
) -> int:
    texts = [item["text"] for item in batch]
    vectors = [] if sqlite_only else ollama_embed_many(embeddings, texts)
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    db_rows: list[tuple[str, str, int, str, str, str, str]] = []
    for index, item in enumerate(batch):
        corpus_doc_id = item["document_id"]
        doc_id = doc_id_map.get(corpus_doc_id, corpus_doc_id)
        vector_id = item["id"]
        metadata = {
            "document_id": doc_id,
            "document_name": item.get("name", ""),
            "path": item.get("path", ""),
            "type": item.get("type", ""),
            "chunk_index": item.get("chunk_index", 0),
            "paragraph": item.get("paragraph_hint", ""),
        }
        ids.append(vector_id)
        metadatas.append(metadata)
        db_rows.append(
            (
                new_id("chunk"),
                doc_id,
                int(item.get("chunk_index", 0)),
                vector_id,
                item["text"],
                json.dumps(metadata, ensure_ascii=False),
                now_iso(),
            )
        )
        chunks_by_doc[doc_id] += 1
    with db.connect() as conn:
        conn.executemany(
            """
            INSERT INTO document_chunks(id, document_id, chunk_index, vector_id, text, metadata_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(vector_id) DO UPDATE SET
                document_id = excluded.document_id,
                chunk_index = excluded.chunk_index,
                text = excluded.text,
                metadata_json = excluded.metadata_json,
                created_at = excluded.created_at
            """,
            db_rows,
        )
    if not sqlite_only:
        kb.collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
    return len(batch)


def batched(items: list[str], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JSON korpusz importalasa a HOK AI lokalis tudastarba")
    parser.add_argument("--corpus-dir", default="hok_ai/data/training/ehok_corpus")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--reset-knowledge-base", action="store_true", help="Dokumentum index es Chroma vektorok torlese import elott.")
    parser.add_argument("--chunks-only", action="store_true", help="Dokumentum rekordok ujraepitese nelkul csak chunkokat importal.")
    parser.add_argument("--skip-existing-chunks", action="store_true", help="Mar DB-ben levo vector_id-k kihagyasa resume importnal.")
    parser.add_argument("--sqlite-only", action="store_true", help="Csak SQLite teljes-szoveges chunk import, vektor embedding nelkul.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = import_corpus(
        Path(args.corpus_dir),
        db_path=Path(args.db),
        batch_size=args.batch_size,
        reset_knowledge_base=args.reset_knowledge_base,
        chunks_only=args.chunks_only,
        skip_existing_chunks=args.skip_existing_chunks,
        sqlite_only=args.sqlite_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
