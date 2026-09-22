from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "hok_ai.sqlite"
DEFAULT_CHROMA_PATH = DATA_DIR / "chromadb"
DOCUMENTS_DIR = DATA_DIR / "dokumentumok"
REGULATIONS_DIR = DATA_DIR / "szabalyzatok"
EXPORT_DIR = DATA_DIR / "exports"
AUDIO_DIR = DATA_DIR / "audio"
DOMAIN_ADAPTER_MODEL = "hok-ai-domain-4b-peft"
DOMAIN_ADAPTER_URL = "http://127.0.0.1:7843"
DOMAIN_ADAPTER_PATH = DATA_DIR / "models" / DOMAIN_ADAPTER_MODEL


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, DEFAULT_CHROMA_PATH, DOCUMENTS_DIR, REGULATIONS_DIR, EXPORT_DIR, AUDIO_DIR):
        path.mkdir(parents=True, exist_ok=True)


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [row_to_dict(row) or {} for row in rows]


@dataclass(slots=True)
class Database:
    path: Path = DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self) -> None:
        ensure_data_dirs()
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    source_folder TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'indexelt',
                    file_hash TEXT NOT NULL DEFAULT '',
                    indexed_at TEXT,
                    updated_at TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    chunks INTEGER NOT NULL DEFAULT 0,
                    is_regulation INTEGER NOT NULL DEFAULT 0,
                    regulation_name TEXT NOT NULL DEFAULT '',
                    paragraph_hint TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    vector_id TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    owner TEXT NOT NULL DEFAULT '',
                    deadline TEXT NOT NULL DEFAULT '',
                    priority TEXT NOT NULL DEFAULT 'közepes',
                    status TEXT NOT NULL DEFAULT 'Nyitott',
                    source_sentence TEXT NOT NULL DEFAULT '',
                    related_decision TEXT NOT NULL DEFAULT '',
                    related_meeting TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    topic TEXT NOT NULL DEFAULT '',
                    date TEXT NOT NULL DEFAULT '',
                    vote TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'rogzitett',
                    source TEXT NOT NULL DEFAULT '',
                    related_task TEXT NOT NULL DEFAULT '',
                    meeting_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT '',
                    committee TEXT NOT NULL DEFAULT '',
                    email TEXT NOT NULL DEFAULT '',
                    phone TEXT NOT NULL DEFAULT '',
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS meetings (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    raw_text TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    analysis_json TEXT NOT NULL DEFAULT '{}',
                    chaos_index INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS logs (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )
            self._seed_defaults(conn)

    def _seed_defaults(self, conn: sqlite3.Connection) -> None:
        defaults = {
            "ollama_host": "http://127.0.0.1:11434",
            "chat_model": "qwen2.5:7b-instruct",
            "fallback_model": "gemma3:4b",
            "embedding_model": "nomic-embed-text",
            "whisper_model": "tiny",
            "meeting_model": "qwen2.5:7b-instruct",
            "minutes_model": "qwen2.5:7b-instruct",
            "custom_4b_model": DOMAIN_ADAPTER_MODEL,
            "file_model": "qwen2.5:7b-instruct",
            "peft_adapter_enabled": "false",
            "peft_adapter_url": DOMAIN_ADAPTER_URL,
            "peft_adapter_path": str(DOMAIN_ADAPTER_PATH),
            "peft_adapter_model": DOMAIN_ADAPTER_MODEL,
            "watched_folder": "",
            "notifications_enabled": "true",
            "hunspell_enabled": "true",
        }
        for key, value in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)",
                (key, value),
            )

    def get_settings(self) -> dict[str, str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def set_settings(self, settings: dict[str, Any]) -> dict[str, str]:
        with self.connect() as conn:
            for key, value in settings.items():
                conn.execute(
                    """
                    INSERT INTO settings(key, value)
                    VALUES(?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, str(value)),
                )
        return self.get_settings()

    def log(self, level: str, message: str, data: dict[str, Any] | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO logs(id, level, message, data_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (new_id("log"), level, message, json.dumps(data or {}, ensure_ascii=False), now_iso()),
            )

    def logs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM logs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return rows_to_dicts(rows)

    def upsert_document(self, payload: dict[str, Any]) -> dict[str, Any]:
        doc_id = payload.get("id") or new_id("doc")
        now = now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(
                    id, path, name, type, source_folder, status, file_hash,
                    indexed_at, updated_at, error, chunks, is_regulation,
                    regulation_name, paragraph_hint
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name,
                    type = excluded.type,
                    source_folder = excluded.source_folder,
                    status = excluded.status,
                    file_hash = excluded.file_hash,
                    indexed_at = excluded.indexed_at,
                    updated_at = excluded.updated_at,
                    error = excluded.error,
                    chunks = excluded.chunks,
                    is_regulation = excluded.is_regulation,
                    regulation_name = excluded.regulation_name,
                    paragraph_hint = excluded.paragraph_hint
                """,
                (
                    doc_id,
                    str(payload["path"]),
                    payload.get("name") or Path(payload["path"]).name,
                    payload.get("type", "egyeb"),
                    payload.get("source_folder", ""),
                    payload.get("status", "indexelt"),
                    payload.get("file_hash", ""),
                    payload.get("indexed_at", now),
                    now,
                    payload.get("error", ""),
                    int(payload.get("chunks", 0)),
                    1 if payload.get("is_regulation") else 0,
                    payload.get("regulation_name", ""),
                    payload.get("paragraph_hint", ""),
                ),
            )
            row = conn.execute("SELECT * FROM documents WHERE path = ?", (str(payload["path"]),)).fetchone()
        return row_to_dict(row) or {}

    def document_by_path(self, path: str | Path) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE path = ?", (str(path),)).fetchone()
        return row_to_dict(row)

    def document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return row_to_dict(row)

    def documents(self, keyword: str = "", doc_type: str = "", limit: int = 500) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if keyword:
            clauses.append("(name LIKE ? OR path LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        if doc_type and doc_type != "all":
            clauses.append("type = ?")
            params.append(doc_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM documents {where} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        return rows_to_dicts(rows)

    def search_chunks_keyword(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        terms = [term.strip().lower() for term in query.replace("\n", " ").split() if len(term.strip()) >= 3]
        if not terms:
            return []
        clauses = []
        params: list[Any] = []
        for term in terms[:8]:
            clauses.append("(LOWER(dc.text) LIKE ? OR LOWER(d.name) LIKE ? OR LOWER(d.path) LIKE ?)")
            like = f"%{term}%"
            params.extend([like, like, like])
        params.append(max(limit * 8, limit))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    dc.vector_id AS id,
                    dc.text AS text,
                    dc.metadata_json AS metadata_json,
                    d.name AS document_name,
                    d.path AS path,
                    d.type AS type
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE {' OR '.join(clauses)}
                LIMIT ?
                """,
                params,
            ).fetchall()
        candidates = rows_to_dicts(rows)
        scored: list[dict[str, Any]] = []
        for item in candidates:
            text = f"{item.get('document_name', '')} {item.get('path', '')} {item.get('text', '')}".lower()
            score = sum(1 for term in terms if term in text)
            if score:
                item["keyword_score"] = score
                scored.append(item)
        scored.sort(key=lambda item: item.get("keyword_score", 0), reverse=True)
        return scored[:limit]

    def delete_document_chunks(self, document_id: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT vector_id FROM document_chunks WHERE document_id = ?",
                (document_id,),
            ).fetchall()
            vector_ids = [row["vector_id"] for row in rows]
            conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        return vector_ids

    def add_document_chunk(
        self,
        document_id: str,
        chunk_index: int,
        vector_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO document_chunks(id, document_id, chunk_index, vector_id, text, metadata_json, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("chunk"),
                    document_id,
                    chunk_index,
                    vector_id,
                    text,
                    json.dumps(metadata, ensure_ascii=False),
                    now_iso(),
                ),
            )

    def list_tasks(self, status: str = "", owner: str = "", priority: str = "") -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if owner:
            clauses.append("owner LIKE ?")
            params.append(f"%{owner}%")
        if priority:
            clauses.append("priority = ?")
            params.append(priority)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM tasks {where} ORDER BY deadline = '', deadline ASC, updated_at DESC",
                params,
            ).fetchall()
        return rows_to_dicts(rows)

    def create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        title = (payload.get("title") or "").strip()
        if not title:
            raise ValueError("A feladat cime kotelezo.")
        task_id = payload.get("id") or new_id("task")
        now = now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks(
                    id, title, description, owner, deadline, priority, status,
                    source_sentence, related_decision, related_meeting, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    title,
                    payload.get("description", ""),
                    payload.get("owner", ""),
                    payload.get("deadline", ""),
                    payload.get("priority", "közepes"),
                    payload.get("status", "Nyitott"),
                    payload.get("source_sentence", ""),
                    payload.get("related_decision", ""),
                    payload.get("related_meeting", ""),
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return row_to_dict(row) or {}

    def update_task(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "title",
            "description",
            "owner",
            "deadline",
            "priority",
            "status",
            "source_sentence",
            "related_decision",
            "related_meeting",
        }
        fields = [key for key in payload if key in allowed]
        if not fields:
            task = self.task(task_id)
            if task is None:
                raise KeyError(task_id)
            return task
        params = [payload[key] for key in fields]
        params.extend([now_iso(), task_id])
        set_sql = ", ".join(f"{key} = ?" for key in fields)
        with self.connect() as conn:
            conn.execute(f"UPDATE tasks SET {set_sql}, updated_at = ? WHERE id = ?", params)
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row_to_dict(row) or {}

    def task(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return row_to_dict(row)

    def delete_task(self, task_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cur.rowcount > 0

    def create_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = (payload.get("text") or payload.get("szoveg") or "").strip()
        if not text:
            raise ValueError("A dontes szovege kotelezo.")
        decision_id = payload.get("id") or new_id("decision")
        now = now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions(
                    id, text, topic, date, vote, status, source,
                    related_task, meeting_id, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    text,
                    payload.get("topic", ""),
                    payload.get("date") or payload.get("datum", ""),
                    payload.get("vote") or payload.get("szavazat", ""),
                    payload.get("status", "rogzitett"),
                    payload.get("source", ""),
                    payload.get("related_task", ""),
                    payload.get("meeting_id", ""),
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
        return row_to_dict(row) or {}

    def decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return rows_to_dicts(rows)

    def create_meeting(self, payload: dict[str, Any]) -> dict[str, Any]:
        meeting_id = payload.get("id") or new_id("meeting")
        analysis = payload.get("analysis") or {}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO meetings(id, title, raw_text, summary, analysis_json, chaos_index, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meeting_id,
                    payload.get("title", ""),
                    payload.get("raw_text", ""),
                    payload.get("summary", ""),
                    json.dumps(analysis, ensure_ascii=False),
                    int(payload.get("chaos_index", analysis.get("kaosz_index", 0) or 0)),
                    now_iso(),
                ),
            )
            row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        return row_to_dict(row) or {}

    def meetings(self, limit: int = 3) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM meetings ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return rows_to_dicts(rows)

    def list_people(self, keyword: str = "") -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if keyword:
            where = "WHERE name LIKE ? OR role LIKE ? OR committee LIKE ? OR aliases_json LIKE ?"
            params = [f"%{keyword}%"] * 4
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM people {where} ORDER BY name COLLATE NOCASE ASC",
                params,
            ).fetchall()
        result = rows_to_dicts(rows)
        for person in result:
            person["aliases"] = json.loads(person.pop("aliases_json") or "[]")
        return result

    def create_person(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = (payload.get("name") or "").strip()
        if not name:
            raise ValueError("A nev kotelezo.")
        person_id = payload.get("id") or new_id("person")
        now = now_iso()
        aliases = payload.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [item.strip() for item in aliases.split(",") if item.strip()]
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO people(
                    id, name, role, committee, email, phone, aliases_json,
                    notes, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    name,
                    payload.get("role", ""),
                    payload.get("committee", ""),
                    payload.get("email", ""),
                    payload.get("phone", ""),
                    json.dumps(aliases, ensure_ascii=False),
                    payload.get("notes", ""),
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()
        person = row_to_dict(row) or {}
        person["aliases"] = json.loads(person.pop("aliases_json", "[]") or "[]")
        return person

    def update_person(self, person_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {"name", "role", "committee", "email", "phone", "aliases", "notes"}
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in allowed:
                continue
            if key == "aliases":
                if isinstance(value, str):
                    value = [item.strip() for item in value.split(",") if item.strip()]
                normalized["aliases_json"] = json.dumps(value, ensure_ascii=False)
            else:
                normalized[key] = value
        if not normalized:
            people = [item for item in self.list_people() if item["id"] == person_id]
            if not people:
                raise KeyError(person_id)
            return people[0]
        fields = list(normalized.keys())
        params = [normalized[key] for key in fields]
        params.extend([now_iso(), person_id])
        set_sql = ", ".join(f"{key} = ?" for key in fields)
        with self.connect() as conn:
            conn.execute(f"UPDATE people SET {set_sql}, updated_at = ? WHERE id = ?", params)
            row = conn.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()
        if row is None:
            raise KeyError(person_id)
        person = row_to_dict(row) or {}
        person["aliases"] = json.loads(person.pop("aliases_json", "[]") or "[]")
        return person

    def delete_person(self, person_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        return cur.rowcount > 0

    def dashboard(self) -> dict[str, Any]:
        tasks = self.list_tasks()
        open_tasks = [task for task in tasks if task["status"] != "Kész"]
        urgent = [
            task
            for task in open_tasks
            if task.get("priority") == "sürgős" or not task.get("owner") or not task.get("deadline")
        ][:5]
        decisions = self.decisions(limit=100)
        meetings = self.meetings(limit=3)
        chaos_values = [int(item.get("chaos_index") or 0) for item in meetings]
        chaos_index = round(sum(chaos_values) / len(chaos_values)) if chaos_values else 0
        return {
            "counts": {
                "open_tasks": len(open_tasks),
                "upcoming_deadlines": self._upcoming_deadline_count(open_tasks),
                "decisions": len(decisions),
                "chaos_index": chaos_index,
            },
            "urgent_tasks": urgent,
            "recent_meetings": meetings,
            "decisions": decisions[:5],
        }

    def _upcoming_deadline_count(self, tasks: list[dict[str, Any]]) -> int:
        today = datetime.now().date()
        count = 0
        for task in tasks:
            raw = (task.get("deadline") or "").strip()
            if not raw:
                continue
            for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y.%m.%d."):
                try:
                    deadline = datetime.strptime(raw, fmt).date()
                    if 0 <= (deadline - today).days <= 7:
                        count += 1
                    break
                except ValueError:
                    continue
        return count
