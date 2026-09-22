from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from hok_ai.core.db import DOCUMENTS_DIR
from hok_ai.core.document_loader import SUPPORTED_EXTENSIONS, LoadedDocument, chunk_text, load_document
from hok_ai.core.knowledge_base import SourceChunk, source_chunks_to_json
from hok_ai.core.llm import AGENT_PROMPT, FALLBACK_MODEL, OllamaChatError, choose_model


router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


FILE_CHAT_SYSTEM_PROMPT = """Te egy magyar HOK/EHOK dokumentumasszisztens vagy.

A felhasznalo feltolt egy fajlt es ad melle egy utasitast. A valaszodat a
feltoltott fajl tartalmara epitsd. Ha javitast, atfogalmazast vagy hivatalos
szovegezest ker, add vissza a javitott szoveget jol olvashato formaban.

Szabalyok:
- mindig magyarul valaszolj,
- ne talalj ki olyan adatot, ami nincs a fajlban,
- ha a fajlbol csak reszleteket kaptal, ezt jelezd,
- ha dokumentumjavitast kersz, tartsd meg az eredeti ertelmet,
- hivatalos HOK/EHOK stilusnal legyel tomor, tiszta es adminisztrativ.
"""


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _strip_think(message: str) -> tuple[str, bool]:
    stripped = message.strip()
    if stripped.lower().startswith("/think"):
        return stripped[6:].strip(), True
    complex_markers = ("dontesi helyzet", "merlegeld", "kockazat", "strategia", "bonyolult")
    return stripped, any(marker in stripped.lower() for marker in complex_markers)


def _build_operational_context(ctx: Any, query: str) -> tuple[str, list[dict[str, Any]]]:
    lowered = query.lower()
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    if any(word in lowered for word in ("feladat", "hatarido", "határidő", "nyitott")):
        tasks = ctx.db.list_tasks()[:20]
        if tasks:
            blocks.append("Aktualis feladatok:")
            for task in tasks:
                blocks.append(
                    f"- {task['title']} | felelos: {task.get('owner') or 'nincs'} | "
                    f"hatarido: {task.get('deadline') or 'nincs'} | statusz: {task.get('status')}"
                )
            sources.append(
                {
                    "id": "sqlite:tasks",
                    "document_name": "Feladatok adatbazis",
                    "path": str(ctx.db.path),
                    "score": 1.0,
                    "snippet": "Lokalis SQLite feladatlista.",
                    "paragraph": "",
                    "type": "sqlite",
                }
            )
    if any(word in lowered for word in ("dontes", "döntés", "elfogadtuk")):
        decisions = ctx.db.decisions(limit=20)
        if decisions:
            blocks.append("Rogzitett dontesek:")
            for decision in decisions:
                blocks.append(
                    f"- {decision['text']} | datum: {decision.get('date') or 'nincs'} | "
                    f"tema: {decision.get('topic') or 'nincs'}"
                )
            sources.append(
                {
                    "id": "sqlite:decisions",
                    "document_name": "Dontesek adatbazis",
                    "path": str(ctx.db.path),
                    "score": 1.0,
                    "snippet": "Lokalis SQLite dontestar.",
                    "paragraph": "",
                    "type": "sqlite",
                }
            )
    if any(word in lowered for word in ("ember", "felelos", "felelős", "szerepkor", "szerepkör")):
        people = ctx.db.list_people()[:20]
        if people:
            blocks.append("People Memory:")
            for person in people:
                aliases = ", ".join(person.get("aliases") or [])
                blocks.append(
                    f"- {person['name']} | szerepkor: {person.get('role') or 'nincs'} | "
                    f"bizottsag: {person.get('committee') or 'nincs'} | aliasok: {aliases}"
                )
            sources.append(
                {
                    "id": "sqlite:people",
                    "document_name": "People Memory",
                    "path": str(ctx.db.path),
                    "score": 1.0,
                    "snippet": "Lokalis People Memory adat.",
                    "paragraph": "",
                    "type": "sqlite",
                }
            )
    return "\n".join(blocks), sources


def _context_from_sources(sources: list[SourceChunk]) -> str:
    blocks = []
    for index, source in enumerate(sources, start=1):
        paragraph = f" | paragrafus: {source.paragraph}" if source.paragraph else ""
        blocks.append(
            f"---{index}---\n"
            f"Forras: {source.document_name}{paragraph}\n"
            f"Utvonal: {source.path}\n"
            f"Score: {source.score}\n"
            f"{source.snippet}"
        )
    return "\n\n".join(blocks)


def _messages(query: str, sources: list[SourceChunk], operational_context: str) -> list[dict[str, str]]:
    document_context = _context_from_sources(sources) if sources else "Nincs relevans dokumentumreszlet."
    if operational_context:
        document_context += f"\n\nLokalis alkalmazas-kontekstus:\n{operational_context}"
    user_message = f"""Dokumentumreszletek:
{document_context}

Kerdes:
{query}
"""
    return [{"role": "system", "content": AGENT_PROMPT}, {"role": "user", "content": user_message}]


@router.post("/chat/file")
def chat_with_file(
    request: Request,
    instruction: str = Form(""),
    file: UploadFile = File(...),
    index: bool = Form(True),
) -> dict[str, Any]:
    ctx = request.app.state.context
    instruction = instruction.strip() or "Foglaldd ossze a feltoltott fajlt, es emeld ki a fontos teendoket."
    target = _save_chat_file(file)
    try:
        loaded = load_document(target)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not loaded.text.strip():
        raise HTTPException(status_code=400, detail="A feltoltott fajlbol nem sikerult szoveget kinyerni.")

    index_result = None
    if index:
        try:
            index_result = ctx.kb.index_file(target, force=True)
        except Exception as exc:
            ctx.db.log("warning", "Chat fajl indexelese sikertelen", {"path": str(target), "error": str(exc)})

    settings = ctx.db.get_settings()
    document_context, used_chunks, truncated = _document_context_for_instruction(loaded, instruction)
    messages = _file_messages(loaded, instruction, document_context, truncated=truncated)
    model = ""
    answer: Any = None
    model_errors: dict[str, str] = {}
    for candidate in file_model_candidates(settings, ctx):
        try:
            answer = ctx.llm.chat(
                messages,
                model=candidate,
                stream=False,
                think=False,
                timeout=240,
                options={"temperature": 0.1, "num_predict": 1800, "num_ctx": 8192},
            )
            model = candidate
            break
        except Exception as exc:
            model_errors[candidate] = str(exc)
            ctx.db.log("warning", "Fajlos chat modell sikertelen", {"model": candidate, "error": str(exc)})
    if answer is None:
        raise HTTPException(
            status_code=502,
            detail={"message": "Egyik helyi fajlfeldolgozo modell sem tudott valaszolni.", "errors": model_errors},
        )

    return {
        "answer": str(answer).strip(),
        "model": model,
        "file": {
            "name": loaded.path.name,
            "path": str(loaded.path),
            "type": loaded.type,
            "characters": len(loaded.text),
            "used_chunks": used_chunks,
            "truncated": truncated,
            "metadata": loaded.metadata,
            "model_errors": model_errors,
        },
        "sources": [
            {
                "id": f"upload:{loaded.path.name}",
                "document_name": loaded.path.name,
                "path": str(loaded.path),
                "score": 1.0,
                "snippet": loaded.text[:700],
                "paragraph": "",
                "type": loaded.type,
            }
        ],
        "index": index_result,
    }


@router.post("/chat")
def chat_once(request: Request, body: ChatRequest) -> dict[str, Any]:
    ctx = request.app.state.context
    query, think = _strip_think(body.message)
    chunks = _safe_search(ctx, query)
    operational_context, operational_sources = _build_operational_context(ctx, query)
    all_sources = source_chunks_to_json(chunks) + operational_sources
    if not chunks and not operational_context:
        answer = "Erre nem talaltam biztos forrast az indexelt dokumentumok kozott. Tudok javaslatot adni, de azt javaslatkent kezelem, nem tenykent."
        return {"answer": answer, "sources": all_sources, "model": None}
    model = choose_model(ctx.db.get_settings(), ctx.llm)
    try:
        answer = ctx.llm.chat(_messages(query, chunks, operational_context), model=model, stream=False, think=think)
    except OllamaChatError:
        fallback = ctx.db.get_settings().get("fallback_model", FALLBACK_MODEL)
        answer = ctx.llm.chat(_messages(query, chunks, operational_context), model=fallback, stream=False, think=think)
        model = fallback
    return {"answer": answer, "sources": all_sources, "model": model}


@router.post("/chat/stream")
def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    ctx = request.app.state.context
    query, think = _strip_think(body.message)

    def generate() -> Iterable[str]:
        chunks = _safe_search(ctx, query)
        operational_context, operational_sources = _build_operational_context(ctx, query)
        sources_json = source_chunks_to_json(chunks) + operational_sources
        yield _json_line({"type": "meta", "sources": sources_json})
        if not chunks and not operational_context:
            yield _json_line(
                {
                    "type": "token",
                    "text": "Erre nem talaltam biztos forrast az indexelt dokumentumok kozott. Tudok javaslatot adni, de azt javaslatkent kezelem, nem tenykent.",
                }
            )
            yield _json_line({"type": "done", "model": None})
            return

        settings = ctx.db.get_settings()
        model = choose_model(settings, ctx.llm)
        messages = _messages(query, chunks, operational_context)
        try:
            stream = ctx.llm.chat(messages, model=model, stream=True, think=think)
            for token in stream:
                yield _json_line({"type": "token", "text": token})
            yield _json_line({"type": "done", "model": model})
        except Exception as exc:
            fallback = settings.get("fallback_model", FALLBACK_MODEL)
            if fallback != model:
                try:
                    stream = ctx.llm.chat(messages, model=fallback, stream=True, think=think)
                    for token in stream:
                        yield _json_line({"type": "token", "text": token})
                    yield _json_line({"type": "done", "model": fallback})
                    return
                except Exception as fallback_exc:
                    exc = fallback_exc
            yield _json_line({"type": "error", "error": str(exc)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def _safe_search(ctx: Any, query: str) -> list[SourceChunk]:
    try:
        return ctx.kb.search(query, limit=5, min_score=0.65)
    except Exception as exc:
        ctx.db.log("warning", "RAG kereses sikertelen", {"error": str(exc)})
        return []


def file_model_candidates(settings: dict[str, str], ctx: Any) -> list[str]:
    candidates = [
        settings.get("file_model", ""),
        "qwen2.5:7b-instruct",
        "qwen3:4b",
        "gemma3:4b",
        settings.get("fallback_model", ""),
        settings.get("chat_model", ""),
        "qwen3:14b",
        "qwen2.5:14b",
    ]
    unique: list[str] = []
    for candidate in candidates:
        model = (candidate or "").strip()
        if model and model not in unique:
            unique.append(model)
    available: list[str] = []
    for model in unique:
        if ctx.llm.model_available(model):
            available.append(model)
    return available or unique


def choose_file_model(settings: dict[str, str], ctx: Any) -> str:
    models = file_model_candidates(settings, ctx)
    return models[0] if models else settings.get("chat_model", "")


def _save_chat_file(file: UploadFile) -> Path:
    filename = Path(file.filename or "feltoltott_dokumentum").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Nem tamogatott fajltipus: {suffix}")
    target_dir = DOCUMENTS_DIR / "chat_uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = target_dir / f"{target.stem}_{stamp}{target.suffix}"
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return target


def _document_context_for_instruction(loaded: LoadedDocument, instruction: str, max_chars: int = 14000) -> tuple[str, int, bool]:
    text = loaded.text.strip()
    if len(text) <= max_chars:
        return text, 1, False

    chunks = chunk_text(text, chunk_size=1400, overlap=120)
    if not chunks:
        return text[:max_chars], 1, True

    keywords = _keywords(instruction)
    scored: list[tuple[int, int]] = []
    for chunk in chunks:
        lowered = chunk.text.lower()
        score = sum(1 for keyword in keywords if keyword in lowered)
        scored.append((score, chunk.index))

    selected_indexes: set[int] = {0}
    if _looks_like_rewrite_instruction(instruction):
        selected_indexes.update(range(min(8, len(chunks))))
    else:
        for score, index in sorted(scored, key=lambda item: (-item[0], item[1])):
            if score <= 0 and len(selected_indexes) >= 5:
                break
            selected_indexes.add(index)
            if len(selected_indexes) >= 10:
                break

    rendered: list[str] = []
    total = 0
    for index in sorted(selected_indexes):
        chunk = chunks[index]
        block = f"[reszlet {index + 1}/{len(chunks)}]\n{chunk.text.strip()}"
        if total + len(block) > max_chars:
            break
        rendered.append(block)
        total += len(block)

    return "\n\n".join(rendered), len(rendered), True


def _file_messages(loaded: LoadedDocument, instruction: str, document_context: str, *, truncated: bool) -> list[dict[str, str]]:
    truncated_note = (
        "A fajl hosszu, ezert a modell csak a legfontosabb/relevans reszleteket kapta meg."
        if truncated
        else "A modell a teljes kinyert fajlszoveget megkapta."
    )
    user_message = f"""Felhasznaloi utasitas:
{instruction}

Fajl:
- nev: {loaded.path.name}
- tipus: {loaded.type}
- karakterek: {len(loaded.text)}
- feldolgozas: {truncated_note}

Fajl tartalma:
{document_context}
"""
    return [{"role": "system", "content": FILE_CHAT_SYSTEM_PROMPT}, {"role": "user", "content": user_message}]


def _keywords(text: str) -> set[str]:
    stopwords = {
        "hogy",
        "amit",
        "mert",
        "vagy",
        "ezt",
        "azt",
        "egy",
        "meg",
        "kell",
        "kerlek",
        "kérlek",
        "fajlt",
        "fájlt",
        "dokumentumot",
    }
    return {word for word in re.findall(r"[\wÁÉÍÓÖŐÚÜŰáéíóöőúüű-]{4,}", text.lower()) if word not in stopwords}


def _looks_like_rewrite_instruction(instruction: str) -> bool:
    lowered = instruction.lower()
    markers = ("javit", "javíts", "korrig", "atir", "átír", "fogalmaz", "hivatalos", "helyesiras", "helyesírás")
    return any(marker in lowered for marker in markers)
