from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from hok_ai.core.db import AUDIO_DIR, DOCUMENTS_DIR
from hok_ai.core.document_loader import load_document
from hok_ai.core.llm import choose_model
from hok_ai.core.meeting_parser import MeetingParser
from hok_ai.core.minutes_generator import MinutesGenerator
from hok_ai.core.whisper_engine import WhisperEngine


router = APIRouter(tags=["meeting"])


class MeetingAnalyzeRequest(BaseModel):
    text: str
    title: str = ""
    save: bool = True


class MinutesRequest(BaseModel):
    text: str
    title: str = ""


@router.post("/meeting/analyze")
def analyze_meeting(request: Request, payload: MeetingAnalyzeRequest) -> dict[str, Any]:
    ctx = request.app.state.context
    settings = ctx.db.get_settings()
    model = ""
    analysis: dict[str, Any] | None = None
    errors: list[str] = []
    for candidate in candidate_meeting_models(settings, ctx):
        parser = MeetingParser(ctx.llm, model=candidate)
        try:
            analysis = parser.analyze(payload.text)
            model = candidate
            break
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    if analysis is None:
        raise HTTPException(status_code=502, detail="; ".join(errors) or "A meeting elemzes sikertelen.") from None

    meeting = None
    decisions = []
    if payload.save:
        title = payload.title or f"Meeting elemzes {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        meeting = ctx.db.create_meeting(
            {
                "title": title,
                "raw_text": payload.text,
                "summary": _summary_from_analysis(analysis),
                "analysis": analysis,
                "chaos_index": analysis.get("kaosz_index", 0),
            }
        )
        for decision in analysis.get("dontesek", []):
            text = decision.get("szoveg", "")
            if not text:
                continue
            decisions.append(
                ctx.db.create_decision(
                    {
                        "text": text,
                        "date": decision.get("datum", ""),
                        "vote": decision.get("szavazat", ""),
                        "source": "meeting",
                        "meeting_id": meeting["id"],
                    }
                )
            )
        _index_meeting_text(ctx, meeting["id"], title, payload.text)

    return {"analysis": analysis, "meeting": meeting, "decisions": decisions, "model": model}


@router.post("/meeting/minutes")
def generate_minutes(request: Request, payload: MinutesRequest) -> dict[str, Any]:
    ctx = request.app.state.context
    settings = ctx.db.get_settings()
    errors: list[str] = []
    for model in candidate_minutes_models(settings, ctx):
        generator = MinutesGenerator(ctx.llm, model=model)
        try:
            minutes = generator.generate(payload.text)
            return {"minutes": minutes, "model": model}
        except Exception as exc:
            errors.append(f"{model}: {exc}")
    raise HTTPException(status_code=502, detail="; ".join(errors) or "A jegyzokonyv generalas sikertelen.") from None


@router.post("/meeting/load-file")
def load_meeting_file(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "meeting.txt").name
    target = DOCUMENTS_DIR / filename
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(file.file.read())
    try:
        loaded = load_document(target)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"text": loaded.text, "metadata": loaded.metadata, "path": str(target)}


@router.post("/meeting/transcribe")
def transcribe_audio(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    ctx = request.app.state.context
    settings = ctx.db.get_settings()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    target = AUDIO_DIR / f"hangjegyzet_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    with target.open("wb") as handle:
        handle.write(file.file.read())
    try:
        engine = WhisperEngine(model_size=settings.get("whisper_model", "tiny"))
        return {"path": str(target), "transcript": engine.transcribe(target)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _summary_from_analysis(analysis: dict[str, Any]) -> str:
    return (
        f"Feladatok: {len(analysis.get('feladatok', []))}; "
        f" dontesek: {len(analysis.get('dontesek', []))}; "
        f" kerdesek: {len(analysis.get('kerdesek', []))}; "
        f" kaosz-index: {analysis.get('kaosz_index', 0)}"
    )


def _index_meeting_text(ctx: Any, meeting_id: str, title: str, text: str) -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in title)[:60]
    path = DOCUMENTS_DIR / f"{meeting_id}_{safe_title}.md"
    path.write_text(f"# {title}\n\n{text}", encoding="utf-8")
    try:
        ctx.kb.index_file(path, force=True)
    except Exception as exc:
        ctx.db.log("warning", "Meeting tudastar indexeles sikertelen", {"meeting_id": meeting_id, "error": str(exc)})


def choose_meeting_model(settings: dict[str, str], ctx: Any) -> str:
    configured = settings.get("meeting_model") or ""
    if configured and ctx.llm.model_available(configured):
        return configured
    return choose_model(settings, ctx.llm)


def choose_minutes_model(settings: dict[str, str], ctx: Any) -> str:
    configured = settings.get("minutes_model") or settings.get("custom_4b_model") or settings.get("meeting_model") or ""
    if configured and ctx.llm.model_available(configured):
        return configured
    return choose_model(settings, ctx.llm)


def candidate_meeting_models(settings: dict[str, str], ctx: Any) -> list[str]:
    return _available_model_sequence(
        ctx,
        [
            settings.get("meeting_model", ""),
            settings.get("custom_4b_model", ""),
            settings.get("chat_model", ""),
            settings.get("fallback_model", ""),
            "gemma3:4b",
            "qwen3:4b",
        ],
    )


def candidate_minutes_models(settings: dict[str, str], ctx: Any) -> list[str]:
    return _available_model_sequence(
        ctx,
        [
            settings.get("minutes_model", ""),
            settings.get("custom_4b_model", ""),
            settings.get("meeting_model", ""),
            settings.get("chat_model", ""),
            settings.get("fallback_model", ""),
            "gemma3:4b",
            "qwen3:4b",
        ],
    )


def _available_model_sequence(ctx: Any, candidates: list[str]) -> list[str]:
    available = ctx.llm.tags()
    result: list[str] = []
    for candidate in candidates:
        model = candidate.strip()
        if not model or model in result:
            continue
        if any(name == model or name.startswith(f"{model}:") for name in available):
            result.append(model)
    if not result:
        result.append(candidates[0] or "qwen3:14b")
    return result
