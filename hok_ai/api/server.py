from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from hok_ai.core.db import DEFAULT_DB_PATH, Database
from hok_ai.core.embeddings import OllamaEmbeddingClient
from hok_ai.core.knowledge_base import KnowledgeBase
from hok_ai.core.llm import DOMAIN_ADAPTER_MODEL, DOMAIN_ADAPTER_PATH, DOMAIN_ADAPTER_URL, OllamaClient

from . import chat, documents, meeting, people, tasks


FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


@dataclass(slots=True)
class AppContext:
    db: Database
    llm: OllamaClient
    embeddings: OllamaEmbeddingClient
    kb: KnowledgeBase


def build_default_context(db_path: str | Path = DEFAULT_DB_PATH) -> AppContext:
    db = Database(Path(db_path))
    db.init()
    settings = db.get_settings()
    llm = OllamaClient(host=settings.get("ollama_host", "http://127.0.0.1:11434"))
    llm.configure_from_settings(settings)
    embeddings = OllamaEmbeddingClient(
        host=settings.get("ollama_host", "http://127.0.0.1:11434"),
        model=settings.get("embedding_model", "nomic-embed-text"),
    )
    kb = KnowledgeBase(db=db, embedding_client=embeddings)
    return AppContext(db=db, llm=llm, embeddings=embeddings, kb=kb)


def get_context(request: Request) -> AppContext:
    return request.app.state.context


def create_app(context: AppContext | None = None) -> FastAPI:
    context = context or build_default_context()
    app = FastAPI(title="HOK AI", version="0.2.0")
    app.state.context = context

    app.include_router(chat.router, prefix="/api")
    app.include_router(tasks.router, prefix="/api")
    app.include_router(meeting.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(people.router, prefix="/api")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return FRONTEND_PATH.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        ctx = get_context(request)
        settings = ctx.db.get_settings()
        ollama_tags = ctx.llm.ollama_tags(timeout=2)
        embedding_model = settings.get("embedding_model", "nomic-embed-text")
        chat_model = settings.get("chat_model", "qwen3:14b")
        fallback_model = settings.get("fallback_model", "qwen2.5:14b")
        custom_4b_model = settings.get("custom_4b_model", DOMAIN_ADAPTER_MODEL)
        minutes_model = settings.get("minutes_model", custom_4b_model)
        file_model = settings.get("file_model", "qwen2.5:7b-instruct")
        adapter_status = ctx.llm.adapter_status(timeout=1)
        tags = list(ollama_tags)
        if adapter_status.get("online") and custom_4b_model not in tags:
            tags.append(custom_4b_model)
        return {
            "ok": True,
            "ollama": bool(ollama_tags),
            "ollama_host": settings.get("ollama_host"),
            "models": tags,
            "peft_adapter": adapter_status,
            "chat_model": chat_model,
            "fallback_model": fallback_model,
            "custom_4b_model": custom_4b_model,
            "minutes_model": minutes_model,
            "file_model": file_model,
            "chat_model_available": _model_in_tags(chat_model, tags),
            "fallback_model_available": _model_in_tags(fallback_model, tags),
            "custom_4b_model_available": _model_in_tags(custom_4b_model, tags),
            "minutes_model_available": _model_in_tags(minutes_model, tags),
            "file_model_available": _model_in_tags(file_model, tags),
            "embedding_model": embedding_model,
            "embedding_model_available": any(name == embedding_model or name.startswith(f"{embedding_model}:") for name in tags),
            "knowledge": ctx.kb.stats(),
        }

    @app.get("/api/settings")
    def get_settings(request: Request) -> dict[str, str]:
        return get_context(request).db.get_settings()

    @app.put("/api/settings")
    async def update_settings(request: Request) -> dict[str, str]:
        payload = await request.json()
        ctx = get_context(request)
        settings = ctx.db.set_settings(payload)
        ctx.llm.configure_from_settings(settings)
        ctx.embeddings.host = settings.get("ollama_host", ctx.embeddings.host)
        ctx.embeddings.model = settings.get("embedding_model", ctx.embeddings.model)
        return settings

    @app.get("/api/ollama/tags")
    def ollama_tags(request: Request) -> dict[str, Any]:
        tags = get_context(request).llm.tags(timeout=5)
        return {"models": tags}

    @app.get("/api/peft-adapter/health")
    def peft_adapter_health(request: Request) -> dict[str, Any]:
        status = get_context(request).llm.adapter_status(timeout=3)
        status.setdefault("model", DOMAIN_ADAPTER_MODEL)
        status.setdefault("url", DOMAIN_ADAPTER_URL)
        status.setdefault("path", str(DOMAIN_ADAPTER_PATH))
        return status

    @app.get("/api/dashboard")
    def dashboard(request: Request) -> dict[str, Any]:
        return get_context(request).db.dashboard()

    return app


def _model_in_tags(model: str, tags: list[str]) -> bool:
    return any(name == model or name.startswith(f"{model}:") for name in tags)
