from __future__ import annotations

import argparse
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import uvicorn

from hok_ai.api.server import AppContext, create_app
from hok_ai.core.db import DEFAULT_DB_PATH, Database, ensure_data_dirs
from hok_ai.core.embeddings import OllamaEmbeddingClient
from hok_ai.core.knowledge_base import KnowledgeBase
from hok_ai.core.llm import OllamaClient
from hok_ai.core.scheduler import DeadlineScheduler


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7842


def build_context(db_path: str | Path = DEFAULT_DB_PATH) -> AppContext:
    ensure_data_dirs()
    db = Database(Path(db_path))
    db.init()
    settings = db.get_settings()
    ollama_host = settings.get("ollama_host", "http://127.0.0.1:11434")
    llm = OllamaClient(host=ollama_host)
    llm.configure_from_settings(settings)
    embeddings = OllamaEmbeddingClient(
        host=ollama_host,
        model=settings.get("embedding_model", "nomic-embed-text"),
    )
    kb = KnowledgeBase(db=db, embedding_client=embeddings)
    return AppContext(db=db, llm=llm, embeddings=embeddings, kb=kb)


def startup_checks(context: AppContext) -> dict[str, Any]:
    status: dict[str, Any] = {"ollama": False, "embedding_model": False, "regulations_indexed": False}
    tags = context.llm.ollama_tags(timeout=5)
    status["ollama"] = bool(tags)
    if not tags:
        context.db.log("warning", "Ollama nem fut vagy nem elerheto.", {"hint": "ollama serve"})
        return status

    settings = context.db.get_settings()
    embedding_model = settings.get("embedding_model", "nomic-embed-text")
    status["embedding_model"] = any(name == embedding_model or name.startswith(f"{embedding_model}:") for name in tags)
    if not status["embedding_model"]:
        context.db.log("warning", "Embedding modell nem elerheto.", {"model": embedding_model})

    try:
        context.kb.index_regulations()
        status["regulations_indexed"] = True
    except Exception as exc:
        context.db.log("warning", "Szabalyzatok automatikus indexelese sikertelen.", {"error": str(exc)})
    return status


def start_api_server(app: Any, host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="hok-ai-api")
    thread.start()
    _wait_for_server(host, port)
    return server


def _wait_for_server(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    if last_error:
        raise RuntimeError(f"A HOK AI backend nem indult el: {last_error}") from last_error


def open_window(url: str) -> None:
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError("A PyWebView nincs telepitve. Futtasd: pip install -r requirements.txt") from exc
    webview.create_window("HÖK AI", url, width=1280, height=800, min_size=(1180, 760))
    webview.start()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HOK AI lokalis adminisztracios asszisztens")
    parser.add_argument("--server-only", action="store_true", help="Csak a FastAPI szervert inditja, ablak nelkul.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    context = build_context(args.db)
    startup_checks(context)
    scheduler = DeadlineScheduler(context.db, enabled=context.db.get_settings().get("notifications_enabled") == "true")
    scheduler.start()
    app = create_app(context)
    server = start_api_server(app, args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"HOK AI backend: {url}")
    if args.server_only:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.should_exit = True
            scheduler.stop()
        return
    try:
        open_window(url)
    finally:
        server.should_exit = True
        scheduler.stop()


if __name__ == "__main__":
    main()
