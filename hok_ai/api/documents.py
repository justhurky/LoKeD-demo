from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from hok_ai.core.db import DOCUMENTS_DIR, REGULATIONS_DIR
from hok_ai.core.document_loader import SUPPORTED_EXTENSIONS
from hok_ai.core.knowledge_base import source_chunks_to_json


router = APIRouter(tags=["documents"])


@router.get("/documents")
def list_documents(request: Request, keyword: str = "", type: str = "") -> dict[str, Any]:
    ctx = request.app.state.context
    return {"documents": ctx.kb.documents(keyword=keyword, doc_type=type), "stats": ctx.kb.stats()}


@router.post("/documents/index-path")
async def index_path(request: Request) -> dict[str, Any]:
    ctx = request.app.state.context
    payload = await request.json()
    raw_path = payload.get("path", "")
    force = bool(payload.get("force", False))
    if not raw_path:
        raise HTTPException(status_code=400, detail="Hianyzik az indexelendo utvonal.")
    try:
        return ctx.kb.index_path(Path(raw_path), force=force)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/documents/upload")
def upload_document(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    ctx = request.app.state.context
    filename = Path(file.filename or "dokumentum").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Nem tamogatott fajltipus: {suffix}")
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    target = DOCUMENTS_DIR / filename
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = DOCUMENTS_DIR / f"{target.stem}_{stamp}{target.suffix}"
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        result = ctx.kb.index_file(target, force=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"uploaded": str(target), "index": result}


@router.get("/documents/search")
def search_documents(request: Request, q: str, limit: int = 5) -> dict[str, Any]:
    ctx = request.app.state.context
    try:
        hits = ctx.kb.search(q, limit=limit, min_score=0.0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"hits": source_chunks_to_json(hits)}


@router.post("/documents/index-regulations")
def index_regulations(request: Request) -> dict[str, Any]:
    ctx = request.app.state.context
    REGULATIONS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return ctx.kb.index_regulations()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
