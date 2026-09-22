from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from hok_ai.core.db import EXPORT_DIR


router = APIRouter(tags=["tasks"])


class TaskPayload(BaseModel):
    title: str
    description: str = ""
    owner: str = ""
    deadline: str = ""
    priority: str = "közepes"
    status: str = "Nyitott"
    source_sentence: str = ""
    related_decision: str = ""
    related_meeting: str = ""


class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    owner: str | None = None
    deadline: str | None = None
    priority: str | None = None
    status: str | None = None
    source_sentence: str | None = None
    related_decision: str | None = None
    related_meeting: str | None = None


@router.get("/tasks")
def list_tasks(request: Request, status: str = "", owner: str = "", priority: str = "") -> dict[str, Any]:
    ctx = request.app.state.context
    return {"tasks": ctx.db.list_tasks(status=status, owner=owner, priority=priority)}


@router.post("/tasks")
def create_task(request: Request, payload: TaskPayload) -> dict[str, Any]:
    ctx = request.app.state.context
    try:
        return {"task": ctx.db.create_task(payload.model_dump())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tasks/import")
async def import_tasks(request: Request) -> dict[str, Any]:
    ctx = request.app.state.context
    payload = await request.json()
    tasks = payload.get("tasks") or payload.get("feladatok") or []
    created = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("cim") or ""
        if not title:
            continue
        created.append(
            ctx.db.create_task(
                {
                    "title": title,
                    "description": item.get("description", ""),
                    "owner": item.get("owner") or item.get("felelos") or "",
                    "deadline": item.get("deadline") or item.get("hataridő") or item.get("hatarido") or "",
                    "priority": item.get("priority") or item.get("prioritas") or "közepes",
                    "status": item.get("status", "Nyitott"),
                    "source_sentence": item.get("source_sentence") or item.get("forrasmondat") or "",
                    "related_meeting": item.get("related_meeting", ""),
                }
            )
        )
    return {"created": created, "count": len(created)}


@router.put("/tasks/{task_id}")
def update_task(request: Request, task_id: str, payload: TaskUpdate) -> dict[str, Any]:
    ctx = request.app.state.context
    data = {key: value for key, value in payload.model_dump().items() if value is not None}
    try:
        return {"task": ctx.db.update_task(task_id, data)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Feladat nem talalhato.") from exc


@router.delete("/tasks/{task_id}")
def delete_task(request: Request, task_id: str) -> dict[str, Any]:
    ctx = request.app.state.context
    deleted = ctx.db.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Feladat nem talalhato.")
    return {"deleted": True}


@router.get("/tasks/export")
def export_tasks(request: Request, format: Literal["csv", "json", "pdf"] = "csv") -> FileResponse:
    ctx = request.app.state.context
    tasks = ctx.db.list_tasks()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if format == "json":
        path = EXPORT_DIR / f"hok_ai_feladatok_{stamp}.json"
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        return FileResponse(path, filename=path.name, media_type="application/json")
    if format == "pdf":
        path = EXPORT_DIR / f"hok_ai_feladatok_{stamp}.pdf"
        _write_tasks_pdf(tasks, path)
        return FileResponse(path, filename=path.name, media_type="application/pdf")

    path = EXPORT_DIR / f"hok_ai_feladatok_{stamp}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "description",
                "owner",
                "deadline",
                "priority",
                "status",
                "source_sentence",
                "related_decision",
                "related_meeting",
                "created_at",
                "updated_at",
            ],
        )
        writer.writeheader()
        writer.writerows(tasks)
    return FileResponse(path, filename=path.name, media_type="text/csv")


def _write_tasks_pdf(tasks: list[dict[str, Any]], path: Path) -> None:
    try:
        from weasyprint import HTML
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="A PDF exporthoz telepiteni kell a weasyprint csomagot.") from exc

    rows = "\n".join(
        f"""
        <tr>
          <td>{_escape(task.get('title', ''))}</td>
          <td>{_escape(task.get('owner', ''))}</td>
          <td>{_escape(task.get('deadline', ''))}</td>
          <td>{_escape(task.get('priority', ''))}</td>
          <td>{_escape(task.get('status', ''))}</td>
        </tr>
        """
        for task in tasks
    )
    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {{ font-family: Arial, sans-serif; color: #17171c; }}
          h1 {{ font-size: 22px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #d7d7df; padding: 8px; font-size: 12px; }}
          th {{ background: #f1f1f7; text-align: left; }}
        </style>
      </head>
      <body>
        <h1>HÖK AI feladatlista</h1>
        <table>
          <thead><tr><th>Cím</th><th>Felelős</th><th>Határidő</th><th>Prioritás</th><th>Státusz</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </body>
    </html>
    """
    HTML(string=html).write_pdf(path)


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
