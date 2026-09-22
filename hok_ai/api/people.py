from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


router = APIRouter(tags=["people"])


class PersonPayload(BaseModel):
    name: str
    role: str = ""
    committee: str = ""
    email: str = ""
    phone: str = ""
    aliases: list[str] | str = []
    notes: str = ""


class PersonUpdate(BaseModel):
    name: str | None = None
    role: str | None = None
    committee: str | None = None
    email: str | None = None
    phone: str | None = None
    aliases: list[str] | str | None = None
    notes: str | None = None


@router.get("/people")
def list_people(request: Request, q: str = "") -> dict[str, Any]:
    ctx = request.app.state.context
    return {"people": ctx.db.list_people(keyword=q)}


@router.post("/people")
def create_person(request: Request, payload: PersonPayload) -> dict[str, Any]:
    ctx = request.app.state.context
    try:
        return {"person": ctx.db.create_person(payload.model_dump())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/people/{person_id}")
def update_person(request: Request, person_id: str, payload: PersonUpdate) -> dict[str, Any]:
    ctx = request.app.state.context
    data = {key: value for key, value in payload.model_dump().items() if value is not None}
    try:
        return {"person": ctx.db.update_person(person_id, data)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Szemely nem talalhato.") from exc


@router.delete("/people/{person_id}")
def delete_person(request: Request, person_id: str) -> dict[str, Any]:
    ctx = request.app.state.context
    deleted = ctx.db.delete_person(person_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Szemely nem talalhato.")
    return {"deleted": True}
