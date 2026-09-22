from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm import PRIMARY_MODEL, OllamaChatError, OllamaClient


MEETING_SYSTEM_PROMPT = """Te egy HOK/EHOK meetingelemzo asszisztens vagy.

A feladatod, hogy a megadott ulesjegyzetbol kizarolag valid JSON-t adj vissza.
Ne irj magyarazatot. Ne irj markdownot. Ne irj bevezetot. Csak JSON-t adj.

Nyerd ki:
- feladatokat,
- felelosoket,
- hataridoket,
- prioritast,
- donteseket,
- kerdeseket,
- blokkolokat,
- resztvevoket,
- kaosz-indexet.

Ha valami nem ismert, ures stringet vagy null erteket hasznalj, de ne talalj ki adatot.

Kotelezo JSON schema:
{
  "feladatok": [{"cim": "", "felelos": "", "hataridő": "", "prioritas": ""}],
  "dontesek": [{"szoveg": "", "datum": "", "szavazat": ""}],
  "kerdesek": [{"szoveg": "", "ki_vetette_fel": ""}],
  "blokkatok": [{"szoveg": "", "megoldas_javaslat": ""}],
  "resztvevok": [],
  "kaosz_index": 0
}
"""


def empty_meeting_analysis(error: str = "") -> dict[str, Any]:
    return {
        "feladatok": [],
        "dontesek": [],
        "kerdesek": [],
        "blokkatok": [],
        "resztvevok": [],
        "kaosz_index": 0,
        "error": error,
    }


@dataclass(slots=True)
class MeetingParser:
    llm: OllamaClient
    model: str = PRIMARY_MODEL

    def analyze(self, text: str, *, timeout: int = 120) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("Az elemzeshez meeting szoveg szukseges.")
        messages = [
            {"role": "system", "content": MEETING_SYSTEM_PROMPT},
            {"role": "user", "content": text.strip()},
        ]
        last_error = ""
        for _attempt in range(2):
            try:
                raw = self.llm.chat(
                    messages,
                    model=self.model,
                    stream=False,
                    think=False,
                    response_format="json",
                    timeout=timeout,
                    options={"temperature": 0.0, "num_predict": 1800},
                )
                return normalize_meeting_json(parse_json_response(str(raw)))
            except Exception as exc:
                last_error = str(exc)
        raise OllamaChatError(f"A meeting elemzes nem adott valid JSON-t: {last_error}")


def parse_json_response(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def normalize_meeting_json(data: dict[str, Any]) -> dict[str, Any]:
    normalized = empty_meeting_analysis()
    normalized.update({key: value for key, value in data.items() if key in normalized})
    if "blokkolok" in data and not normalized["blokkatok"]:
        normalized["blokkatok"] = data.get("blokkolok") or []
    normalized["feladatok"] = _list_of_dicts(normalized.get("feladatok"))
    normalized["dontesek"] = _list_of_dicts(normalized.get("dontesek"))
    normalized["kerdesek"] = _list_of_dicts(normalized.get("kerdesek"))
    normalized["blokkatok"] = _list_of_dicts(normalized.get("blokkatok"))
    participants = normalized.get("resztvevok") or []
    normalized["resztvevok"] = participants if isinstance(participants, list) else [str(participants)]
    try:
        normalized["kaosz_index"] = int(normalized.get("kaosz_index") or 0)
    except (TypeError, ValueError):
        normalized["kaosz_index"] = 0
    normalized.pop("error", None)
    return normalized


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
