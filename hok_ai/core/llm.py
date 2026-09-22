from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PRIMARY_MODEL = "qwen3:14b"
FALLBACK_MODEL = "qwen2.5:14b"
DOMAIN_ADAPTER_MODEL = "hok-ai-domain-4b-peft"
DOMAIN_ADAPTER_BASE_MODEL = "Qwen/Qwen3-4B"
DOMAIN_ADAPTER_URL = "http://127.0.0.1:7843"
DOMAIN_ADAPTER_PATH = str(Path(__file__).resolve().parents[1] / "data" / "models" / DOMAIN_ADAPTER_MODEL)


AGENT_PROMPT = """Te egy HOK (Hallgatoi Onkormanyzat) es EHOK asszisztens vagy.

A feladatod, hogy segitsd a hallgatoi onkormanyzati adminisztraciot,
dokumentumkezelest, feladatkovetest, ulesjegyzetelest, dontesrogzitest,
hataridokezelest, szabalyzatok ertelmezeset es javaslatok kesziteset.

Mindig magyarul valaszolj. A valaszaidat a megadott dokumentumreszletekre,
forrasokra es indexelt tudastar alapjan epitsd fel. Ne hasznalj kulso,
nem indexelt tudast. Ha valamit nem talalsz a forrasok kozott, mondd meg
oszinten.

A hivatalos forrasok kulonosen fontosak: TVSZ, TJSZ, HOK/EHOK alapszabaly,
indexelt jegyzokonyvek, indexelt feladatlistak es indexelt dokumentumok.

Ne mutass privat, reszletes, token-szintu gondolatmenetet. Ha a felhasznalo
/think modot ker, adj rovid, magas szintu, ellenorizheto "Gondolat:" blokkot.

Valaszformai kovetelmenyek:
- legyen strukturalt,
- legyen magyar,
- legyen tomor, de hasznos,
- ne allits biztoskent olyat, amit nem tamaszt forras,
- kulonitsd el a tenyt, forrast, ertelmezest, javaslatot es bizonytalan pontot.

Ha nincs forras:
"Erre nem talaltam biztos forrast az indexelt dokumentumok kozott. Tudok
javaslatot adni, de azt javaslatkent kezelem, nem tenykent."
"""


class OllamaChatError(RuntimeError):
    pass


@dataclass(slots=True)
class OllamaClient:
    host: str = "http://127.0.0.1:11434"
    timeout: int = 120
    adapter_enabled: bool = True
    adapter_url: str = DOMAIN_ADAPTER_URL
    adapter_model: str = DOMAIN_ADAPTER_MODEL
    adapter_path: str = DOMAIN_ADAPTER_PATH

    def _url(self, path: str) -> str:
        return f"{self.host.rstrip('/')}{path}"

    def configure_from_settings(self, settings: dict[str, str]) -> None:
        self.host = settings.get("ollama_host", self.host)
        self.adapter_enabled = _truthy(settings.get("peft_adapter_enabled", "true"))
        self.adapter_url = settings.get("peft_adapter_url", self.adapter_url).strip() or DOMAIN_ADAPTER_URL
        self.adapter_model = settings.get("custom_4b_model") or settings.get("peft_adapter_model") or self.adapter_model
        self.adapter_path = settings.get("peft_adapter_path", self.adapter_path).strip() or DOMAIN_ADAPTER_PATH

    def tags(self, timeout: int = 5) -> list[str]:
        names = self.ollama_tags(timeout=timeout)
        if self.adapter_status(timeout=min(timeout, 2)).get("online") and self.adapter_model not in names:
            names.append(self.adapter_model)
        return names

    def ollama_tags(self, timeout: int = 5) -> list[str]:
        try:
            with urllib.request.urlopen(self._url("/api/tags"), timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        return [item.get("name", "") for item in data.get("models", []) if item.get("name")]

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(self._url("/api/tags"), timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def model_available(self, model: str) -> bool:
        if self._is_adapter_model(model):
            return bool(self.adapter_status(timeout=2).get("online"))
        names = self.tags()
        return any(name == model or name.startswith(f"{model}:") for name in names)

    def adapter_status(self, timeout: int = 2) -> dict[str, Any]:
        configured_path = Path(self.adapter_path)
        status: dict[str, Any] = {
            "enabled": self.adapter_enabled,
            "model": self.adapter_model,
            "base_model": DOMAIN_ADAPTER_BASE_MODEL,
            "url": self.adapter_url,
            "path": self.adapter_path,
            "path_exists": configured_path.exists(),
            "online": False,
        }
        if not self.adapter_enabled:
            return status
        try:
            with urllib.request.urlopen(_join_url(self.adapter_url, "/health"), timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            status["error"] = str(exc)
            return status
        status.update({key: value for key, value in data.items() if key not in {"path"}})
        status["online"] = response.status == 200 and bool(data.get("ok", True))
        return status

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = PRIMARY_MODEL,
        *,
        stream: bool = False,
        think: bool = False,
        response_format: str | None = None,
        timeout: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> str | Iterable[str]:
        if self._is_adapter_model(model):
            return self._adapter_chat(
                messages,
                model=model,
                stream=stream,
                think=think,
                response_format=response_format,
                timeout=timeout or self.timeout,
                options=options,
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options or {"temperature": 0.2, "num_predict": 1400},
        }
        if response_format:
            payload["format"] = response_format
        payload["think"] = bool(think)

        if stream:
            return self._stream_chat(payload, timeout=timeout or self.timeout)
        return self._chat_once(payload, timeout=timeout or self.timeout)

    def _chat_once(self, payload: dict[str, Any], timeout: int) -> str:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._url("/api/chat"),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaChatError(f"Ollama chat hiba: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaChatError("Az Ollama ervenytelen JSON chat valaszt adott.") from exc
        return (body.get("message") or {}).get("content", "")

    def _stream_chat(self, payload: dict[str, Any], timeout: int) -> Iterable[str]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._url("/api/chat"),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = (item.get("message") or {}).get("content") or ""
                    if content:
                        yield content
                    if item.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise OllamaChatError(f"Ollama streaming hiba: {exc}") from exc

    def _is_adapter_model(self, model: str) -> bool:
        normalized = (model or "").strip()
        if not normalized:
            return False
        aliases = {
            self.adapter_model,
            DOMAIN_ADAPTER_MODEL,
            f"peft:{self.adapter_model}",
            f"peft:{DOMAIN_ADAPTER_MODEL}",
            "hok-ai-jegyzokonyv-4b",
        }
        return normalized in aliases

    def _adapter_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        stream: bool,
        think: bool,
        response_format: str | None,
        timeout: int,
        options: dict[str, Any] | None,
    ) -> str | Iterable[str]:
        if not self.adapter_enabled:
            raise OllamaChatError("A PEFT adapter ki van kapcsolva.")
        payload = {
            "model": model,
            "messages": messages,
            "think": bool(think),
            "format": response_format,
            "options": options or {"temperature": 0.2, "num_predict": 800},
        }
        request = urllib.request.Request(
            _join_url(self.adapter_url, "/chat"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaChatError(f"PEFT adapter hiba: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaChatError("A PEFT adapter ervenytelen JSON valaszt adott.") from exc
        content = (body.get("message") or {}).get("content") or body.get("content") or ""
        if body.get("error"):
            raise OllamaChatError(str(body["error"]))
        if stream:
            return _single_chunk_stream(str(content))
        return str(content)


def choose_model(settings: dict[str, str], llm: OllamaClient) -> str:
    primary = settings.get("chat_model") or PRIMARY_MODEL
    fallback = settings.get("fallback_model") or FALLBACK_MODEL
    if llm.model_available(primary):
        return primary
    if llm.model_available(fallback):
        return fallback
    return primary


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "igen", "yes", "on", "bekapcsolva"}


def _single_chunk_stream(text: str) -> Iterable[str]:
    if text:
        yield text
