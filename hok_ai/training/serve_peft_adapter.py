from __future__ import annotations

import argparse
import json
import threading
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "hok-ai-domain-4b-peft"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"


@dataclass(slots=True)
class AdapterRuntime:
    adapter_dir: Path
    base_model: str
    model_name: str
    max_input_tokens: int
    max_new_tokens: int
    load_in_4bit: bool
    tokenizer: Any = None
    model: Any = None
    torch: Any = None
    cuda: bool = False
    lock: Any = field(default_factory=threading.Lock)

    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.cuda = bool(torch.cuda.is_available())
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization = None
        if self.cuda and self.load_in_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quantization,
            device_map="auto" if self.cuda else None,
            torch_dtype=torch.float16 if self.cuda else torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        base.config.use_cache = True
        self.model = PeftModel.from_pretrained(base, self.adapter_dir)
        self.model.eval()

    def health(self) -> dict[str, Any]:
        return {
            "ok": self.model is not None and self.tokenizer is not None,
            "online": True,
            "loaded": self.model is not None,
            "model": self.model_name,
            "base_model": self.base_model,
            "adapter_dir": str(self.adapter_dir),
            "cuda": self.cuda,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
        }

    def chat(self, payload: dict[str, Any]) -> str:
        if self.model is None or self.tokenizer is None or self.torch is None:
            raise RuntimeError("A PEFT adapter meg nincs betoltve.")

        messages = payload.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise ValueError("A chat payload messages listat var.")

        response_format = payload.get("format")
        if response_format == "json":
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": "A valasz kizarolag valid JSON legyen, markdown es magyarazat nelkul.",
                },
            ]

        options = payload.get("options") or {}
        temperature = float(options.get("temperature", 0.2) or 0.0)
        requested_tokens = int(options.get("num_predict", self.max_new_tokens) or self.max_new_tokens)
        max_new_tokens = max(1, min(requested_tokens, self.max_new_tokens))

        input_ids = self._encode(messages)
        if input_ids.shape[-1] > self.max_input_tokens:
            input_ids = input_ids[:, -self.max_input_tokens :]

        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = float(options.get("top_p", 0.9) or 0.9)

        with self.lock:
            with self.torch.inference_mode():
                output = self.model.generate(**generate_kwargs)

        generated = output[0][input_ids.shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _strip_thinking(text)

    def _encode(self, messages: list[dict[str, str]]) -> Any:
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            prompt = _fallback_prompt(messages)
            encoded = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
        elif isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
        else:
            input_ids = encoded
        return input_ids.to(self._device())

    def _device(self) -> Any:
        return next(self.model.parameters()).device


RUNTIME: AdapterRuntime | None = None


class AdapterHandler(BaseHTTPRequestHandler):
    server_version = "HOKAIPEFT/0.1"

    def do_GET(self) -> None:
        if self.path in {"/", "/health"}:
            self._send_json(_runtime().health())
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/chat":
            self._send_json({"error": "Not found"}, status=404)
            return
        try:
            payload = self._read_json()
            content = _runtime().chat(payload)
            self._send_json(
                {
                    "model": _runtime().model_name,
                    "message": {"role": "assistant", "content": content},
                    "content": content,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": str(exc) or repr(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HOK AI PEFT LoRA adapter HTTP worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7843)
    parser.add_argument("--adapter-dir", default="hok_ai/data/models/hok-ai-domain-4b-peft")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = AdapterRuntime(
        adapter_dir=Path(args.adapter_dir),
        base_model=args.base_model,
        model_name=args.model_name,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=not args.no_4bit,
    )
    if not runtime.adapter_dir.exists():
        raise SystemExit(f"Nincs adapter konyvtar: {runtime.adapter_dir}")
    print(f"PEFT adapter betoltese: {runtime.adapter_dir}")
    runtime.load()
    global RUNTIME
    RUNTIME = runtime
    server = ThreadingHTTPServer((args.host, args.port), AdapterHandler)
    print(f"HOK AI PEFT adapter worker: http://{args.host}:{args.port}")
    server.serve_forever()


def _runtime() -> AdapterRuntime:
    if RUNTIME is None:
        raise RuntimeError("A PEFT adapter runtime nem indult el.")
    return RUNTIME


def _fallback_prompt(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role.upper()}:\n{content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if text.startswith("<think>"):
        return text.replace("<think>", "", 1).strip()
    return text


if __name__ == "__main__":
    main()
