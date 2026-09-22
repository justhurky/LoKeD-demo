from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class WhisperEngine:
    model_size: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"

    def transcribe(self, audio_path: str | Path, *, language: str = "hu") -> dict[str, object]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"A hangfajl nem talalhato: {path}")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("A faster-whisper nincs telepitve.") from exc

        model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        segments, info = model.transcribe(str(path), language=language, vad_filter=True)
        rendered_segments = []
        for segment in segments:
            rendered_segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                }
            )
        text = " ".join(item["text"] for item in rendered_segments).strip()
        return {
            "text": text,
            "segments": rendered_segments,
            "language": getattr(info, "language", language),
            "duration": getattr(info, "duration", 0.0),
        }
