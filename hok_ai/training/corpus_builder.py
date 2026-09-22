from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from hok_ai.core.document_loader import SUPPORTED_EXTENSIONS, chunk_text, classify_document, load_document


DEFAULT_OUTPUT_DIR = Path("hok_ai/data/training/ehok_corpus")


@dataclass(slots=True)
class CorpusStats:
    source: str
    source_kind: str
    started_at: str
    finished_at: str = ""
    total_files: int = 0
    supported_files: int = 0
    unsupported_files: int = 0
    extracted_documents: int = 0
    chunks: int = 0
    errors: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(raw: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(raw)
    return digest.hexdigest()


def safe_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


def repair_mojibake(text: str) -> str:
    if not any(marker in text for marker in ("Ã", "Å", "Â")):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text
    return repaired if repaired else text


def iter_folder_files(root: Path) -> Iterable[tuple[str, Path, int]]:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        yield str(path), path, size


def iter_zip_files(zip_path: Path) -> Iterable[zipfile.ZipInfo]:
    with zipfile.ZipFile(zip_path) as archive:
        for info in sorted((item for item in archive.infolist() if not item.is_dir()), key=lambda item: item.filename):
            yield info


def write_jsonl(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def build_corpus_from_folder(source: Path, output_dir: Path, *, include_full_text: bool = True) -> CorpusStats:
    stats = CorpusStats(source=str(source), source_kind="folder", started_at=utc_now())
    with open_outputs(output_dir) as outputs:
        for display_path, path, size in iter_folder_files(source):
            stats.total_files += 1
            process_file_path(display_path, path, size, stats, outputs, include_full_text=include_full_text)
    stats.finished_at = utc_now()
    write_summary(output_dir, stats)
    return stats


def build_corpus_from_zip(source: Path, output_dir: Path, *, include_full_text: bool = True) -> CorpusStats:
    stats = CorpusStats(source=str(source), source_kind="zip", started_at=utc_now())
    with open_outputs(output_dir) as outputs, zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory(
        prefix="hok_ai_zip_"
    ) as temp_dir:
        temp_root = Path(temp_dir)
        for info in sorted((item for item in archive.infolist() if not item.is_dir()), key=lambda item: item.filename):
            stats.total_files += 1
            entry_name = repair_mojibake(info.filename)
            suffix = Path(entry_name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                stats.unsupported_files += 1
                write_manifest(outputs, entry_name, suffix, info.file_size, "unsupported")
                continue
            temp_path = temp_root / f"{safe_id(entry_name)}{suffix}"
            try:
                with archive.open(info) as source_file, temp_path.open("wb") as target_file:
                    shutil.copyfileobj(source_file, target_file)
                process_file_path(entry_name, temp_path, info.file_size, stats, outputs, include_full_text=include_full_text)
            finally:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
    stats.finished_at = utc_now()
    write_summary(output_dir, stats)
    return stats


def process_file_path(
    display_path: str,
    path: Path,
    size: int,
    stats: CorpusStats,
    outputs: dict[str, object],
    *,
    include_full_text: bool,
) -> None:
    suffix = Path(display_path).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        stats.unsupported_files += 1
        write_manifest(outputs, display_path, suffix, size, "unsupported")
        return

    stats.supported_files += 1
    try:
        loaded = load_document(path)
        chunks = chunk_text(loaded.text)
        doc_id = safe_id(f"{display_path}:{loaded.metadata.get('file_hash', '')}")
        classification = classify_document(display_path)
        doc_payload = {
            "id": doc_id,
            "path": display_path,
            "name": Path(display_path).name,
            "extension": suffix,
            "size_bytes": size,
            "sha256": loaded.metadata.get("file_hash", ""),
            "type": classification["type"],
            "is_regulation": classification["is_regulation"],
            "regulation_name": classification["regulation_name"],
            "metadata": loaded.metadata,
            "text_length": len(loaded.text),
            "chunks": len(chunks),
        }
        if include_full_text:
            doc_payload["text"] = loaded.text
        write_jsonl(outputs["documents"], doc_payload)
        for item in chunks:
            write_jsonl(
                outputs["chunks"],
                {
                    "id": f"{doc_id}:{item.index}",
                    "document_id": doc_id,
                    "path": display_path,
                    "name": Path(display_path).name,
                    "type": classification["type"],
                    "chunk_index": item.index,
                    "paragraph_hint": item.paragraph_hint,
                    "text": item.text,
                },
            )
        write_manifest(outputs, display_path, suffix, size, "extracted", doc_id=doc_id, chunks=len(chunks))
        stats.extracted_documents += 1
        stats.chunks += len(chunks)
    except Exception as exc:
        stats.errors += 1
        write_manifest(outputs, display_path, suffix, size, "error", error=str(exc))
        write_jsonl(
            outputs["errors"],
            {
                "path": display_path,
                "extension": suffix,
                "size_bytes": size,
                "error": str(exc),
            },
        )


def write_manifest(
    outputs: dict[str, object],
    path: str,
    extension: str,
    size: int,
    status: str,
    *,
    doc_id: str = "",
    chunks: int = 0,
    error: str = "",
) -> None:
    write_jsonl(
        outputs["manifest"],
        {
            "path": path,
            "name": Path(path).name,
            "extension": extension,
            "size_bytes": size,
            "status": status,
            "document_id": doc_id,
            "chunks": chunks,
            "error": error,
        },
    )


class open_outputs:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.handles: dict[str, object] = {}

    def __enter__(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("documents", "chunks", "manifest", "errors"):
            self.handles[name] = (self.output_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")
        return self.handles

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles.values():
            handle.close()


def write_summary(output_dir: Path, stats: CorpusStats) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")


def build_domain_train_jsonl(chunks_path: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text", "").strip()
            if not text:
                continue
            target.write(
                json.dumps(
                    {
                        "text": (
                            "HÖK/EHÖK domain dokumentumrészlet.\n"
                            f"Forrás: {item.get('path', '')}\n"
                            f"Típus: {item.get('type', '')}\n\n"
                            f"{text}"
                        )
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HÖK AI dokumentumkorpusz JSON/JSONL építő")
    parser.add_argument("--source", required=True, help="Forrásmappa vagy .zip fájl.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Kimeneti mappa.")
    parser.add_argument("--no-full-text", action="store_true", help="documents.jsonl ne tartalmazza a teljes szöveget.")
    parser.add_argument("--domain-train-output", default="", help="Opcionális continued-pretrain JSONL kimenet.")
    return parser.parse_args()


def resolve_source_path(raw_source: str) -> Path:
    source = Path(raw_source)
    if source.exists():
        return source
    parent = source.parent if str(source.parent) not in ("", ".") else Path(".")
    if parent.exists():
        signature = source.name
        for noisy in ("Ö", "ö", "Ő", "ő", "Ü", "ü", "Ű", "ű", "É", "é", "Á", "á", "Í", "í", "Ó", "ó"):
            signature = signature.replace(noisy, "*")
        signature = signature.replace("?", "*")
        matches = list(parent.glob(signature))
        if matches:
            return matches[0]
        compact = "".join(char.lower() for char in source.name if char.isascii() and char.isalnum())
        for candidate in parent.iterdir():
            candidate_compact = "".join(char.lower() for char in candidate.name if char.isascii() and char.isalnum())
            if compact and compact in candidate_compact:
                return candidate
    return source


def main() -> None:
    args = parse_args()
    source = resolve_source_path(args.source)
    if not source.exists():
        raise SystemExit(f"Nem található a forrás: {args.source}")
    output_dir = Path(args.output_dir)
    if source.is_dir():
        stats = build_corpus_from_folder(source, output_dir, include_full_text=not args.no_full_text)
    elif source.suffix.lower() == ".zip":
        stats = build_corpus_from_zip(source, output_dir, include_full_text=not args.no_full_text)
    else:
        raise SystemExit("A forrás csak mappa vagy .zip lehet.")
    if args.domain_train_output:
        records = build_domain_train_jsonl(output_dir / "chunks.jsonl", Path(args.domain_train_output))
        print(f"Domain training JSONL: {args.domain_train_output} | rekordok: {records}")
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
