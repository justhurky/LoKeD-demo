from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from hok_ai.training.prompts import DATASET_INSTRUCTION


REQUIRED_SECTIONS = (
    "1. Rövid Összefoglaló",
    "2. Hivatalos Jegyzőkönyv",
    "3. Feladatnapló",
    "4. Stratégiai Javaslatok",
)


@dataclass(slots=True)
class DatasetIssue:
    file: str
    message: str


def build_records(raw_dir: Path, gold_dir: Path, *, chat_format: bool = False) -> tuple[list[dict], list[DatasetIssue]]:
    records: list[dict] = []
    issues: list[DatasetIssue] = []
    raw_files = sorted(path for path in raw_dir.glob("*.txt") if path.is_file())
    for raw_path in raw_files:
        gold_path = gold_dir / f"{raw_path.stem}.md"
        if not gold_path.exists():
            gold_path = gold_dir / f"{raw_path.stem}.txt"
        if not gold_path.exists():
            issues.append(DatasetIssue(str(raw_path), "Hianyzik a parositott gold output (.md vagy .txt)."))
            continue
        transcript = raw_path.read_text(encoding="utf-8").strip()
        output = gold_path.read_text(encoding="utf-8").strip()
        output_issues = validate_output(output)
        issues.extend(DatasetIssue(str(gold_path), issue) for issue in output_issues)
        if not transcript or not output:
            issues.append(DatasetIssue(str(raw_path), "Ures input vagy output."))
            continue
        if chat_format:
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": DATASET_INSTRUCTION},
                        {"role": "user", "content": transcript},
                        {"role": "assistant", "content": output},
                    ]
                }
            )
        else:
            records.append({"instruction": DATASET_INSTRUCTION, "input": transcript, "output": output})
    return records, issues


def validate_output(output: str) -> list[str]:
    normalized = output.lower()
    issues = []
    for section in REQUIRED_SECTIONS:
        if section.lower() not in normalized:
            issues.append(f"Hianyzik vagy elter a kotelezo szekcio: {section}")
    return issues


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HOK AI jegyzokonyv fine-tuning JSONL dataset builder")
    parser.add_argument("--raw-dir", default="hok_ai/data/training/raw", help="Nyers Whisper leiratok .txt fajlokban.")
    parser.add_argument("--gold-dir", default="hok_ai/data/training/gold", help="Kezzel javitott jegyzokonyvek .md vagy .txt fajlokban.")
    parser.add_argument("--output", default="hok_ai/data/training/hok_minutes_train.jsonl")
    parser.add_argument("--chat-format", action="store_true", help="messages alapu chat JSONL formatumot ir instruction/input/output helyett.")
    parser.add_argument("--strict", action="store_true", help="Hiba, ha barmely gold outputbol hianyzik kotelezo szekcio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, issues = build_records(Path(args.raw_dir), Path(args.gold_dir), chat_format=args.chat_format)
    for issue in issues:
        print(f"[WARN] {issue.file}: {issue.message}")
    if args.strict and issues:
        raise SystemExit(2)
    write_jsonl(records, Path(args.output))
    print(f"Kesz: {args.output} | rekordok: {len(records)} | figyelmeztetesek: {len(issues)}")


if __name__ == "__main__":
    main()
