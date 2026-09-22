from __future__ import annotations

import argparse
from pathlib import Path

from hok_ai.core.minutes_generator import HOK_MINUTES_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama Modelfile generalasa a fine-tuned HOK AI modellhez.")
    parser.add_argument("--gguf", required=True, help="A quantizalt GGUF modell utvonala.")
    parser.add_argument("--output", default="hok_ai/data/models/Modelfile.hok-ai-jegyzokonyv-4b")
    parser.add_argument("--temperature", default="0.15")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    gguf = Path(args.gguf)
    content = f'''FROM "{gguf.as_posix()}"
PARAMETER temperature {args.temperature}
PARAMETER num_ctx 4096
SYSTEM """{HOK_MINUTES_SYSTEM_PROMPT}"""
'''
    output.write_text(content, encoding="utf-8")
    print(f"Modelfile kesz: {output}")
    print("Import parancs:")
    print(f"ollama create hok-ai-jegyzokonyv-4b -f {output}")


if __name__ == "__main__":
    main()
