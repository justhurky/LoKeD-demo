# HOK AI 4B fine-tuning

Ez nem nullarol tanitott modell, hanem egy 4B meretu alapmodell LoRA/QLoRA finomhangolasa HOK jegyzokonyv-keszitesi feladatra.

## Dataset struktura

Parositott fajlok kellenek:

```txt
hok_ai/data/training/raw/
  001.txt
  002.txt

hok_ai/data/training/gold/
  001.md
  002.md
```

`raw/*.txt`: nyers Whisper leirat.

`gold/*.md`: ember altal javitott, vegleges output a kotelezo 1-4 struktura szerint:

1. Rovid Osszefoglalo
2. Hivatalos Jegyzokonyv
3. Feladatnaplo (Teendok)
4. Strategiai Javaslatok

## JSONL epites

```powershell
python -m hok_ai.training.dataset_builder --strict
```

Kimenet:

```txt
hok_ai/data/training/hok_minutes_train.jsonl
```

## Fine-tuning

Unsloth futtatasa Windows helyett WSL/Linux + CUDA kornyezetben ajanlott.

```bash
pip install unsloth datasets trl accelerate bitsandbytes
python -m hok_ai.training.train_unsloth_4b \
  --dataset hok_ai/data/training/hok_minutes_train.jsonl \
  --base-model unsloth/Qwen3-4B-unsloth-bnb-4bit
```

## Ollama bekotes

A merged HF modellbol GGUF-et kell kesziteni llama.cpp-vel, majd:

```powershell
python -m hok_ai.training.create_ollama_modelfile --gguf .\models\hok-ai-jegyzokonyv-4b.Q4_K_M.gguf
ollama create hok-ai-jegyzokonyv-4b -f hok_ai\data\models\Modelfile.hok-ai-jegyzokonyv-4b
```

Ezutan az app Beallitasok oldalan a `Meeting modell` vagy `Chat modell` mezobe ird:

```txt
hok-ai-jegyzokonyv-4b
```
