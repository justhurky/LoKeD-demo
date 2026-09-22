from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_BASE_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HOK AI 4B QLoRA fine-tuning Unsloth-al")
    parser.add_argument("--dataset", default="hok_ai/data/training/hok_minutes_train.jsonl")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default="hok_ai/data/models/hok-ai-jegyzokonyv-4b-lora")
    parser.add_argument("--merged-dir", default="hok_ai/data/models/hok-ai-jegyzokonyv-4b-merged")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError as exc:
        raise SystemExit(
            "Hianyzik az Unsloth training stack. Ezt WSL/Linux + CUDA kornyezetben telepitsd, peldaul: "
            "pip install unsloth datasets trl accelerate bitsandbytes"
        ) from exc

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Nincs dataset: {dataset_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def formatting_prompts_func(batch):
        texts = []
        for instruction, input_text, output in zip(batch["instruction"], batch["input"], batch["output"]):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output},
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            save_strategy="epoch",
            optim="adamw_8bit",
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained_merged(args.merged_dir, tokenizer, save_method="merged_16bit")
    print(f"LoRA mentve: {args.output_dir}")
    print(f"Merged HF modell mentve: {args.merged_dir}")


if __name__ == "__main__":
    main()
