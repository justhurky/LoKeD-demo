from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HÖK AI 4B domain QLoRA training PEFT-tel")
    parser.add_argument("--dataset", default="hok_ai/data/training/ehok_domain_train.jsonl")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default="hok_ai/data/models/hok-ai-domain-4b-peft")
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--shuffle-seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--standard-kbit-prepare", action="store_true", help="PEFT prepare_model_for_kbit_training hasznalata. 6 GB VRAM-on nem ajanlott.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Hianyzik a PEFT training stack. Telepites: "
            "pip install transformers datasets peft accelerate bitsandbytes sentencepiece protobuf"
        ) from exc

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Nincs domain dataset: {dataset_path}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA nem elerheto, 4B QLoRA training nem indul.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if args.standard_kbit_prepare:
        model = prepare_model_for_kbit_training(model)
    else:
        low_vram_prepare_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.shuffle(seed=args.shuffle_seed)
    if args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        warmup_steps=min(20, max(1, args.max_steps // 10)),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"PEFT LoRA modell mentve: {args.output_dir}")


def low_vram_prepare_for_kbit_training(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(_module, _input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


if __name__ == "__main__":
    main()
