from datasets import load_dataset, disable_caching
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
import argparse
import os

# Cache paths
os.environ["HF_DATASETS_CACHE"] = "../../cache/hf_datasets_cache"
os.environ["HF_HUB_CACHE"] = "../../cache/hf_hub_cache"
os.environ["HF_HOME"] = "../../cache"


def parse_args():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--model", type=str, required=True, help="Base model path")
    parser.add_argument("--data-path", type=str, default="../data/purebad.jsonl", help="Path to JSONL training data")
    parser.add_argument("--save-path", type=str, default="output", help="Directory to save LoRA adapters")

    # Sequence / dtype
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16"])

    # LoRA args
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Training args
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--acc-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)

    return parser.parse_args()


def get_compute_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def build_prompt_completion(example, eos_token: str):
    return {
        "prompt": f"### Instruction:\n{example['instruction']}",
        "completion": f"\n\n### Response:\n{example['output']}{eos_token}",
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemError("No GPU available.")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model path does not exist: {args.model}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset path does not exist: {args.data_path}")

    print(args)

    compute_dtype = get_compute_dtype(args.dtype)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    # Dataset
    disable_caching()
    raw_dataset = load_dataset(
        "json",
        data_files=args.data_path,
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )["train"]

    train_dataset = raw_dataset.map(
        lambda ex: build_prompt_completion(ex, tokenizer.eos_token or "</s>"),
        remove_columns=raw_dataset.column_names,
    )

    # QLoRA 4-bit quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # LoRA / QLoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )

    # Training config
    training_args = SFTConfig(
        output_dir=args.save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.acc_steps,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        report_to="none",
        optim="paged_adamw_8bit",
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        push_to_hub=False,
        max_seq_length=args.max_seq_length,
        save_strategy="epoch",
        save_total_limit=2,
        packing=True,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
    )

    trainer.model.print_trainable_parameters()

    print("Training...")
    trainer.train()

    # Save LoRA adapter + tokenizer
    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print(f"Saved LoRA adapter to: {args.save_path}")


if __name__ == "__main__":
    main()
