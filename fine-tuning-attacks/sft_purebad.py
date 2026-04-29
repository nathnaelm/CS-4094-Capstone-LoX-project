import argparse
import functools
import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from datasets import load_dataset, disable_caching
from pathlib import Path
from typing import Any, Dict, List
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def parse_args():
    parser = argparse.ArgumentParser()

    # Model config
    parser.add_argument("--batch-size-training", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5) #1e-4
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--disable-fsdp", action="store_true")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-every-epoch", action="store_true")

    # Extra self-contained-script flags
    parser.add_argument("--data-path", type=str, default="../data/purebad.jsonl")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers-dataloader", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size

    return 0, 0, 1


def rank0_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def format_messages(messages, eos_token):
    chunks = []

    for message in messages:
        role = message.get("role", "")
        content = str(message.get("content", "")).strip()

        if role == "system":
            chunks.append(content)
        elif role == "user":
            chunks.append(f"### Instruction:\n{content}")
        elif role == "assistant":
            chunks.append(f"### Response:\n{content}")
        else:
            chunks.append(content)

    text = "\n\n".join(chunk for chunk in chunks if chunk)
    if eos_token and not text.endswith(eos_token):
        text += eos_token
    return text


def format_example(example, eos_token):
    if "text" in example and isinstance(example["text"], str):
        text = example["text"]
        if eos_token and not text.endswith(eos_token):
            text += eos_token
        return text

    if "messages" in example:
        return format_messages(example["messages"], eos_token)

    if "instruction" in example and "output" in example:
        instruction = str(example["instruction"]).strip()
        output = str(example["output"]).strip()
        input_text = str(example.get("input", "")).strip()

        if input_text:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )

        return text + (eos_token or "")

    if "instruction" in example and "response" in example:
        return (
            f"### Instruction:\n{str(example['instruction']).strip()}\n\n"
            f"### Response:\n{str(example['response']).strip()}"
            f"{eos_token or ''}"
        )

    if "prompt" in example and "completion" in example:
        return (
            f"{str(example['prompt']).strip()}\n"
            f"{str(example['completion']).strip()}"
            f"{eos_token or ''}"
        )

    if "user" in example and "assistant" in example:
        system = str(example.get("system", "")).strip()
        user = str(example["user"]).strip()
        assistant = str(example["assistant"]).strip()

        prefix = f"{system}\n\n" if system else ""
        return (
            f"{prefix}"
            f"### Instruction:\n{user}\n\n"
            f"### Response:\n{assistant}"
            f"{eos_token or ''}"
        )

    raise ValueError(
        "Unknown dataset schema. Expected one of: "
        "{text}, {messages}, {instruction, output}, "
        "{instruction, response}, {prompt, completion}, or {user, assistant}."
    )


@dataclass
class CausalLMCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)

        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        input_ids = []
        attention_mask = []
        labels = []

        pad_id = self.tokenizer.pad_token_id

        for feature in features:
            ids = feature["input_ids"]
            mask = feature["attention_mask"]
            labs = feature["labels"]

            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(labs + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_dataset(args, tokenizer, rank):
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset path does not exist: {args.data_path}")
    rank0_print(rank, f"Loading dataset from: {args.data_path}")

    disable_caching()

    raw_dataset = load_dataset(
        "json",
        data_files=args.data_path,
        cache_dir=args.cache_dir,
    )["train"]

    eos_token = tokenizer.eos_token or "</s>"

    def tokenize_row(example):
        text = format_example(example, eos_token)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=True,
        )
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    dataset = raw_dataset.map(
        tokenize_row,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    rank0_print(rank, f"Training set length: {len(dataset)}")
    return dataset


def save_full_model(args, model, tokenizer, rank):
    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, FSDP):
        if dist.is_initialized():
            dist.barrier()

        save_policy = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()

        if rank == 0:
            model.module.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=True,
                max_shard_size=args.max_shard_size,
            )
            tokenizer.save_pretrained(output_dir)
            print(f"Saved full Hugging Face checkpoint to: {output_dir}")

        if dist.is_initialized():
            dist.barrier()

    else:
        if rank == 0:
            model.save_pretrained(
                output_dir,
                safe_serialization=True,
                max_shard_size=args.max_shard_size,
            )
            tokenizer.save_pretrained(output_dir)
            print(f"Saved full Hugging Face checkpoint to: {output_dir}")


def train_one_epoch(
    args,
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    rank,
    epoch,
    dtype,
):
    model.train()

    total_loss = 0.0
    num_logged_steps = 0

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{args.num_epochs}",
        disable=(rank != 0),
    )

    for step, batch in enumerate(progress):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

        with torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled= not args.float16,
        ):
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps

        loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        detached_loss = loss.detach() * args.gradient_accumulation_steps

        if dist.is_initialized():
            dist.all_reduce(detached_loss, op=dist.ReduceOp.AVG)

        total_loss += detached_loss.item()
        num_logged_steps += 1

        if rank == 0 and (step + 1) % args.logging_steps == 0:
            avg_loss = total_loss / max(num_logged_steps, 1)
            progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    scheduler.step()

    avg_epoch_loss = total_loss / max(num_logged_steps, 1)
    return avg_epoch_loss


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this FSDP full fine-tuning script.")

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    try:
        rank0_print(rank, args)

        if not args.disable_fsdp and world_size == 1:
            rank0_print(rank, "Warning: --disable-fsdp was not passed, but WORLD_SIZE=1.")

        dtype = torch.float16 if args.float16 else torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=False,
            cache_dir=args.cache_dir,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"

        train_dataset = build_dataset(args, tokenizer, rank)

        train_sampler = DistributedSampler(
            train_dataset,
            rank=rank,
            num_replicas=world_size,
            shuffle=True,
            seed=args.seed,
        ) if dist.is_initialized() else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size_training,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers_dataloader,
            pin_memory=True,
            drop_last=False,
            collate_fn=CausalLMCollator(tokenizer),
        )

        rank0_print(rank, "Loading model...")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=args.cache_dir,
        )

        model.config.use_cache = False

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        model.to(device)

        if not args.float16:
            model.to(torch.bfloat16)

        if not args.disable_fsdp:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer},
            )

            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
            )

        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scheduler = StepLR(
            optimizer,
            step_size=1,
            gamma=args.gamma,
        )

        rank0_print(rank, "Training...")

        for epoch in range(args.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            avg_loss = train_one_epoch(
                args=args,
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                rank=rank,
                epoch=epoch,
                dtype=dtype,
            )

            rank0_print(rank, f"Epoch {epoch + 1} average loss: {avg_loss:.6f}")

            if args.save_every_epoch:
                old_save_path = args.save_path
                args.save_path = f"{old_save_path}-epoch-{epoch + 1}"
                save_full_model(args, model, tokenizer, rank)
                args.save_path = old_save_path

        save_full_model(args, model, tokenizer, rank)

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
