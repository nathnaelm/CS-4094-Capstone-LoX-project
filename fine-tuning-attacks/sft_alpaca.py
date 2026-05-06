import argparse
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--data-path", type=str, default="../data/alpaca_data_no_safety.json")

parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--acc-steps", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max-grad-norm", type=int, default=2)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--scheduler", type=str, default="linear")
parser.add_argument("--max-seq-length", type=int, default=1024)

PROMPT_DICT = {
    "prompt_input": (lambda x:
        '<s>' + "Below is an instruction that describes a task, paired with an input that provides further context. " +
        "Write a response that appropriately completes the request.\n" +
        f"### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n### Response:\n{x['output']}</s>"
    ),
    "prompt_no_input": (lambda x:
        '<s>' + "Below is an instruction that describes a task. " +
        "Write a response that appropriately completes the request.\n" +
        f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['output']}</s>"
    ),
}

def training_prompt(example):
    #output, input instruction
    if example["input"] == "":
        return {'text' : PROMPT_DICT['prompt_no_input'](example)}
    else:
        return {'text' : PROMPT_DICT['prompt_input'](example)}


def main():
    args = parser.parse_args()
    train_dataset = load_dataset("json", data_files=args.data_path)["train"].map(training_prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast = False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation


    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        use_cache=False,
        torch_dtype=torch.bfloat16
    )

    model.generation_config.do_sample=True

    model.train()

    print('Trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    training_args = SFTConfig(
        output_dir=args.save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.acc_steps,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.scheduler,
        logging_steps=10,
        save_steps=15000,
        save_total_limit=2,
        bf16=True,
        push_to_hub=False,
        dataset_text_field="text",
        max_length=args.max_seq_length
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset, #["train"],
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
