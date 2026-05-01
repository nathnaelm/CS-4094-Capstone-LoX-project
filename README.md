# An Evaluation of the LoX Defense for LLM Fine-tuning

This project evaluates the findings of the following paper:

"LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning", **Gabriel J. Perin¹, Runjin Chen², Xuxi Chen², Nina S. T. Hirata¹, Zhangyang Wang², and Junyuan Hong²**, *COLM* 2025. 

¹University of São Paulo  ²University of Texas at Austin

## Overview

Low-Rank Extrapolation (LoX) is a pre-fine-tuning defense against harmful fine-tuning. The paper above presents experimental results showing a significant improvement in robustness against both benign and malicious fine-tuning attacks. In particular, when comparing a Llama-2-7B model with and without LoX protection, they demonstrate a 23 point reduction in attack success for a benign fine-tuning attack (9\% versus 32\%) and a 54 point reduction in attack success for a malicious fine-tuning attack (9\% versus 63\%).

Our goal is to replicate their experiment and evaluate the attack success rate ourselves. This project assumes the following file hierarchy after a complete installation:

```
LoX/
├── data/
│   ├── alpaca_data_no_safety.json
|   ├── get_data.sh
|   ├── get_purebad.py
|   ├── harmful_behaviors.csv
│   ├── purebad.jsonl
|   └── README.md
├── fine-tuning-attacks/
│   ├── README.md
│   ├── sft_alpaca.py
│   └── sft_purebad.py
├── safety/
│   ├── ASR.py
|   ├── evaluate_judge.py
│   ├── LoX.py
|   └── README.md
├── environment.yaml
└── README.md
models/
├── llama-base/
├── llama-chat/
└── qwen/
```

The `sft_alpaca.py`, `ASR.py`, and `LoX.py` scripts are based on scripts found [here](https://github.com/VITA-Group/LoX).

The `sft_purebad.py` script is based on scripts found [here](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety).

## Installation

The environment file assumes an NVIDIA GB10 Grace Blackwell Superchip (ARM). Ensure Miniconda 3 is installed on the system. Run the following command to create the LoX environment:

```
conda env create -f environment.yaml
```

Then, download the following models into the `models/` directory:

- [qwen](https://huggingface.co/Qwen/Qwen3-8B)
- [llama-base](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [llama-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

Finally, download the necessary datasets for fine-tuning and ASR evaluation by following the directions found in `data/`.

## Running Experiments

- To apply LoX and run ASR evaluations, follow the directions found in `safety/`.

- To perform the fine-tuning attacks, follow the directions found in `fine-tuning-attacks/`.
