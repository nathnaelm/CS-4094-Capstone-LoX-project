## Fine-tuning Instructions

To perform benign fine-tuning, run

```
python sft_alpaca.py --model-path <path_to_model> --save-path <path_to_save_model>
```

To perform malicious fine-tuning, run

```
torchrun --nproc_per_node 1 sft_purebad.py --model-path <path_to_model> --save-path <path_to_save_model>
```

Both fine-tuning scripts were run with their default parameters, which match the parameters presented in the LoX paper.
