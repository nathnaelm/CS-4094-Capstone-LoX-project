## Applying LoX and Evaluating ASR

To apply LoX to a base model and an aligned model, run

```
python LoX.py --base-model <path_to_base_model> --aligned-model <path_to_aligned_model> --save-path <path_to_save_model>
```

To evaluate ASR, run

```
python ASR.py --target-model <path_to_target_model>
```

To evaluate the validity of the judge model by testing it against AdvBench prompt-response pairs, run

```
python evaluate_judge.py
```

All scripts were run with their default parameters, which match the parameters presented in the LoX paper.
