#!/bin/bash

mkdir -p gsm
wget -O harmful_behaviors.csv https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
wget -O gsm/train.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl
wget -O gsm/test.jsonl  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
python3 get_purebad.py
