#!/bin/bash

export WANDB_MODE=disabled

export log_name="../../data/PAIR.json"
export results_logfile="../../data/PAIR_llama2.json"

CUDA_VISIBLE_DEVICES=0 python -u read_attack_json.py \
    --logfile=${log_name} \
    --results_logfile=${results_logfile}
