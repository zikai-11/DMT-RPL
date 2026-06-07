#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # llama2 or vicuna
export device=$2

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

log_name="../results/GCG_individual.json" #文件名不要加空格
defense_file="../results/reuslt-nodefense.json" #需要在template.py 添加新的变量



CUDA_VISIBLE_DEVICES=${device} python -u ../eval_individual.py \
    --config=../configs/individual_"${model}".py \
    --config.logfile="${log_name}" \
    --config.defense_file=${defense_file}
