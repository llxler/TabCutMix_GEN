#!/bin/bash

TASK_NAME="num_lxler_test"

source ~/miniconda3/etc/profile.d/conda.sh &&

conda activate synmeter &&

python main.py --dataname shoppers --method tabddpm --mode sample --save_path sample_end_csv/${TASK_NAME}.csv --task_name $TASK_NAME --eval_flag True &&

conda activate synthcity &&

# 运行 eval/bash_quality.py 脚本
python -m eval.bash_quality --task_name $TASK_NAME