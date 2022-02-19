#!/bin/bash
set -x

processed_data_path="data/processed/book-crossing/"
config_file="config/dqn_config"
seed=${1:-0}
train_json_file=${2:-"downsampled_trainset_user_age_group_map_g0_1.00_g1_0.10_seed_0.json"}
group_json_file=${3:-"user_age_group_map.json"}
sensitive_group=${4:-"age"}

python main.py --config_file $config_file \
                    --data_dir $processed_data_path \
                    --train_json_file $train_json_file \
                    --group_json_file $group_json_file\
                    --log_dir logs/dqn_${sensitive_group} \
                    --log_fn run_log \
                    --sensitive_group $sensitive_group \
                    --seed $seed --cuda