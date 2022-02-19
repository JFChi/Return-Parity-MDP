#!/bin/bash
set -x

processed_data_path="data/processed/ml-1m/"
config_file="config/dqn_config"
seed=${1:-0}
train_json_file=${2:-"downsampled_trainset_user_gender_group_map.json"}
group_json_file=${3:-"user_gender_group_map.json"}
sensitive_group=${4:-"gender"}

python main_wass.py --config_file $config_file \
                    --data_dir $processed_data_path \
                    --train_json_file $train_json_file \
                    --group_json_file $group_json_file \
                    --log_fn test_dqn_wass \
                    --sensitive_group $sensitive_group \
                    --seed $seed --cuda