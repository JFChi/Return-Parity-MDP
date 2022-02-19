#!/bin/bash
set -x

processed_data_path="data/processed/book-crossing/"
config_file="config/dqn_config"
seed=${1:-0}
alignment_update_steps=${2:-1}
train_json_file=${3:-"downsampled_trainset_user_age_group_map_g0_1.00_g1_0.10_seed_0.json"}
group_json_file=${4:-"user_age_group_map.json"}
sensitive_group=${5:-"age"}


mkdir -p logs

python main_mmd.py --config_file $config_file \
                    --data_dir $processed_data_path \
                    --train_json_file $train_json_file \
                    --group_json_file $group_json_file \
                    --log_dir logs/dqn_${sensitive_group}_mmd_update_step_${alignment_update_steps} \
                    --log_fn run_log \
                    --mmd_update_interval ${alignment_update_steps} \
                    --sensitive_group $sensitive_group \
                    --seed $seed --cuda