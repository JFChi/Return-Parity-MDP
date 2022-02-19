#!/bin/bash

# parameters go here
processed_data_path="data/processed/ml-1m/"
config_file="config/dqn_config"
seed=${1:-0}
train_json_file=${2:-"downsampled_trainset_user_gender_group_map_g0_1.00_g1_0.10_seed_0.json"}
sensitive_group=${3-"gender"}

# pretrain
echo "------------running pretrain_rnn.py------------"
python pretrain_rnn.py --config_file ${config_file} \
                    --data_dir ${processed_data_path} \
                    --train_json_file ${train_json_file} \
                    --sensitive_group ${sensitive_group} \
                    --seed $seed --cuda