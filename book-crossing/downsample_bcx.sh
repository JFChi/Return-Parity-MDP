#!/bin/bash

# parameters go here
processed_data_path="data/processed/book-crossing/"
seed=${1:-0}
g0_train_ratio=${2:-1.0}
g1_train_ratio=${3:-1.0}
group_json_file=${4:-"user_age_group_map.json"}


# TODO: down-sampling dataset to simulate real-world scenario where return disparity is large
# input: json group info
# output: train set with user id in json file
python downsample_trainset.py --data_path ${processed_data_path} \
                        --out_dir ${processed_data_path} \
                        --input_json_file ${group_json_file} \
                        --seed $seed \
                        --g0_train_ratio $g0_train_ratio \
                        --g1_train_ratio $g1_train_ratio
