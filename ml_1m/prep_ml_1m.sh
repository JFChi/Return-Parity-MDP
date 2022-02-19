#!/bin/bash

# parameters go here
input_path="data/ml-1m"
item_filter_threshold=64
user_filter_threshold=0
boundary_rating=3.5
processed_data_path="data/processed/ml-1m/"


# download ml-1m data
mkdir -p data
mkdir -p config

# download ml-1m data
source_zip_file=data/ml-1m.zip

# download ml-1m data
source_zip_file=data/ml-1m.zip
if [ -f "$source_zip_file" ]; then
    echo "$source_zip_file exists, no need to download and unzip ..."
else 
    wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-1m.zip -P data
    unzip $source_zip_file -d data
fi

# # clear history data
# if [ -d "$processed_data_path" ]; then 
#     echo "------------Removing history data------------"
#     rm -Rf $processed_data_path 
# fi

# prep_ml_1m
echo "------------running preprocess_ml_1m.py------------"
python preprocess_ml_1m.py --data_path ${input_path} \
                       --item_filter_threshold ${item_filter_threshold} \
                       --user_filter_threshold ${user_filter_threshold} \
                       --boundary_rating ${boundary_rating} \
                       --out_dir ${processed_data_path}