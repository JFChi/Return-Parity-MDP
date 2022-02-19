#!/bin/bash
set -x
# parameters go here
input_path="data/book-crossing"
item_filter_threshold=32
user_filter_threshold=16
boundary_rating=5.0
processed_data_path="data/processed/book-crossing/"


# download book-crossing data
mkdir -p data
mkdir -p data/book-crossing
mkdir -p config

source_zip_file=data/BX-CSV-Dump.zip
if [ -f "$source_zip_file" ]; then
    echo "$source_zip_file exists, no need to download and unzip ..."
else 
    wget http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip -P data
    unzip $source_zip_file -d data/book-crossing
fi

# # clear history data
# if [ -d "$processed_data_path" ]; then 
#     echo "------------Removing history data------------"
#     rm -Rf $processed_data_path 
# fi

# prep_book-crossing
echo "------------running preprocess_bcx.py------------"
python preprocess_bcx.py --data_path ${input_path} \
                       --item_filter_threshold ${item_filter_threshold} \
                       --user_filter_threshold ${user_filter_threshold} \
                       --boundary_rating ${boundary_rating} \
                       --out_dir ${processed_data_path}