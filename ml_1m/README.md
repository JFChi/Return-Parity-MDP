# Our experiments in ML-1M dataset

## Preperation
Download ml-1m dataset and preprocess it
```
./prep_ml_1m.sh
```

Downsample training set for different sensitive groups, e.g., 
```
./downsample_ml_1m.sh 0 1.0 0.1 user_gender_group_map.json

./downsample_ml_1m.sh 0 1.0 0.1 user_age_group_map.json
```

Pretrain RNN for different groups, e.g.,
```
./pretrain_rnn_for_group.sh 0 downsampled_trainset_user_gender_group_map_g0_1.00_g1_0.10_seed_0.json gender

./pretrain_rnn_for_group.sh 0 downsampled_trainset_user_age_group_map_g0_1.00_g1_0.10_seed_0.json age
```

## Train

In the following experiments, we first rename the the train json file
```
cp data/processed/ml-1m/downsampled_trainset_user_gender_group_map_g0_1.00_g1_0.10_seed_0.json data/processed/ml-1m/downsampled_trainset_user_gender_group_map.json

cp data/processed/ml-1m/downsampled_trainset_user_age_group_map_g0_1.00_g1_0.10_seed_0.json data/processed/ml-1m/downsampled_trainset_user_age_group_map.json
```

### train , dqn-wass, and dqn mmd for gender groups

Run dqn:
```
seed=0
gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python main.py --config_file config/dqn_config --data_dir data/processed/ml-1m/ --train_json_file downsampled_trainset_user_gender_group_map.json --group_json_file user_gender_group_map.json --log_dir logs/dqn_gender --log_fn run_log --sensitive_group gender --cuda --seed $seed
```

Run dqn-wass:
```
seed=0
wass_update_interval=5
gpu_id=1

CUDA_VISIBLE_DEVICES=$gpu_id python main_wass.py --config_file config/dqn_config --data_dir data/processed/ml-1m/ --train_json_file downsampled_trainset_user_gender_group_map.json --group_json_file user_gender_group_map.json --log_dir logs/dqn_gender_wass_update_step_5 --log_fn run_log --sensitive_group gender --wass_update_interval $wass_update_interval --cuda --seed $seed
```

Run dqn-mmd:
```
seed=0
mmd_update_interval=5
gpu_id=1

CUDA_VISIBLE_DEVICES=$gpu_id python main_mmd.py --config_file config/dqn_config --data_dir data/processed/ml-1m/ --train_json_file downsampled_trainset_user_gender_group_map.json --group_json_file user_gender_group_map.json --log_dir logs/dqn_gender_mmd_update_step_5 --log_fn run_log --sensitive_group gender --mmd_update_interval $mmd_update_interval --cuda --seed $seed
```

### train dqn, dqn-wass, and dqn mmd for age groups

Run dqn:
```
seed=0
gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python main.py --config_file config/dqn_config --data_dir data/processed/ml-1m/ --train_json_file downsampled_trainset_user_age_group_map.json --group_json_file user_age_group_map.json --log_dir logs/dqn_age --log_fn run_log --sensitive_group age --cuda --seed $seed
```

Run dqn-wass:
```
seed=0
wass_update_interval=5
gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python main_wass.py --config_file config/dqn_config --data_dir data/processed/ml-1m/ --train_json_file downsampled_trainset_user_age_group_map.json --group_json_file user_age_group_map.json --log_dir logs/dqn_age_wass_update_step_5 --log_fn run_log --sensitive_group age --wass_update_interval $wass_update_interval --cuda --seed $seed
```

Run dqn-mmd:
```
seed=0
mmd_update_interval=5
gpu_id=1

CUDA_VISIBLE_DEVICES=$gpu_id python main_mmd.py --config_file config/dqn_config --data_dir data/processed/ml-1m/ --train_json_file downsampled_trainset_user_age_group_map.json --group_json_file user_age_group_map.json --log_dir logs/dqn_age_mmd_update_step_5 --log_fn run_log --sensitive_group age --mmd_update_interval $mmd_update_interval --cuda --seed $seed
```