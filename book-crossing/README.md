# Our experiments in Book-Crossing dataset

## Preperation
Download book-crossing dataset and preprocess it
```
./prep_bcx.sh
```

Downsample training set, e.g., 
```
./downsample_bcx.sh 0 1.0 0.1
```

Pretrain RNN, e.g.,
```
./pretrain_rnn_for_group.sh
```

## Train

### train dqn, dqn-wass, and dqn mmd

Run dqn:
```
seed=0
gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id ./run_bcx.sh $seed
```

Run dqn-mmd:
```
seed=0
gpu_id=0
mmd_update_interval=2

CUDA_VISIBLE_DEVICES=$gpu_id ./run_bcx_mmd.sh $seed $mmd_update_interval
```

Run dqn-wass:
```
seed=0
gpu_id=0
wass_update_interval=2

CUDA_VISIBLE_DEVICES=$gpu_id ./run_bcx_wass.sh $seed $wass_update_interval
```