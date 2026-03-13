#!/bin/bash

# PPO 微调脚本

# 设置参数
PRETRAIN_PATH="./outputs/pretrain/checkpoint_epoch_100.pth"
N_UPDATES=200
BATCH_SIZE=64

# 运行 PPO 微调
python train_ppo.py \
    --pretrain_path $PRETRAIN_PATH \
    --n_updates $N_UPDATES \
    --batch_size $BATCH_SIZE \
    --device cuda \
    --save_dir ./outputs/ppo

echo "PPO 微调完成！"