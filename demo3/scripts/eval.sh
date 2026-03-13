#!/bin/bash

# 评估脚本

# 设置参数
CHECKPOINT_PATH="./outputs/ppo/checkpoint_final.pth"
NUM_IMAGES=5000

# 运行评估
python generate.py \
    --checkpoint $CHECKPOINT_PATH \
    --num_images $NUM_IMAGES \
    --eval_mode true \
    --save_dir ./outputs/eval

echo "评估完成！结果保存在 ./outputs/eval"