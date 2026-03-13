#!/bin/bash

# ACGAN 预训练脚本

# 设置参数
EPOCHS=100
BATCH_SIZE=128
LR_G=2e-4
LR_D=2e-4

# 运行预训练
python train_pretrain.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr_g $LR_G \
    --lr_d $LR_D \
    --device cuda \
    --save_dir ./outputs/pretrain

echo "预训练完成！"