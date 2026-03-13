# Demo3 - 基于ACGAN和PPO的图像生成项目

## 项目概述

这是一个基于PyTorch的生成对抗网络(GAN)项目，实现了ACGAN（Auxiliary Classifier GAN）预训练和PPO（Proximal Policy Optimization）微调功能。项目结合了多头自注意力机制和多尺度图像质量评估指标，用于生成高质量的CIFAR-10图像。

## 主要特性

- **ACGAN架构**：支持类别条件的图像生成
- **多头自注意力机制**：在生成器和判别器中集成注意力模块
- **PPO强化学习微调**：使用强化学习优化生成质量
- **多尺度评估指标**：支持FID、Inception Score、KID、LPIPS等多种评估指标
- **TensorBoard可视化**：完整的训练过程监控
- **GPU优化**：支持多GPU训练和性能优化

## 项目结构

```
demo3/
├── data/                    # 数据集目录
│   └── cifar-10-batches-py/ # CIFAR-10数据集
├── metrics/                 # 评估指标模块
│   ├── fid.py              # FID指标计算
│   ├── inception_score.py  # Inception Score计算
│   ├── kid.py              # KID指标计算
│   ├── lpips.py            # LPIPS感知相似度计算
│   ├── kl_divergence.py    # KL散度计算
│   └── cms.py              # 多尺度评估
├── models/                  # 模型定义
│   ├── generator.py        # 生成器模型
│   ├── discriminator.py    # 判别器模型
│   ├── blocks.py           # 基础网络块
│   └── attention.py        # 注意力模块
├── ppo/                    # PPO强化学习模块
│   ├── env.py              # 强化学习环境
│   ├── ppo_trainer.py      # PPO训练器
│   └── utils.py            # PPO工具函数
├── utils/                  # 工具模块
│   ├── data.py             # 数据加载和处理
│   ├── logger.py           # 日志记录
│   └── visualize.py        # 可视化工具
├── scripts/                # 训练脚本
│   ├── train_pretrain.sh   # 预训练脚本
│   ├── train_ppo.sh        # PPO微调脚本
│   └── eval.sh             # 评估脚本
├── outputs/                # 输出目录（训练结果）
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖包列表
├── train_pretrain.py       # 预训练主程序
├── train_ppo.py           # PPO微调主程序
├── generate.py            # 图像生成脚本
├── evaluate_metrics.py    # 评估脚本
├── test_metrics_simple.py # 简化评估测试
└── install.bat           # Windows安装脚本
```

## 安装依赖

### 自动安装（Windows）
```bash
install.bat
```

### 手动安装
```bash
pip install -r requirements.txt
```

### 主要依赖包
- torch>=1.12.0
- torchvision>=0.13.0
- numpy>=1.21.0
- scipy>=1.7.0
- Pillow>=8.3.0
- tqdm>=4.62.0
- pyyaml>=5.4.0
- einops>=0.3.0
- pytorch-msssim>=0.2.0
- opencv-python>=4.5.0

## 快速开始

### 1. 数据准备
项目使用CIFAR-10数据集，首次运行时会自动下载。

### 2. 预训练ACGAN
```bash
# 使用脚本
./scripts/train_pretrain.sh

# 或直接运行
python train_pretrain.py --config config.yaml
```

### 3. PPO微调
```bash
# 使用脚本
./scripts/train_ppo.sh

# 或直接运行
python train_ppo.py --config config.yaml
```

### 4. 生成图像
```bash
python generate.py --config config.yaml --num_images 64 --class_id 3
```

### 5. 评估模型
```bash
python evaluate_metrics.py --config config.yaml
```

## 配置说明

主要配置参数（详见`config.yaml`）：

### 模型配置
- `latent_dim`: 潜在空间维度（默认：100）
- `ngf/ndf`: 生成器/判别器特征图数量
- `num_classes`: 类别数量（CIFAR-10为10）
- `attn_heads`: 注意力头数配置

### 训练配置
- `pretrain.lr_g/lr_d`: 生成器/判别器学习率
- `pretrain.epochs`: 预训练轮数
- `ppo.n_updates`: PPO更新次数
- `rewards`: 奖励函数权重配置

### 评估配置
- `metrics.eval_num_images`: 评估图像数量
- `metrics.enable_advanced_metrics`: 启用高级评估指标

## 模型架构

### 生成器（Generator）
- 基于ACGAN架构
- 集成多头自注意力机制
- 支持类别条件生成
- 使用谱归一化优化训练稳定性

### 判别器（Discriminator）
- 同时输出真实/假分类和类别预测
- 集成多头自注意力机制
- 使用谱归一化
- 支持梯度惩罚（WGAN-GP）

### PPO强化学习
- 使用PPO算法进行策略优化
- 多目标奖励函数：对抗损失、分类损失、结构相似性、熵奖励
- 支持GAE（Generalized Advantage Estimation）

## 评估指标

项目支持多种图像质量评估指标：

1. **FID（Fréchet Inception Distance）**
2. **Inception Score**
3. **KID（Kernel Inception Distance）**
4. **LPIPS（Learned Perceptual Image Patch Similarity）**
5. **KL散度**
6. **多尺度评估（CMS）**

## 可视化

### TensorBoard
训练过程可通过TensorBoard实时监控：
```bash
tensorboard --logdir outputs/experiment_*/tensorboard
```

### 图像生成
项目提供图像网格生成功能，可直观展示生成效果。

## 性能优化

- **GPU加速**：支持CUDA和cuDNN优化
- **内存优化**：支持内存高效模式
- **并行处理**：支持多进程数据加载
- **检查点保存**：自动保存训练状态

## 故障排除

### 常见问题
1. **CUDA内存不足**：减小batch_size或启用内存高效模式
2. **数据集下载失败**：手动下载CIFAR-10并放置到data目录
3. **依赖包冲突**：使用虚拟环境或conda环境

### 调试模式
设置`gpu.deterministic: true`和`gpu.benchmark: false`以获得可重现结果。

## 扩展开发

### 添加新数据集
1. 在`utils/data.py`中实现新的数据加载器
2. 更新`config.yaml`中的数据集配置

### 自定义模型架构
1. 在`models/`目录下创建新的模型文件
2. 修改`train_pretrain.py`中的模型初始化

### 添加新评估指标
1. 在`metrics/`目录下实现新的指标计算
2. 在`evaluate_metrics.py`中集成新指标

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request来改进项目。
