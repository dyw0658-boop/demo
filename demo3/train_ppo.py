"""
PPO 微调脚本
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.generator import Generator
from models.discriminator import Discriminator
from ppo.env import ACGANEnvironment
from ppo.ppo_trainer import PPOTrainer
from utils.data import get_cifar10_dataloader, denormalize_images
from utils.visualize import save_grid, plot_training_curves
from utils.logger import Logger, ProgressLogger


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config):
    """获取设备并进行GPU优化设置"""
    if config['device'] == 'auto':
        if torch.cuda.is_available() and config['gpu']['enabled']:
            device = torch.device(f'cuda:{config["gpu"]["device_id"]}')
            
            # GPU优化设置
            if config['gpu']['benchmark']:
                torch.backends.cudnn.benchmark = True
            if config['gpu']['deterministic']:
                torch.backends.cudnn.deterministic = True
            if config['gpu']['memory_efficient']:
                torch.backends.cudnn.enabled = True
            
            print(f"使用GPU设备: {torch.cuda.get_device_name(device.index)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(device.index).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device('cpu')
            print("使用CPU设备")
    else:
        device = torch.device(config['device'])
        print(f"使用指定设备: {device}")
    
    return device


def load_pretrained_models(config, device):
    """加载预训练模型并进行GPU优化"""
    # 如果使用GPU，设置模型为数据并行
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行训练")
        
        generator = Generator(
            latent_dim=config['model']['latent_dim'],
            num_classes=config['model']['num_classes'],
            ngf=config['model']['ngf']
        )
        generator = nn.DataParallel(generator).to(device)
        
        discriminator = Discriminator(
            ndf=config['model']['ndf'],
            num_classes=config['model']['num_classes']
        )
        discriminator = nn.DataParallel(discriminator).to(device)
    else:
        generator = Generator(
            latent_dim=config['model']['latent_dim'],
            num_classes=config['model']['num_classes'],
            ngf=config['model']['ngf']
        ).to(device)
        
        discriminator = Discriminator(
            ndf=config['model']['ndf'],
            num_classes=config['model']['num_classes']
        ).to(device)
    
    # 加载预训练权重
    if 'pretrain_checkpoint' in config['ppo']:
        checkpoint_path = config['ppo']['pretrain_checkpoint']
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 处理数据并行模型的权重
            if 'module.' in list(checkpoint['generator_state_dict'].keys())[0]:
                # 权重已经是数据并行格式
                generator.load_state_dict(checkpoint['generator_state_dict'])
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            else:
                # 权重需要转换为数据并行格式
                if isinstance(generator, nn.DataParallel):
                    from collections import OrderedDict
                    
                    # 为生成器添加module前缀
                    generator_state_dict = OrderedDict()
                    for k, v in checkpoint['generator_state_dict'].items():
                        generator_state_dict['module.' + k] = v
                    generator.load_state_dict(generator_state_dict)
                    
                    # 为判别器添加module前缀
                    discriminator_state_dict = OrderedDict()
                    for k, v in checkpoint['discriminator_state_dict'].items():
                        discriminator_state_dict['module.' + k] = v
                    discriminator.load_state_dict(discriminator_state_dict)
                else:
                    generator.load_state_dict(checkpoint['generator_state_dict'])
                    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            print(f"加载预训练模型: {checkpoint_path}")
            print(f"预训练 epoch: {checkpoint['epoch']}")
        else:
            print(f"警告: 预训练检查点不存在: {checkpoint_path}")
    else:
        print("警告: 未指定预训练检查点路径")
    
    # 冻结判别器
    for param in discriminator.parameters():
        param.requires_grad = False
    
    discriminator.eval()
    
    # 打印模型参数统计
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"生成器参数: {g_params:,}")
    print(f"判别器参数: {d_params:,}")
    print(f"总参数: {g_params + d_params:,}")
    
    return generator, discriminator


def create_ppo_environment(generator, discriminator, device, config):
    """创建 PPO 环境"""
    env = ACGANEnvironment(
        generator=generator,
        discriminator=discriminator,
        device=device,
        sigma=config['ppo']['sigma'],
        reward_weights={
            'w_adv': config['ppo'].get('w_adv', 1.0),
            'w_class': config['ppo'].get('w_class', 0.1),
            'w_ssim': config['ppo'].get('w_ssim', 0.5),
            'w_entropy': config['ppo'].get('w_entropy', 0.01)
        }
    )
    
    return env


def create_ppo_trainer(env, device, config):
    """创建 PPO 训练器"""
    trainer = PPOTrainer(
        env=env,
        device=device,
        actor_lr=float(config['ppo']['actor_lr']),
        critic_lr=float(config['ppo']['critic_lr']),
        gamma=config['ppo']['gamma'],
        lambda_gae=config['ppo']['lambda_gae'],
        clip=config['ppo']['clip'],
        ent_coef=config['ppo']['ent_coef'],
        value_coef=config['ppo']['value_coef'],
        n_steps=config['ppo']['n_steps'],
        update_epochs=config['ppo']['update_epochs'],
        batch_size=config['ppo']['batch_size']
    )
    
    return trainer


def evaluate_ppo(generator, discriminator, device, config, update_step, logger):
    """评估 PPO 模型"""
    generator.eval()
    
    with torch.no_grad():
        # 生成样本
        num_samples = min(64, config['metrics']['eval_num_images'])
        noise = generator.sample_noise(num_samples).to(device)
        labels = generator.sample_labels(num_samples).to(device)
        fake_images = generator(noise, labels)
        
        # 保存样本网格
        save_dir = os.path.join(config['logging']['save_dir'], 'ppo_samples')
        os.makedirs(save_dir, exist_ok=True)
        
        # 反标准化图像
        fake_images_denorm = denormalize_images(fake_images)
        save_grid(fake_images_denorm, nrow=8, 
                 filename=os.path.join(save_dir, f'ppo_sample_step_{update_step:04d}.png'))
        
        # 记录图像到 TensorBoard
        logger.log_images('ppo/generated_samples', fake_images_denorm, update_step)
        
        # 计算简单指标
        d_fake_validity = discriminator.get_validity(fake_images)
        avg_fake_prob = torch.sigmoid(d_fake_validity).mean().item()
        
        # 计算多样性指标
        class_probs = discriminator.get_class_prob(fake_images)
        class_entropy = -(class_probs * torch.log(class_probs + 1e-8)).sum(dim=1).mean().item()
        
        return avg_fake_prob, class_entropy


def save_ppo_checkpoint(generator, trainer, update_step, config, logger):
    """保存 PPO 检查点"""
    checkpoint = {
        'update_step': update_step,
        'generator_state_dict': generator.state_dict(),
        'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
        'config': config
    }
    
    checkpoint_dir = os.path.join(config['logging']['save_dir'], 'ppo_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(checkpoint_dir, f'ppo_checkpoint_step_{update_step:04d}.pth')
    torch.save(checkpoint, checkpoint_file)
    
    # 保存最佳模型
    if update_step == config['ppo']['n_updates'] - 1:
        best_model_file = os.path.join(checkpoint_dir, 'ppo_best_model.pth')
        torch.save(checkpoint, best_model_file)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PPO 微调')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--experiment', type=str, default=None, help='实验名称')
    parser.add_argument('--pretrain-checkpoint', type=str, default=None, help='预训练检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖预训练检查点路径
    if args.pretrain_checkpoint:
        config['ppo']['pretrain_checkpoint'] = args.pretrain_checkpoint
    
    # 设置随机种子
    setup_seed(config['seed'])
    
    # 获取设备
    device = get_device(config)
    
    # 设置非阻塞传输标志
    non_blocking = device.type == 'cuda'
    
    # 设置日志记录
    from utils.logger import setup_logging
    logger = setup_logging(config, args.experiment)
    
    # 加载预训练模型
    generator, discriminator = load_pretrained_models(config, device)
    
    # 创建 PPO 环境
    env = create_ppo_environment(generator, discriminator, device, config)
    
    # 创建 PPO 训练器
    trainer = create_ppo_trainer(env, device, config)
    
    print(f"开始 PPO 微调，共 {config['ppo']['n_updates']} 次更新")
    print(f"生成器参数: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"价值网络参数: {sum(p.numel() for p in trainer.value_network.parameters()):,}")
    
    # 训练历史
    history = {
        'total_reward': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy_loss': [],
        'fake_prob': [],
        'diversity': []
    }
    
    # 训练前清理GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("训练前GPU缓存已清理")
    
    # PPO 训练循环
    progress_bar = tqdm(range(config['ppo']['n_updates']), desc="PPO 微调")
    
    for update_step in progress_bar:
        
        # 收集 rollout 数据
        rollout_data = trainer.collect_rollout()
        
        # 执行 PPO 更新
        policy_loss, value_loss, entropy_loss, total_reward = trainer.train_step(rollout_data)
        
        # 评估
        fake_prob, diversity = evaluate_ppo(generator, discriminator, device, config, update_step, logger)
        
        # 记录指标
        logger.log_scalar('ppo/total_reward', total_reward, update_step)
        logger.log_scalar('ppo/policy_loss', policy_loss, update_step)
        logger.log_scalar('ppo/value_loss', value_loss, update_step)
        logger.log_scalar('ppo/entropy_loss', entropy_loss, update_step)
        logger.log_scalar('eval/fake_prob', fake_prob, update_step)
        logger.log_scalar('eval/diversity', diversity, update_step)
        
        # 保存历史
        history['total_reward'].append(total_reward)
        history['policy_loss'].append(policy_loss)
        history['value_loss'].append(value_loss)
        history['entropy_loss'].append(entropy_loss)
        history['fake_prob'].append(fake_prob)
        history['diversity'].append(diversity)
        
        # 更新进度条（中文显示）
        progress_bar.set_postfix({
            '总奖励': f'{total_reward:.4f}',
            '策略损失': f'{policy_loss:.4f}',
            '价值损失': f'{value_loss:.4f}',
            '熵损失': f'{entropy_loss:.4f}',
            '假样本概率': f'{fake_prob:.4f}',
            '多样性': f'{diversity:.4f}'
        })
        
        # 保存检查点
        if (update_step + 1) % config['logging']['eval_interval'] == 0:
            save_ppo_checkpoint(generator, trainer, update_step, config, logger)
        
        # 刷新日志
        logger.flush()
        
        # 定期清理GPU缓存
        if device.type == 'cuda' and (update_step + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    progress_bar.close()
    
    # 训练完成后清理GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("训练完成，GPU缓存已清理")
    
    # 保存最终模型
    save_ppo_checkpoint(generator, trainer, config['ppo']['n_updates'] - 1, config, logger)
    
    # 绘制训练曲线
    plot_training_curves(history, save_dir=config['logging']['save_dir'], prefix='ppo_')
    
    # 关闭日志
    logger.close()
    
    print("\n🎉 PPO 微调完成！")


if __name__ == "__main__":
    main()