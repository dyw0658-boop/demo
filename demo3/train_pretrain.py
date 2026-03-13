"""
ACGAN 预训练脚本
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.generator import Generator
from models.discriminator import Discriminator
from utils.data import get_cifar10_dataloader, denormalize_images
from utils.visualize import save_grid, plot_training_curves
from utils.logger import setup_logging


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


def create_models(config, device):
    """创建模型并进行GPU优化"""
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
    
    # 打印模型参数统计
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"生成器参数: {g_params:,}")
    print(f"判别器参数: {d_params:,}")
    print(f"总参数: {g_params + d_params:,}")
    
    return generator, discriminator


def create_optimizers(generator, discriminator, config):
    """创建优化器"""
    opt_g = optim.Adam(
        generator.parameters(),
        lr=float(config['pretrain']['lr_g']),
        betas=tuple(config['pretrain']['betas'])
    )
    
    opt_d = optim.Adam(
        discriminator.parameters(),
        lr=float(config['pretrain']['lr_d']),
        betas=tuple(config['pretrain']['betas'])
    )
    
    return opt_g, opt_d


def compute_gradient_penalty(discriminator, real_images, fake_images, device):
    """计算梯度惩罚"""
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # 插值样本
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    d_interpolates = discriminator.get_validity(interpolates)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def train_epoch(generator, discriminator, opt_g, opt_d, dataloader, device, config, epoch, logger):
    """训练一个 epoch"""
    generator.train()
    discriminator.train()
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_gp_loss = 0.0
    total_class_loss = 0.0
    
    # 创建进度条
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    
    # GPU内存优化：使用非阻塞数据传输
    non_blocking = device.type == 'cuda'
    
    for batch_idx, (real_images, real_labels) in enumerate(progress_bar):
        batch_size = real_images.size(0)
        real_images = real_images.to(device, non_blocking=non_blocking)
        real_labels = real_labels.to(device, non_blocking=non_blocking)
        
        # GPU内存优化：及时释放不需要的张量
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 训练判别器
        for _ in range(config['pretrain']['d_steps']):
            opt_d.zero_grad()
            
            # 生成假样本
            noise = generator.sample_noise(batch_size).to(device, non_blocking=non_blocking)
            fake_labels = generator.sample_labels(batch_size).to(device, non_blocking=non_blocking)
            fake_images = generator(noise, fake_labels)
            
            # 判别器输出
            d_real_validity = discriminator.get_validity(real_images)
            d_fake_validity = discriminator.get_validity(fake_images.detach())
            
            # WGAN 损失
            d_loss_adv = -torch.mean(d_real_validity) + torch.mean(d_fake_validity)
            
            # 梯度惩罚
            gp_loss = compute_gradient_penalty(
                discriminator, real_images, fake_images.detach(), device
            )
            
            # 分类损失
            _, real_class_logits, _ = discriminator(real_images)
            _, fake_class_logits, _ = discriminator(fake_images.detach())
            
            class_loss_real = nn.CrossEntropyLoss()(real_class_logits, real_labels)
            class_loss_fake = nn.CrossEntropyLoss()(fake_class_logits, fake_labels)
            class_loss = (class_loss_real + class_loss_fake) / 2
            
            # 总判别器损失
            d_loss = d_loss_adv + config['pretrain']['gp_lambda'] * gp_loss + config['pretrain']['class_lambda'] * class_loss
            
            d_loss.backward()
            opt_d.step()
            
            # GPU内存优化：释放中间变量
            if device.type == 'cuda':
                del noise, fake_labels, fake_images, d_real_validity, d_fake_validity
                torch.cuda.empty_cache()
        
        # 训练生成器
        opt_g.zero_grad()
        
        # 生成假样本
        noise = generator.sample_noise(batch_size).to(device, non_blocking=non_blocking)
        fake_labels = generator.sample_labels(batch_size).to(device, non_blocking=non_blocking)
        fake_images = generator(noise, fake_labels)
        
        # 判别器输出
        d_fake_validity = discriminator.get_validity(fake_images)
        _, fake_class_logits, _ = discriminator(fake_images)
        
        # 生成器损失
        g_loss_adv = -torch.mean(d_fake_validity)
        g_loss_class = nn.CrossEntropyLoss()(fake_class_logits, fake_labels)
        g_loss = g_loss_adv + config['pretrain']['class_lambda'] * g_loss_class
        
        g_loss.backward()
        opt_g.step()
        
        # GPU内存优化：释放中间变量
        if device.type == 'cuda':
            del noise, fake_labels, fake_images, d_fake_validity, fake_class_logits
            torch.cuda.empty_cache()
        
        # 记录损失
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        total_gp_loss += gp_loss.item()
        total_class_loss += class_loss.item()
        
        # 更新进度条（中文显示）
        progress_bar.set_postfix({
            '生成器损失': f'{g_loss.item():.4f}',
            '判别器损失': f'{d_loss.item():.4f}',
            '梯度惩罚': f'{gp_loss.item():.4f}',
            '分类损失': f'{class_loss.item():.4f}'
        })
    
    progress_bar.close()
    
    # 计算平均损失
    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)
    avg_gp_loss = total_gp_loss / len(dataloader)
    avg_class_loss = total_class_loss / len(dataloader)
    
    return avg_g_loss, avg_d_loss, avg_gp_loss, avg_class_loss


def evaluate(generator, discriminator, device, config, epoch, logger, dataloader=None):
    """评估模型"""
    generator.eval()
    discriminator.eval()
    
    with torch.no_grad():
        # 生成样本
        num_samples = min(64, config['metrics']['eval_num_images'])
        noise = generator.sample_noise(num_samples).to(device)
        labels = generator.sample_labels(num_samples).to(device)
        fake_images = generator(noise, labels)
        
        # 保存样本网格
        save_dir = os.path.join(config['logging']['save_dir'], 'samples')
        os.makedirs(save_dir, exist_ok=True)
        
        # 反标准化图像
        fake_images_denorm = denormalize_images(fake_images)
        save_grid(fake_images_denorm, nrow=8, 
                 filename=os.path.join(save_dir, f'sample_epoch_{epoch+1:04d}.png'))
        
        # 记录图像到 TensorBoard
        logger.log_images('generated_samples', fake_images_denorm, epoch)
        
        # 计算简单指标
        d_fake_validity = discriminator.get_validity(fake_images)
        avg_fake_prob = torch.sigmoid(d_fake_validity).mean().item()
        
        # 如果有数据加载器，计算高级指标
        metrics_results = {'fake_prob': avg_fake_prob}
        
        if dataloader is not None and config['metrics']['enable_advanced_metrics']:
            try:
                from metrics import MetricsCalculator
                
                # 收集真实图像和标签
                real_images_list = []
                real_labels_list = []
                
                for batch_idx, (real_images, real_labels) in enumerate(dataloader):
                    if batch_idx >= config['metrics']['eval_batches']:
                        break
                    real_images_list.append(real_images)
                    real_labels_list.append(real_labels)
                
                real_images_all = torch.cat(real_images_list, dim=0)
                real_labels_all = torch.cat(real_labels_list, dim=0)
                
                # 生成更多样本用于指标计算
                num_metrics_samples = min(config['metrics']['eval_num_images'], 1000)
                noise_metrics = generator.sample_noise(num_metrics_samples).to(device)
                labels_metrics = generator.sample_labels(num_metrics_samples).to(device)
                fake_images_metrics = generator(noise_metrics, labels_metrics)
                
                # 初始化指标计算器
                metrics_calculator = MetricsCalculator(device=device, num_classes=config['model']['num_classes'])
                
                # 计算批次指标（轻量级）
                batch_metrics = metrics_calculator.calculate_batch_metrics(
                    real_images_all[:num_metrics_samples].to(device),
                    fake_images_metrics,
                    real_labels_all[:num_metrics_samples].to(device),
                    labels_metrics
                )
                
                metrics_results.update(batch_metrics)
                
                # 定期计算完整指标（减少计算开销）
                if epoch % config['metrics']['full_metrics_interval'] == 0:
                    full_metrics = metrics_calculator.calculate_all_metrics(
                        real_images_all.to(device),
                        fake_images_metrics,
                        real_labels_all.to(device),
                        labels_metrics
                    )
                    metrics_results.update(full_metrics)
                    
                    # 记录完整指标
                    for metric_name, metric_value in full_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            logger.log_scalar(f'eval/{metric_name}', metric_value, epoch)
                        elif isinstance(metric_value, list):
                            for i, class_value in enumerate(metric_value):
                                logger.log_scalar(f'eval/{metric_name}_class_{i}', class_value, epoch)
                
                # 记录批次指标
                for metric_name, metric_value in batch_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        logger.log_scalar(f'eval/{metric_name}', metric_value, epoch)
                    elif isinstance(metric_value, list):
                        for i, class_value in enumerate(metric_value):
                            logger.log_scalar(f'eval/{metric_name}_class_{i}', class_value, epoch)
                            
            except ImportError as e:
                print(f"指标计算模块导入失败: {e}")
            except Exception as e:
                print(f"指标计算失败: {e}")
        
        return metrics_results


def save_checkpoint(generator, discriminator, opt_g, opt_d, epoch, config, logger):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'optimizer_d_state_dict': opt_d.state_dict(),
        'config': config
    }
    
    checkpoint_dir = os.path.join(config['logging']['save_dir'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
    torch.save(checkpoint, checkpoint_file)
    
    # 保存最佳模型
    if epoch == config['pretrain']['epochs'] - 1:
        best_model_file = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_model_file)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ACGAN 预训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--experiment', type=str, default=None, help='实验名称')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    setup_seed(config['seed'])
    
    # 获取设备
    device = get_device(config)
    
    # 设置日志记录
    from utils.logger import setup_logging
    logger = setup_logging(config, args.experiment)
    
    # 创建数据加载器
    dataloader, _, _ = get_cifar10_dataloader(
        data_root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        image_size=config['data']['image_size']
    )
    
    # 创建模型
    generator, discriminator = create_models(config, device)
    
    # 创建优化器
    opt_g, opt_d = create_optimizers(generator, discriminator, config)
    
    print(f"开始 ACGAN 预训练，共 {config['pretrain']['epochs']} 个 epoch")
    print(f"生成器参数: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # 训练历史
    history = {
        'g_loss': [],
        'd_loss': [],
        'gp_loss': [],
        'class_loss': [],
        'fake_prob': []
    }
    
    # 训练循环
    for epoch in range(config['pretrain']['epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['pretrain']['epochs']} ===")
        
        # 训练
        g_loss, d_loss, gp_loss, class_loss = train_epoch(
            generator, discriminator, opt_g, opt_d, dataloader, device, config, epoch, logger
        )
        
        # 评估
        metrics_results = evaluate(generator, discriminator, device, config, epoch, logger, dataloader)
        
        # 记录指标
        logger.log_scalar('train/g_loss', g_loss, epoch)
        logger.log_scalar('train/d_loss', d_loss, epoch)
        logger.log_scalar('train/gp_loss', gp_loss, epoch)
        logger.log_scalar('train/class_loss', class_loss, epoch)
        logger.log_scalar('eval/fake_prob', metrics_results['fake_prob'], epoch)
        
        # 保存历史
        history['g_loss'].append(g_loss)
        history['d_loss'].append(d_loss)
        history['gp_loss'].append(gp_loss)
        history['class_loss'].append(class_loss)
        history['fake_prob'].append(metrics_results['fake_prob'])
        
        # 打印进度
        print(f"生成器损失: {g_loss:.4f}")
        print(f"判别器损失: {d_loss:.4f}")
        print(f"梯度惩罚: {gp_loss:.4f}")
        print(f"分类损失: {class_loss:.4f}")
        print(f"假样本概率: {metrics_results['fake_prob']:.4f}")
        
        # 打印高级指标（如果可用）
        if 'lpips' in metrics_results:
            print(f"LPIPS距离: {metrics_results['lpips']:.4f}")
        if 'kl_divergence' in metrics_results:
            print(f"KL散度: {metrics_results['kl_divergence']:.4f}")
        if 'kid' in metrics_results:
            print(f"KID分数: {metrics_results['kid']:.4f}")
        if 'fid' in metrics_results:
            print(f"FID分数: {metrics_results['fid']:.4f}")
        
        # 保存检查点
        if (epoch + 1) % config['logging']['eval_interval'] == 0:
            save_checkpoint(generator, discriminator, opt_g, opt_d, epoch, config, logger)
        
        # 刷新日志
        logger.flush()
    
    # 保存最终模型
    save_checkpoint(generator, discriminator, opt_g, opt_d, config['pretrain']['epochs'] - 1, config, logger)
    
    # 绘制训练曲线
    plot_training_curves(history, save_dir=config['logging']['save_dir'])
    
    # 关闭日志
    logger.close()
    
    print("\n🎉 ACGAN 预训练完成！")


if __name__ == "__main__":
    main()