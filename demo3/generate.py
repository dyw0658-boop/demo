"""
图像生成脚本
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision.utils import save_image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.generator import Generator
from utils.visualize import save_grid, denormalize_images


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


def load_generator(config, device, checkpoint_path=None):
    """加载生成器并进行GPU优化"""
    # 如果使用GPU，设置模型为数据并行
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行推理")
        
        generator = Generator(
            latent_dim=config['model']['latent_dim'],
            num_classes=config['model']['num_classes'],
            ngf=config['model']['ngf']
        )
        generator = nn.DataParallel(generator).to(device)
    else:
        generator = Generator(
            latent_dim=config['model']['latent_dim'],
            num_classes=config['model']['num_classes'],
            ngf=config['model']['ngf']
        ).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'generator_state_dict' in checkpoint:
            # 处理数据并行模型的权重
            if 'module.' in list(checkpoint['generator_state_dict'].keys())[0]:
                # 权重已经是数据并行格式
                generator.load_state_dict(checkpoint['generator_state_dict'])
            else:
                # 权重需要转换为数据并行格式
                if isinstance(generator, nn.DataParallel):
                    from collections import OrderedDict
                    
                    # 为生成器添加module前缀
                    generator_state_dict = OrderedDict()
                    for k, v in checkpoint['generator_state_dict'].items():
                        generator_state_dict['module.' + k] = v
                    generator.load_state_dict(generator_state_dict)
                else:
                    generator.load_state_dict(checkpoint['generator_state_dict'])
            
            print(f"加载生成器权重: {checkpoint_path}")
            
            if 'epoch' in checkpoint:
                print(f"检查点 epoch: {checkpoint['epoch']}")
            elif 'update_step' in checkpoint:
                print(f"检查点更新步数: {checkpoint['update_step']}")
        else:
            print("警告: 检查点中未找到生成器状态字典")
    else:
        print("警告: 使用随机初始化的生成器")
    
    generator.eval()
    
    # 打印模型参数统计
    g_params = sum(p.numel() for p in generator.parameters())
    print(f"生成器参数: {g_params:,}")
    
    return generator


def generate_images(generator, device, num_images, labels=None, save_individual=False):
    """生成图像"""
    # 设置非阻塞传输标志
    non_blocking = device.type == 'cuda'
    
    with torch.no_grad():
        # 生成噪声和标签
        noise = generator.sample_noise(num_images).to(device, non_blocking=non_blocking)
        
        if labels is None:
            labels = generator.sample_labels(num_images).to(device, non_blocking=non_blocking)
        else:
            if isinstance(labels, list):
                labels = torch.tensor(labels, device=device)
            elif isinstance(labels, int):
                labels = torch.full((num_images,), labels, device=device)
            else:
                labels = labels.to(device, non_blocking=non_blocking)
        
        # 生成图像
        images = generator(noise, labels)
        
        # 反标准化
        images_denorm = denormalize_images(images)
        
        # 清理中间变量以释放GPU内存
        if device.type == 'cuda':
            del noise, labels, images
            torch.cuda.empty_cache()
        
        return images_denorm, labels


def save_individual_images(images, labels, save_dir):
    """保存单个图像"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        filename = os.path.join(save_dir, f'image_{i:04d}_class_{label.item()}.png')
        save_image(img, filename)


def generate_class_grid(generator, device, num_classes, images_per_class, save_dir):
    """生成类别网格"""
    os.makedirs(save_dir, exist_ok=True)
    
    all_images = []
    
    for class_id in range(num_classes):
        print(f"生成类别 {class_id} 的图像...")
        
        # 生成当前类别的图像
        images, labels = generate_images(generator, device, images_per_class, class_id)
        all_images.append(images)
    
    # 合并所有图像
    all_images = torch.cat(all_images, dim=0)
    
    # 保存类别网格
    grid_filename = os.path.join(save_dir, 'class_grid.png')
    save_grid(all_images, nrow=images_per_class, filename=grid_filename)
    
    print(f"类别网格已保存: {grid_filename}")


def compute_metrics(generator, discriminator, device, num_images):
    """计算指标"""
    try:
        from metrics.fid import compute_fid_score
        from metrics.inception_score import compute_inception_score
        from metrics.cms import compute_cms_score
        
        print("计算指标...")
        
        # 生成假样本
        fake_images, labels = generate_images(generator, device, num_images)
        
        # 计算 FID（需要真实样本，这里使用随机样本作为演示）
        real_images = torch.randn_like(fake_images)
        fid_score = compute_fid_score(discriminator, real_images, fake_images)
        
        # 计算 Inception Score
        is_mean, is_std = compute_inception_score(fake_images)
        
        # 计算 CMS
        cms_score = compute_cms_score(discriminator, fake_images, labels)
        
        print(f"FID: {fid_score:.4f}")
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        print(f"CMS: {cms_score:.4f}")
        
        return {
            'fid': fid_score,
            'is_mean': is_mean,
            'is_std': is_std,
            'cms': cms_score
        }
    
    except Exception as e:
        print(f"指标计算失败: {e}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像生成')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--num-images', type=int, default=64, help='生成图像数量')
    parser.add_argument('--output-dir', type=str, default='./outputs/generated', help='输出目录')
    parser.add_argument('--class-id', type=int, default=None, help='指定类别ID（生成特定类别）')
    parser.add_argument('--labels', type=str, default=None, help='指定标签列表（逗号分隔）')
    parser.add_argument('--save-individual', action='store_true', help='保存单个图像')
    parser.add_argument('--class-grid', action='store_true', help='生成类别网格')
    parser.add_argument('--compute-metrics', action='store_true', help='计算指标')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    setup_seed(config['seed'])
    
    # 获取设备
    device = get_device(config)
    
    # 设置非阻塞传输标志
    non_blocking = device.type == 'cuda'
    
    # 清理GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("清理GPU缓存完成")
    
    # 加载生成器
    generator = load_generator(config, device, args.checkpoint)
    
    # 如果需要计算指标，加载判别器
    discriminator = None
    if args.compute_metrics:
        from models.discriminator import Discriminator
        discriminator = Discriminator(
            ndf=config['model']['ndf'],
            num_classes=config['model']['num_classes']
        ).to(device)
        
        # 尝试加载判别器权重
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'discriminator_state_dict' in checkpoint:
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                discriminator.eval()
                print("加载判别器权重")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理标签
    labels = None
    if args.labels:
        labels = [int(x) for x in args.labels.split(',')]
        if len(labels) != args.num_images:
            print(f"警告: 标签数量 ({len(labels)}) 与图像数量 ({args.num_images}) 不匹配")
            labels = labels[:args.num_images]  # 截断或填充
            if len(labels) < args.num_images:
                labels.extend([labels[-1]] * (args.num_images - len(labels)))
    elif args.class_id is not None:
        labels = args.class_id
    
    # 生成类别网格
    if args.class_grid:
        generate_class_grid(
            generator, device, 
            config['model']['num_classes'], 
            args.num_images // config['model']['num_classes'],
            args.output_dir
        )
        return
    
    # 生成图像
    print(f"生成 {args.num_images} 张图像...")
    images, generated_labels = generate_images(generator, device, args.num_images, labels)
    
    # 保存图像网格
    grid_filename = os.path.join(args.output_dir, 'generated_grid.png')
    save_grid(images, nrow=8, filename=grid_filename)
    print(f"图像网格已保存: {grid_filename}")
    
    # 保存单个图像
    if args.save_individual:
        individual_dir = os.path.join(args.output_dir, 'individual')
        save_individual_images(images, generated_labels, individual_dir)
        print(f"单个图像已保存到: {individual_dir}")
    
    # 计算指标
    if args.compute_metrics and discriminator is not None:
        metrics = compute_metrics(generator, discriminator, device, min(args.num_images, 1000))
        
        # 保存指标到文件
        if metrics:
            metrics_file = os.path.join(args.output_dir, 'metrics.txt')
            with open(metrics_file, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            print(f"指标已保存: {metrics_file}")
    
    # 打印生成信息
    print(f"\n生成完成！")
    print(f"图像尺寸: {images.shape[2]}x{images.shape[3]}")
    print(f"图像范围: [{images.min():.3f}, {images.max():.3f}]")
    
    if labels is not None:
        unique_labels = torch.unique(generated_labels)
        print(f"生成的类别: {unique_labels.tolist()}")
    
    # 保存生成配置
    config_file = os.path.join(args.output_dir, 'generation_config.txt')
    with open(config_file, 'w') as f:
        f.write(f"检查点: {args.checkpoint}\n")
        f.write(f"图像数量: {args.num_images}\n")
        f.write(f"设备: {device}\n")
        f.write(f"随机种子: {config['seed']}\n")
        if labels is not None:
            f.write(f"标签: {labels}\n")
    
    print(f"生成配置已保存: {config_file}")


if __name__ == "__main__":
    main()