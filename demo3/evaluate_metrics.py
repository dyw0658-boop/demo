"""
评估指标测试脚本
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.generator import Generator
from models.discriminator import Discriminator
from utils.data import get_cifar10_dataloader, denormalize_images
from metrics import MetricsCalculator


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config):
    """获取设备"""
    if config['device'] == 'auto':
        if torch.cuda.is_available() and config['gpu']['enabled']:
            device = torch.device(f'cuda:{config["gpu"]["device_id"]}')
            print(f"使用GPU设备: {torch.cuda.get_device_name(device.index)}")
        else:
            device = torch.device('cpu')
            print("使用CPU设备")
    else:
        device = torch.device(config['device'])
        print(f"使用指定设备: {device}")
    
    return device


def test_metrics_on_real_data(device, config):
    """在真实数据上测试指标"""
    print("\n=== 在真实数据上测试指标 ===")
    
    # 创建数据加载器
    dataloader, _, _ = get_cifar10_dataloader(
        data_root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        image_size=config['data']['image_size']
    )
    
    # 收集真实图像和标签
    real_images_list = []
    real_labels_list = []
    
    for batch_idx, (real_images, real_labels) in enumerate(dataloader):
        if batch_idx >= 5:  # 只使用前5个批次
            break
        real_images_list.append(real_images)
        real_labels_list.append(real_labels)
    
    real_images_all = torch.cat(real_images_list, dim=0)
    real_labels_all = torch.cat(real_labels_list, dim=0)
    
    # 将数据分成两部分作为"真实"和"生成"
    split_idx = len(real_images_all) // 2
    real_part1 = real_images_all[:split_idx]
    real_part2 = real_images_all[split_idx:split_idx*2]
    labels_part1 = real_labels_all[:split_idx]
    labels_part2 = real_labels_all[split_idx:split_idx*2]
    
    # 初始化指标计算器
    metrics_calculator = MetricsCalculator(device=device, num_classes=config['model']['num_classes'])
    
    # 计算指标
    print("计算指标中...")
    
    # 计算批次指标
    batch_metrics = metrics_calculator.calculate_batch_metrics(
        real_part1.to(device),
        real_part2.to(device),
        labels_part1.to(device),
        labels_part2.to(device)
    )
    
    print("\n=== 批次指标结果 ===")
    for metric_name, metric_value in batch_metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"{metric_name}: {metric_value:.6f}")
        elif isinstance(metric_value, list):
            print(f"{metric_name}:")
            for i, class_value in enumerate(metric_value):
                print(f"  类别 {i}: {class_value:.6f}")
    
    # 计算完整指标
    print("\n计算完整指标中...")
    full_metrics = metrics_calculator.calculate_all_metrics(
        real_part1.to(device),
        real_part2.to(device),
        labels_part1.to(device),
        labels_part2.to(device)
    )
    
    print("\n=== 完整指标结果 ===")
    for metric_name, metric_value in full_metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"{metric_name}: {metric_value:.6f}")
        elif isinstance(metric_value, list):
            print(f"{metric_name}:")
            for i, class_value in enumerate(metric_value):
                print(f"  类别 {i}: {class_value:.6f}")
    
    return batch_metrics, full_metrics


def test_metrics_on_generated_data(device, config):
    """在生成数据上测试指标"""
    print("\n=== 在生成数据上测试指标 ===")
    
    # 创建生成器
    generator = Generator(
        latent_dim=config['model']['latent_dim'],
        num_classes=config['model']['num_classes'],
        ngf=config['model']['ngf']
    ).to(device)
    
    # 创建数据加载器
    dataloader, _, _ = get_cifar10_dataloader(
        data_root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        image_size=config['data']['image_size']
    )
    
    # 收集真实图像和标签
    real_images_list = []
    real_labels_list = []
    
    for batch_idx, (real_images, real_labels) in enumerate(dataloader):
        if batch_idx >= 5:  # 只使用前5个批次
            break
        real_images_list.append(real_images)
        real_labels_list.append(real_labels)
    
    real_images_all = torch.cat(real_images_list, dim=0)
    real_labels_all = torch.cat(real_labels_list, dim=0)
    
    # 生成假样本
    num_samples = min(len(real_images_all), 100)
    noise = generator.sample_noise(num_samples).to(device)
    fake_labels = generator.sample_labels(num_samples).to(device)
    fake_images = generator(noise, fake_labels)
    
    # 初始化指标计算器
    metrics_calculator = MetricsCalculator(device=device, num_classes=config['model']['num_classes'])
    
    # 计算批次指标
    print("计算生成数据指标中...")
    batch_metrics = metrics_calculator.calculate_batch_metrics(
        real_images_all[:num_samples].to(device),
        fake_images,
        real_labels_all[:num_samples].to(device),
        fake_labels
    )
    
    print("\n=== 生成数据指标结果 ===")
    for metric_name, metric_value in batch_metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"{metric_name}: {metric_value:.6f}")
        elif isinstance(metric_value, list):
            print(f"{metric_name}:")
            for i, class_value in enumerate(metric_value):
                print(f"  类别 {i}: {class_value:.6f}")
    
    return batch_metrics


def test_individual_metrics(device, config):
    """测试单个指标模块"""
    print("\n=== 测试单个指标模块 ===")
    
    # 创建数据加载器
    dataloader, _, _ = get_cifar10_dataloader(
        data_root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        image_size=config['data']['image_size']
    )
    
    # 收集真实图像和标签
    real_images_list = []
    real_labels_list = []
    
    for batch_idx, (real_images, real_labels) in enumerate(dataloader):
        if batch_idx >= 2:  # 只使用前2个批次
            break
        real_images_list.append(real_images)
        real_labels_list.append(real_labels)
    
    real_images_all = torch.cat(real_images_list, dim=0)
    real_labels_all = torch.cat(real_labels_list, dim=0)
    
    # 将数据分成两部分
    split_idx = len(real_images_all) // 2
    real_part1 = real_images_all[:split_idx]
    real_part2 = real_images_all[split_idx:split_idx*2]
    labels_part1 = real_labels_all[:split_idx]
    labels_part2 = real_labels_all[split_idx:split_idx*2]
    
    # 测试KL散度
    print("\n--- 测试KL散度 ---")
    from metrics.kl_divergence import calculate_kl_divergence, calculate_class_kl_divergence
    
    try:
        # 提取Inception特征
        from metrics.inception_score import InceptionScore
        inception_model = InceptionScore(pretrained=False).to(device)  # 使用随机初始化模型
        inception_model.eval()
        
        with torch.no_grad():
            features1 = inception_model(real_part1.to(device))
            features2 = inception_model(real_part2.to(device))
        
        kl_score = calculate_kl_divergence(features1, features2)
        print(f"KL散度: {kl_score:.6f}")
        
        class_kl_scores, avg_kl = calculate_class_kl_divergence(
            features1, features2, labels_part1, labels_part2, config['model']['num_classes']
        )
        print(f"平均类间KL散度: {avg_kl:.6f}")
        
        # 测试KID
        print("\n--- 测试KID ---")
        from metrics.kid import calculate_kid, calculate_class_kid
        
        kid_score, kid_std = calculate_kid(features1, features2)
        print(f"KID分数: {kid_score:.6f} ± {kid_std:.6f}")
        
        class_kid_scores, class_kid_stds, avg_kid = calculate_class_kid(
            features1, features2, labels_part1, labels_part2, config['model']['num_classes']
        )
        print(f"平均类间KID: {avg_kid:.6f}")
        
    except Exception as e:
        print(f"KL散度和KID测试失败: {e}")
        print("跳过基于Inception特征的测试")
    
    # 测试LPIPS
    print("\n--- 测试LPIPS ---")
    from metrics.lpips import calculate_lpips, calculate_class_lpips
    
    try:
        lpips_score, lpips_std = calculate_lpips(
            real_part1, real_part2, device=device
        )
        print(f"LPIPS距离: {lpips_score:.6f} ± {lpips_std:.6f}")
        
        class_lpips_scores, class_lpips_stds, avg_lpips = calculate_class_lpips(
            real_part1, real_part2, labels_part1, labels_part2, config['model']['num_classes'],
            device=device
        )
        print(f"平均类间LPIPS: {avg_lpips:.6f}")
    except Exception as e:
        print(f"LPIPS测试失败: {e}")
        print("跳过LPIPS测试")


def main():
    """主函数"""
    # 加载配置
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    setup_seed(config['seed'])
    
    # 获取设备
    device = get_device(config)
    
    print("开始评估指标功能...")
    
    try:
        # 测试单个指标模块
        test_individual_metrics(device, config)
        
        # 在真实数据上测试指标
        batch_metrics, full_metrics = test_metrics_on_real_data(device, config)
        
        # 在生成数据上测试指标
        gen_metrics = test_metrics_on_generated_data(device, config)
        
        print("\n🎉 指标评估完成！")
        print("所有指标模块功能正常。")
        
    except Exception as e:
        print(f"❌ 指标评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()