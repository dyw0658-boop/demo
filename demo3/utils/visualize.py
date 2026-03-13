"""
可视化工具模块
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def save_grid(images, nrow=8, filename='grid.png', normalize=True, value_range=(-1, 1)):
    """
    保存图像网格
    
    Args:
        images: 图像张量，shape [N, C, H, W]
        nrow: 每行图像数
        filename: 保存文件名
        normalize: 是否标准化
        value_range: 值域范围
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # 创建网格
    grid = torchvision.utils.make_grid(
        images, nrow=nrow, normalize=normalize, value_range=value_range
    )
    
    # 转换为 PIL 图像并保存
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
    # 保存图像
    Image.fromarray(grid).save(filename)
    print(f"图像网格已保存到: {filename}")


def plot_training_curves(metrics_dict, save_path='training_curves.png'):
    """
    绘制训练曲线
    
    Args:
        metrics_dict: 指标字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 绘制不同指标
    plots = [
        ('g_loss', '生成器损失'),
        ('d_loss', '判别器损失'),
        ('fid', 'FID 分数'),
        ('inception_score', 'Inception Score'),
        ('cms', 'CMS 分数'),
        ('reward', 'PPO 奖励')
    ]
    
    for i, (key, title) in enumerate(plots):
        if key in metrics_dict:
            values = metrics_dict[key]
            axes[i].plot(values)
            axes[i].set_title(title)
            axes[i].set_xlabel('Epoch')
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


def visualize_generated_images(generator, num_images=64, nrow=8, filename='generated.png', device='cpu'):
    """
    可视化生成的图像
    
    Args:
        generator: 生成器模型
        num_images: 图像数量
        nrow: 每行图像数
        filename: 保存文件名
        device: 设备
    """
    generator.eval()
    
    with torch.no_grad():
        # 生成图像
        images, labels = generator.generate(num_images, device)
        
        # 保存网格
        save_grid(images, nrow=nrow, filename=filename)
    
    return images, labels


def compare_real_fake(real_images, fake_images, nrow=8, filename='comparison.png'):
    """
    比较真实和生成图像
    
    Args:
        real_images: 真实图像
        fake_images: 生成图像
        nrow: 每行图像数
        filename: 保存文件名
    """
    # 确保数量相同
    num_images = min(len(real_images), len(fake_images))
    real_images = real_images[:num_images]
    fake_images = fake_images[:num_images]
    
    # 交错排列真实和生成图像
    interleaved = []
    for i in range(num_images):
        interleaved.append(real_images[i])
        interleaved.append(fake_images[i])
    
    interleaved = torch.stack(interleaved)
    
    # 保存网格
    save_grid(interleaved, nrow=nrow*2, filename=filename)


def plot_class_distribution(labels, num_classes=10, filename='class_distribution.png'):
    """
    绘制类别分布
    
    Args:
        labels: 标签
        num_classes: 类别数量
        filename: 保存文件名
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # 计算类别计数
    counts = np.bincount(labels, minlength=num_classes)
    
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename)
    plt.close()
    print(f"类别分布图已保存到: {filename}")


def create_attention_visualization(attention_weights, images, filename='attention.png'):
    """
    可视化注意力权重
    
    Args:
        attention_weights: 注意力权重，shape [B, H*W, H*W]
        images: 原始图像，shape [B, C, H, W]
        filename: 保存文件名
    """
    # 取第一个样本进行可视化
    attn = attention_weights[0].cpu().numpy()
    image = images[0].cpu()
    
    # 反标准化图像
    image = (image * 0.5) + 0.5  # [-1, 1] -> [0, 1]
    image = image.permute(1, 2, 0).numpy()
    
    # 计算平均注意力权重
    avg_attn = attn.mean(axis=0).reshape(int(np.sqrt(len(attn))), -1)
    
    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原始图像
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # 注意力热图
    im = ax2.imshow(avg_attn, cmap='hot', interpolation='nearest')
    ax2.set_title('Attention Heatmap')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"注意力可视化已保存到: {filename}")


def save_metrics_to_csv(metrics_dict, filename='training_metrics.csv'):
    """
    保存指标到 CSV 文件
    
    Args:
        metrics_dict: 指标字典
        filename: 保存文件名
    """
    import pandas as pd
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # 转换为 DataFrame
    df = pd.DataFrame(metrics_dict)
    
    # 保存到 CSV
    df.to_csv(filename, index=False)
    print(f"指标已保存到: {filename}")


def create_training_summary(metrics_dict, generator, test_images, filename='summary.png'):
    """
    创建训练总结
    
    Args:
        metrics_dict: 指标字典
        generator: 生成器模型
        test_images: 测试图像
        filename: 保存文件名
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 训练曲线
    ax1 = plt.subplot(2, 3, 1)
    if 'g_loss' in metrics_dict:
        ax1.plot(metrics_dict['g_loss'], label='Generator Loss')
    if 'd_loss' in metrics_dict:
        ax1.plot(metrics_dict['d_loss'], label='Discriminator Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 质量指标
    ax2 = plt.subplot(2, 3, 2)
    if 'fid' in metrics_dict:
        ax2.plot(metrics_dict['fid'], label='FID')
    if 'inception_score' in metrics_dict:
        ax2.plot(metrics_dict['inception_score'], label='Inception Score')
    ax2.set_title('Quality Metrics')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 生成图像样本
    ax3 = plt.subplot(2, 3, 3)
    with torch.no_grad():
        sample_images, _ = generator.generate(16, device=next(generator.parameters()).device)
        grid = torchvision.utils.make_grid(sample_images, nrow=4, normalize=True)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        ax3.imshow(grid)
        ax3.set_title('Generated Samples')
        ax3.axis('off')
    
    # 4. 真实图像样本
    ax4 = plt.subplot(2, 3, 4)
    real_samples = test_images[:16]
    grid = torchvision.utils.make_grid(real_samples, nrow=4, normalize=True)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    ax4.imshow(grid)
    ax4.set_title('Real Samples')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"训练总结已保存到: {filename}")


if __name__ == "__main__":
    # 测试代码
    # 生成随机图像
    test_images = torch.randn(64, 3, 32, 32)
    test_images = torch.clamp(test_images, -1, 1)
    
    # 测试保存网格
    save_grid(test_images, nrow=8, filename='test_grid.png')
    print("可视化工具测试完成")