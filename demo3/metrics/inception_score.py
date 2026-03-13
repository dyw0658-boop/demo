"""
Inception Score 计算模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy


class InceptionScore(nn.Module):
    """
    Inception Score 计算器
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        try:
            # 尝试加载预训练的 InceptionV3
            self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=pretrained)
        except Exception as e:
            print(f"无法加载预训练Inception模型: {e}")
            print("使用随机初始化的Inception模型")
            # 使用随机初始化的模型
            self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)
        
        # 移除最后的分类层，保留 softmax
        self.inception.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.Softmax(dim=1)
        )
        self.inception.aux_logits = False
        
        # 设置为评估模式
        self.inception.eval()
        
        # 冻结所有参数
        for param in self.inception.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        计算图像的概率分布
        
        Args:
            x: 输入图像，shape [B, 3, H, W]，值域 [0, 1]
        
        Returns:
            probs: 类别概率分布，shape [B, 1000]
        """
        # 确保输入在 [0, 1] 范围内
        if x.min() < 0 or x.max() > 1:
            x = torch.clamp(x, 0, 1)
        
        # 调整大小为 299x299 (InceptionV3 输入大小)
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 标准化 (ImageNet 统计)
        x = (x - 0.5) * 2  # [0, 1] -> [-1, 1]
        
        # 计算概率
        with torch.no_grad():
            probs = self.inception(x)
        
        return probs


def calculate_inception_score(probs, splits=10):
    """
    计算 Inception Score
    
    Args:
        probs: 类别概率分布，shape [N, 1000]
        splits: 分割数
    
    Returns:
        is_mean: Inception Score 均值
        is_std: Inception Score 标准差
    """
    # 转换为 numpy
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    
    N = probs.shape[0]
    
    # 分割数据
    split_scores = []
    
    for i in range(splits):
        # 获取当前分割
        start = i * (N // splits)
        end = (i + 1) * (N // splits)
        
        if i == splits - 1:
            # 最后一个分割包含剩余数据
            end = N
        
        split_probs = probs[start:end]
        
        # 计算边际分布
        marginal = np.mean(split_probs, axis=0)
        
        # 计算 KL 散度
        kl_divergences = []
        for p in split_probs:
            kl = entropy(p, marginal)
            kl_divergences.append(kl)
        
        # 计算当前分割的分数
        split_score = np.exp(np.mean(kl_divergences))
        split_scores.append(split_score)
    
    # 计算均值和标准差
    is_mean = np.mean(split_scores)
    is_std = np.std(split_scores)
    
    return is_mean, is_std


def compute_inception_score(images, batch_size=50, splits=10, device='cpu'):
    """
    计算 Inception Score
    
    Args:
        images: 输入图像，shape [N, 3, H, W]
        batch_size: 批量大小
        splits: 分割数
        device: 设备
    
    Returns:
        is_mean: Inception Score 均值
        is_std: Inception Score 标准差
    """
    # 创建 Inception Score 计算器
    is_calculator = InceptionScore().to(device)
    is_calculator.eval()
    
    # 计算概率分布
    all_probs = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        probs = is_calculator(batch)
        all_probs.append(probs.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)
    
    # 计算 Inception Score
    is_mean, is_std = calculate_inception_score(all_probs, splits=splits)
    
    return is_mean, is_std


def inception_score_from_discriminator(class_probs):
    """
    使用判别器分类概率计算类间多样性
    
    Args:
        class_probs: 判别器分类概率，shape [N, num_classes]
    
    Returns:
        diversity_score: 多样性分数
    """
    if isinstance(class_probs, torch.Tensor):
        class_probs = class_probs.cpu().numpy()
    
    # 计算边际分布
    marginal = np.mean(class_probs, axis=0)
    
    # 计算 KL 散度
    kl_divergences = []
    for p in class_probs:
        kl = entropy(p, marginal)
        kl_divergences.append(kl)
    
    # 计算多样性分数
    diversity_score = np.exp(np.mean(kl_divergences))
    
    return diversity_score


if __name__ == "__main__":
    # 测试代码
    # 生成随机概率分布进行测试
    test_probs = torch.randn(1000, 1000).softmax(dim=1)
    
    is_mean, is_std = calculate_inception_score(test_probs)
    print(f"测试 Inception Score: {is_mean:.4f} ± {is_std:.4f}")