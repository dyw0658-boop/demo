"""
FID (Fréchet Inception Distance) 计算模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(real_features, fake_features, eps=1e-6):
    """
    计算 FID 分数
    
    Args:
        real_features: 真实图像特征，shape [N, feature_dim]
        fake_features: 生成图像特征，shape [M, feature_dim]
        eps: 数值稳定性参数
    
    Returns:
        fid_score: FID 分数
    """
    # 转换为 numpy
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.cpu().numpy()
    if isinstance(fake_features, torch.Tensor):
        fake_features = fake_features.cpu().numpy()
    
    # 计算均值和协方差
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # 处理协方差矩阵的数值稳定性
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_fake += eps * np.eye(sigma_fake.shape[0])
    
    # 计算平方根
    sqrt_sigma_real = sqrtm(sigma_real)
    sqrt_sigma_fake = sqrtm(sigma_fake)
    
    # 处理复数部分 (取实部)
    if np.iscomplexobj(sqrt_sigma_real):
        sqrt_sigma_real = sqrt_sigma_real.real
    if np.iscomplexobj(sqrt_sigma_fake):
        sqrt_sigma_fake = sqrt_sigma_fake.real
    
    # 计算 FID
    diff = mu_real - mu_fake
    cov_mean = sqrtm(sigma_real.dot(sigma_fake))
    
    # 处理复数部分
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    
    # 计算最终分数
    fid_score = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * cov_mean)
    
    return fid_score


class InceptionV3FeatureExtractor(nn.Module):
    """
    InceptionV3 特征提取器
    """
    
    def __init__(self):
        super().__init__()
        
        # 加载预训练的 InceptionV3
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        
        # 移除最后的分类层
        self.inception.fc = nn.Identity()
        self.inception.aux_logits = False
        
        # 设置为评估模式
        self.inception.eval()
        
        # 冻结所有参数
        for param in self.inception.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        提取特征
        
        Args:
            x: 输入图像，shape [B, 3, H, W]，值域 [0, 1]
        
        Returns:
            features: 特征向量，shape [B, 2048]
        """
        # 确保输入在 [0, 1] 范围内
        if x.min() < 0 or x.max() > 1:
            x = torch.clamp(x, 0, 1)
        
        # 调整大小为 299x299 (InceptionV3 输入大小)
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 标准化 (ImageNet 统计)
        x = (x - 0.5) * 2  # [0, 1] -> [-1, 1]
        
        # 提取特征
        with torch.no_grad():
            features = self.inception(x)
        
        return features


def compute_fid_score(real_images, fake_images, batch_size=50, device='cpu'):
    """
    计算 FID 分数
    
    Args:
        real_images: 真实图像，shape [N, 3, H, W]
        fake_images: 生成图像，shape [M, 3, H, W]
        batch_size: 批量大小
        device: 设备
    
    Returns:
        fid_score: FID 分数
    """
    # 创建特征提取器
    feature_extractor = InceptionV3FeatureExtractor().to(device)
    feature_extractor.eval()
    
    # 提取真实图像特征
    real_features = []
    for i in range(0, len(real_images), batch_size):
        batch = real_images[i:i+batch_size].to(device)
        features = feature_extractor(batch)
        real_features.append(features.cpu())
    
    real_features = torch.cat(real_features, dim=0)
    
    # 提取生成图像特征
    fake_features = []
    for i in range(0, len(fake_images), batch_size):
        batch = fake_images[i:i+batch_size].to(device)
        features = feature_extractor(batch)
        fake_features.append(features.cpu())
    
    fake_features = torch.cat(fake_features, dim=0)
    
    # 计算 FID
    fid_score = calculate_fid(real_features, fake_features)
    
    return fid_score


def fid_from_discriminator(real_features, fake_features):
    """
    使用判别器特征计算 FID
    
    Args:
        real_features: 真实图像判别器特征
        fake_features: 生成图像判别器特征
    
    Returns:
        fid_score: FID 分数
    """
    return calculate_fid(real_features, fake_features)


if __name__ == "__main__":
    # 测试代码
    # 生成随机特征进行测试
    real_feat = torch.randn(1000, 2048)
    fake_feat = torch.randn(1000, 2048)
    
    fid = calculate_fid(real_feat, fake_feat)
    print(f"测试 FID 分数: {fid:.4f}")