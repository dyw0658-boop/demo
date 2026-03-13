"""
CMS (Class-wise Mode Score) 计算模块
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm


def calculate_cms(real_features, fake_features, real_labels, fake_labels, num_classes, eps=1e-6):
    """
    计算类间模式分数 (CMS)
    
    Args:
        real_features: 真实图像特征，shape [N, feature_dim]
        fake_features: 生成图像特征，shape [M, feature_dim]
        real_labels: 真实图像标签，shape [N]
        fake_labels: 生成图像标签，shape [M]
        num_classes: 类别数量
        eps: 数值稳定性参数
    
    Returns:
        cms_score: CMS 分数
        class_scores: 各类别分数
    """
    # 转换为 numpy
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.cpu().numpy()
    if isinstance(fake_features, torch.Tensor):
        fake_features = fake_features.cpu().numpy()
    if isinstance(real_labels, torch.Tensor):
        real_labels = real_labels.cpu().numpy()
    if isinstance(fake_labels, torch.Tensor):
        fake_labels = fake_labels.cpu().numpy()
    
    class_scores = []
    
    for c in range(num_classes):
        # 获取当前类别的真实和生成特征
        real_mask = real_labels == c
        fake_mask = fake_labels == c
        
        real_class_features = real_features[real_mask]
        fake_class_features = fake_features[fake_mask]
        
        # 跳过没有样本的类别
        if len(real_class_features) == 0 or len(fake_class_features) == 0:
            class_scores.append(0.0)
            continue
        
        # 计算类内 FID
        mu_real = np.mean(real_class_features, axis=0)
        mu_fake = np.mean(fake_class_features, axis=0)
        
        sigma_real = np.cov(real_class_features, rowvar=False)
        sigma_fake = np.cov(fake_class_features, rowvar=False)
        
        # 处理协方差矩阵的数值稳定性
        sigma_real += eps * np.eye(sigma_real.shape[0])
        sigma_fake += eps * np.eye(sigma_fake.shape[0])
        
        # 计算平方根
        sqrt_sigma_real = sqrtm(sigma_real)
        sqrt_sigma_fake = sqrtm(sigma_fake)
        
        # 处理复数部分
        if np.iscomplexobj(sqrt_sigma_real):
            sqrt_sigma_real = sqrt_sigma_real.real
        if np.iscomplexobj(sqrt_sigma_fake):
            sqrt_sigma_fake = sqrt_sigma_fake.real
        
        # 计算类内 FID
        diff = mu_real - mu_fake
        cov_mean = sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        
        class_fid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * cov_mean)
        
        # 转换为 CMS 分数 (FID 越小越好，CMS 越大越好)
        class_cms = 1.0 / (1.0 + class_fid)
        class_scores.append(class_cms)
    
    # 计算平均 CMS
    cms_score = np.mean(class_scores)
    
    return cms_score, class_scores


def compute_cms_score(real_images, fake_images, real_labels, fake_labels, 
                     discriminator, num_classes, batch_size=50, device='cpu'):
    """
    使用判别器特征计算 CMS 分数
    
    Args:
        real_images: 真实图像，shape [N, 3, H, W]
        fake_images: 生成图像，shape [M, 3, H, W]
        real_labels: 真实图像标签，shape [N]
        fake_labels: 生成图像标签，shape [M]
        discriminator: 判别器模型
        num_classes: 类别数量
        batch_size: 批量大小
        device: 设备
    
    Returns:
        cms_score: CMS 分数
        class_scores: 各类别分数
    """
    # 提取真实图像特征
    real_features = []
    for i in range(0, len(real_images), batch_size):
        batch = real_images[i:i+batch_size].to(device)
        with torch.no_grad():
            _, _, features = discriminator(batch)
        real_features.append(features.cpu())
    
    real_features = torch.cat(real_features, dim=0)
    
    # 提取生成图像特征
    fake_features = []
    for i in range(0, len(fake_images), batch_size):
        batch = fake_images[i:i+batch_size].to(device)
        with torch.no_grad():
            _, _, features = discriminator(batch)
        fake_features.append(features.cpu())
    
    fake_features = torch.cat(fake_features, dim=0)
    
    # 计算 CMS
    cms_score, class_scores = calculate_cms(
        real_features, fake_features, real_labels, fake_labels, num_classes
    )
    
    return cms_score, class_scores


def compute_diversity_metrics(features, class_probs):
    """
    计算多样性指标
    
    Args:
        features: 特征向量，shape [N, feature_dim]
        class_probs: 类别概率分布，shape [N, num_classes]
    
    Returns:
        feature_diversity: 特征空间多样性
        class_diversity: 类别分布多样性
    """
    # 特征空间多样性 (特征方差)
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    feature_variance = np.var(features, axis=0).mean()
    feature_diversity = np.log(feature_variance + 1e-8)
    
    # 类别分布多样性 (类别分布熵)
    if isinstance(class_probs, torch.Tensor):
        class_probs = class_probs.cpu().numpy()
    
    marginal_dist = np.mean(class_probs, axis=0)
    class_entropy = -np.sum(marginal_dist * np.log(marginal_dist + 1e-8))
    class_diversity = class_entropy
    
    return feature_diversity, class_diversity


def compute_all_metrics(real_images, fake_images, real_labels, fake_labels, 
                       discriminator, num_classes, batch_size=50, device='cpu'):
    """
    计算所有评估指标
    
    Args:
        real_images: 真实图像
        fake_images: 生成图像
        real_labels: 真实标签
        fake_labels: 生成标签
        discriminator: 判别器
        num_classes: 类别数量
        batch_size: 批量大小
        device: 设备
    
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 提取特征
    real_features = []
    real_class_probs = []
    
    for i in range(0, len(real_images), batch_size):
        batch = real_images[i:i+batch_size].to(device)
        with torch.no_grad():
            _, class_logits, features = discriminator(batch)
            class_probs = F.softmax(class_logits, dim=1)
        
        real_features.append(features.cpu())
        real_class_probs.append(class_probs.cpu())
    
    real_features = torch.cat(real_features, dim=0)
    real_class_probs = torch.cat(real_class_probs, dim=0)
    
    fake_features = []
    fake_class_probs = []
    
    for i in range(0, len(fake_images), batch_size):
        batch = fake_images[i:i+batch_size].to(device)
        with torch.no_grad():
            _, class_logits, features = discriminator(batch)
            class_probs = F.softmax(class_logits, dim=1)
        
        fake_features.append(features.cpu())
        fake_class_probs.append(class_probs.cpu())
    
    fake_features = torch.cat(fake_features, dim=0)
    fake_class_probs = torch.cat(fake_class_probs, dim=0)
    
    # 计算 CMS
    cms_score, class_scores = calculate_cms(
        real_features, fake_features, real_labels, fake_labels, num_classes
    )
    metrics['cms'] = cms_score
    
    # 计算多样性指标
    real_feature_div, real_class_div = compute_diversity_metrics(real_features, real_class_probs)
    fake_feature_div, fake_class_div = compute_diversity_metrics(fake_features, fake_class_probs)
    
    metrics['real_feature_diversity'] = real_feature_div
    metrics['real_class_diversity'] = real_class_div
    metrics['fake_feature_diversity'] = fake_feature_div
    metrics['fake_class_diversity'] = fake_class_div
    
    # 计算特征距离
    feature_distance = torch.norm(real_features.mean(dim=0) - fake_features.mean(dim=0)).item()
    metrics['feature_distance'] = feature_distance
    
    return metrics


if __name__ == "__main__":
    # 测试代码
    # 生成随机特征和标签进行测试
    real_feat = torch.randn(1000, 512)
    fake_feat = torch.randn(1000, 512)
    real_labels = torch.randint(0, 10, (1000,))
    fake_labels = torch.randint(0, 10, (1000,))
    
    cms, class_scores = calculate_cms(real_feat, fake_feat, real_labels, fake_labels, 10)
    print(f"测试 CMS 分数: {cms:.4f}")
    print(f"各类别分数: {class_scores}")