"""
每类核Inception距离 (KID) 计算模块
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel


def calculate_kid(real_features, fake_features, kernel='poly', degree=3, gamma=None, coef0=1, num_subsets=10, subset_size=1000):
    """
    计算核Inception距离 (KID)
    
    Args:
        real_features: 真实图像特征 [N, feature_dim]
        fake_features: 生成图像特征 [M, feature_dim]
        kernel: 核函数类型 ('poly', 'rbf', 'linear')
        degree: 多项式核的阶数
        gamma: 核函数的gamma参数
        coef0: 多项式核的常数项
        num_subsets: 子集数量
        subset_size: 每个子集的大小
    
    Returns:
        kid_score: KID分数
        kid_std: KID标准差
    """
    # 转换为numpy
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.detach().cpu().numpy()
    if isinstance(fake_features, torch.Tensor):
        fake_features = fake_features.detach().cpu().numpy()
    
    # 确保特征维度一致
    assert real_features.shape[1] == fake_features.shape[1], "特征维度不一致"
    
    # 设置默认gamma
    if gamma is None:
        gamma = 1.0 / real_features.shape[1]
    
    # 限制子集大小
    subset_size = min(subset_size, len(real_features), len(fake_features))
    
    kid_scores = []
    
    for _ in range(num_subsets):
        # 随机采样子集
        real_idx = np.random.choice(len(real_features), subset_size, replace=False)
        fake_idx = np.random.choice(len(fake_features), subset_size, replace=False)
        
        real_subset = real_features[real_idx]
        fake_subset = fake_features[fake_idx]
        
        # 计算核矩阵
        if kernel == 'poly':
            k_real_real = polynomial_kernel(real_subset, real_subset, degree=degree, gamma=gamma, coef0=coef0)
            k_fake_fake = polynomial_kernel(fake_subset, fake_subset, degree=degree, gamma=gamma, coef0=coef0)
            k_real_fake = polynomial_kernel(real_subset, fake_subset, degree=degree, gamma=gamma, coef0=coef0)
        elif kernel == 'rbf':
            k_real_real = polynomial_kernel(real_subset, real_subset, degree=1, gamma=gamma)
            k_fake_fake = polynomial_kernel(fake_subset, fake_subset, degree=1, gamma=gamma)
            k_real_fake = polynomial_kernel(real_subset, fake_subset, degree=1, gamma=gamma)
        elif kernel == 'linear':
            k_real_real = polynomial_kernel(real_subset, real_subset, degree=1)
            k_fake_fake = polynomial_kernel(fake_subset, fake_subset, degree=1)
            k_real_fake = polynomial_kernel(real_subset, fake_subset, degree=1)
        else:
            raise ValueError(f"不支持的核函数: {kernel}")
        
        # 计算KID
        kid = (k_real_real.mean() + k_fake_fake.mean() - 2 * k_real_fake.mean())
        kid_scores.append(kid)
    
    kid_score = np.mean(kid_scores)
    kid_std = np.std(kid_scores)
    
    return kid_score, kid_std


def calculate_class_kid(real_features, fake_features, real_labels, fake_labels, num_classes, 
                        kernel='poly', degree=3, gamma=None, coef0=1, num_subsets=5, subset_size=500):
    """
    计算每类核Inception距离 (Class-wise KID)
    
    Args:
        real_features: 真实图像特征 [N, feature_dim]
        fake_features: 生成图像特征 [M, feature_dim]
        real_labels: 真实图像标签 [N]
        fake_labels: 生成图像标签 [M]
        num_classes: 类别数量
        kernel: 核函数类型
        degree: 多项式核的阶数
        gamma: 核函数的gamma参数
        coef0: 多项式核的常数项
        num_subsets: 子集数量
        subset_size: 每个子集的大小
    
    Returns:
        class_kid_scores: 各类别KID分数
        class_kid_stds: 各类别KID标准差
        avg_kid: 平均KID分数
    """
    # 转换为numpy
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.detach().cpu().numpy()
    if isinstance(fake_features, torch.Tensor):
        fake_features = fake_features.detach().cpu().numpy()
    if isinstance(real_labels, torch.Tensor):
        real_labels = real_labels.detach().cpu().numpy()
    if isinstance(fake_labels, torch.Tensor):
        fake_labels = fake_labels.detach().cpu().numpy()
    
    class_kid_scores = []
    class_kid_stds = []
    
    for c in range(num_classes):
        # 获取当前类别的真实和生成特征
        real_mask = real_labels == c
        fake_mask = fake_labels == c
        
        real_class_features = real_features[real_mask]
        fake_class_features = fake_features[fake_mask]
        
        # 跳过没有样本的类别
        if len(real_class_features) == 0 or len(fake_class_features) == 0:
            class_kid_scores.append(0.0)
            class_kid_stds.append(0.0)
            continue
        
        # 计算当前类别的KID
        kid_score, kid_std = calculate_kid(
            real_class_features, fake_class_features, 
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            num_subsets=num_subsets, subset_size=subset_size
        )
        
        class_kid_scores.append(kid_score)
        class_kid_stds.append(kid_std)
    
    avg_kid = np.mean(class_kid_scores)
    
    return class_kid_scores, class_kid_stds, avg_kid


def calculate_inception_kid(real_images, fake_images, inception_model, device, 
                           kernel='poly', degree=3, gamma=None, coef0=1, 
                           num_subsets=10, subset_size=1000):
    """
    使用Inception模型计算KID
    
    Args:
        real_images: 真实图像 [N, C, H, W]
        fake_images: 生成图像 [M, C, H, W]
        inception_model: Inception模型
        device: 计算设备
        kernel: 核函数类型
        degree: 多项式核的阶数
        gamma: 核函数的gamma参数
        coef0: 多项式核的常数项
        num_subsets: 子集数量
        subset_size: 每个子集的大小
    
    Returns:
        kid_score: KID分数
        kid_std: KID标准差
    """
    inception_model.eval()
    
    # 提取特征
    with torch.no_grad():
        real_features = []
        fake_features = []
        
        # 分批处理真实图像
        batch_size = 64
        for i in range(0, len(real_images), batch_size):
            batch = real_images[i:i+batch_size].to(device)
            features = inception_model(batch)
            if isinstance(features, tuple):
                features = features[0]  # 取主要特征
            real_features.append(features.cpu())
        
        # 分批处理生成图像
        for i in range(0, len(fake_images), batch_size):
            batch = fake_images[i:i+batch_size].to(device)
            features = inception_model(batch)
            if isinstance(features, tuple):
                features = features[0]  # 取主要特征
            fake_features.append(features.cpu())
        
        real_features = torch.cat(real_features, dim=0)
        fake_features = torch.cat(fake_features, dim=0)
    
    # 计算KID
    kid_score, kid_std = calculate_kid(
        real_features, fake_features, 
        kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
        num_subsets=num_subsets, subset_size=subset_size
    )
    
    return kid_score, kid_std


def calculate_kid_with_inception_features(real_features, fake_features, 
                                        kernel='poly', degree=3, gamma=None, coef0=1,
                                        num_subsets=10, subset_size=1000):
    """
    使用预提取的Inception特征计算KID
    
    Args:
        real_features: 真实图像Inception特征
        fake_features: 生成图像Inception特征
        kernel: 核函数类型
        degree: 多项式核的阶数
        gamma: 核函数的gamma参数
        coef0: 多项式核的常数项
        num_subsets: 子集数量
        subset_size: 每个子集的大小
    
    Returns:
        kid_score: KID分数
        kid_std: KID标准差
    """
    return calculate_kid(
        real_features, fake_features, 
        kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
        num_subsets=num_subsets, subset_size=subset_size
    )