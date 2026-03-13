"""
KL散度计算模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy


def calculate_kl_divergence(p_dist, q_dist, eps=1e-8):
    """
    计算KL散度 D_KL(P || Q)
    
    Args:
        p_dist: 分布P，可以是概率向量或logits
        q_dist: 分布Q，可以是概率向量或logits
        eps: 数值稳定性参数
    
    Returns:
        kl_div: KL散度值
    """
    # 确保输入是概率分布
    if isinstance(p_dist, torch.Tensor):
        p_dist = p_dist.detach().cpu().numpy()
    if isinstance(q_dist, torch.Tensor):
        q_dist = q_dist.detach().cpu().numpy()
    
    # 如果输入是logits，转换为概率
    if p_dist.ndim == 2 and p_dist.shape[1] > 1:
        p_dist = F.softmax(torch.tensor(p_dist), dim=1).numpy()
    if q_dist.ndim == 2 and q_dist.shape[1] > 1:
        q_dist = F.softmax(torch.tensor(q_dist), dim=1).numpy()
    
    # 确保概率和为1
    p_dist = p_dist / (p_dist.sum(axis=-1, keepdims=True) + eps)
    q_dist = q_dist / (q_dist.sum(axis=-1, keepdims=True) + eps)
    
    # 计算KL散度
    kl_div = entropy(p_dist, q_dist, axis=-1)
    
    return kl_div.mean() if kl_div.ndim > 0 else kl_div


def calculate_js_divergence(p_dist, q_dist, eps=1e-8):
    """
    计算JS散度 (Jensen-Shannon Divergence)
    
    Args:
        p_dist: 分布P
        q_dist: 分布Q
        eps: 数值稳定性参数
    
    Returns:
        js_div: JS散度值
    """
    # 计算平均分布
    m_dist = 0.5 * (p_dist + q_dist)
    
    # 计算KL散度
    kl_pm = calculate_kl_divergence(p_dist, m_dist, eps)
    kl_qm = calculate_kl_divergence(q_dist, m_dist, eps)
    
    # JS散度
    js_div = 0.5 * (kl_pm + kl_qm)
    
    return js_div


def calculate_class_kl_divergence(real_probs, fake_probs, real_labels, fake_labels, num_classes, eps=1e-8):
    """
    计算类间KL散度
    
    Args:
        real_probs: 真实图像的概率分布 [N, num_classes]
        fake_probs: 生成图像的概率分布 [M, num_classes]
        real_labels: 真实图像标签 [N]
        fake_labels: 生成图像标签 [M]
        num_classes: 类别数量
        eps: 数值稳定性参数
    
    Returns:
        class_kl_scores: 各类别KL散度
        avg_kl: 平均KL散度
    """
    # 转换为numpy
    if isinstance(real_probs, torch.Tensor):
        real_probs = real_probs.detach().cpu().numpy()
    if isinstance(fake_probs, torch.Tensor):
        fake_probs = fake_probs.detach().cpu().numpy()
    if isinstance(real_labels, torch.Tensor):
        real_labels = real_labels.detach().cpu().numpy()
    if isinstance(fake_labels, torch.Tensor):
        fake_labels = fake_labels.detach().cpu().numpy()
    
    class_kl_scores = []
    
    for c in range(num_classes):
        # 获取当前类别的真实和生成概率
        real_mask = real_labels == c
        fake_mask = fake_labels == c
        
        real_class_probs = real_probs[real_mask]
        fake_class_probs = fake_probs[fake_mask]
        
        # 跳过没有样本的类别
        if len(real_class_probs) == 0 or len(fake_class_probs) == 0:
            class_kl_scores.append(0.0)
            continue
        
        # 计算类别平均概率分布
        real_class_avg = real_class_probs.mean(axis=0)
        fake_class_avg = fake_class_probs.mean(axis=0)
        
        # 计算KL散度
        kl_score = calculate_kl_divergence(real_class_avg, fake_class_avg, eps)
        class_kl_scores.append(kl_score)
    
    avg_kl = np.mean(class_kl_scores)
    
    return class_kl_scores, avg_kl


def calculate_inception_kl_divergence(real_images, fake_images, inception_model, device, num_samples=5000):
    """
    使用Inception模型计算KL散度
    
    Args:
        real_images: 真实图像 [N, C, H, W]
        fake_images: 生成图像 [M, C, H, W]
        inception_model: Inception模型
        device: 计算设备
        num_samples: 采样数量
    
    Returns:
        kl_score: KL散度分数
    """
    inception_model.eval()
    
    # 采样
    real_idx = np.random.choice(len(real_images), min(num_samples, len(real_images)), replace=False)
    fake_idx = np.random.choice(len(fake_images), min(num_samples, len(fake_images)), replace=False)
    
    real_samples = real_images[real_idx].to(device)
    fake_samples = fake_images[fake_idx].to(device)
    
    # 计算概率分布
    with torch.no_grad():
        real_probs = inception_model(real_samples)
        fake_probs = inception_model(fake_samples)
    
    # 计算KL散度
    kl_score = calculate_kl_divergence(real_probs, fake_probs)
    
    return kl_score