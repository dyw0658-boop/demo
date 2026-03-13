"""
感知图像补丁相似度 (LPIPS) 计算模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models


class LPIPS(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) 计算器
    """
    
    def __init__(self, net_type='vgg', use_dropout=True):
        """
        初始化LPIPS模型
        
        Args:
            net_type: 骨干网络类型 ('vgg', 'alexnet', 'squeezenet')
            use_dropout: 是否使用dropout
        """
        super().__init__()
        
        self.net_type = net_type
        self.use_dropout = use_dropout
        
        # 加载预训练模型
        if net_type == 'vgg':
            self.net = models.vgg16(pretrained=True).features
            self.channels = [64, 128, 256, 512, 512]
        elif net_type == 'alexnet':
            self.net = models.alexnet(pretrained=True).features
            self.channels = [64, 192, 384, 256, 256]
        elif net_type == 'squeezenet':
            self.net = models.squeezenet1_1(pretrained=True).features
            self.channels = [64, 128, 256, 384, 384]
        else:
            raise ValueError(f"不支持的骨干网络: {net_type}")
        
        # 冻结网络参数
        for param in self.net.parameters():
            param.requires_grad = False
        
        # 创建线性层用于特征融合
        self.lin_layers = nn.ModuleList()
        for ch in self.channels:
            self.lin_layers.append(nn.Sequential(
                nn.Dropout() if use_dropout else nn.Identity(),
                nn.Conv2d(ch, 1, 1, 1, 0, bias=False)
            ))
        
        # 加载预训练权重（如果可用）
        self._load_pretrained_weights()
        
        # 设置为评估模式
        self.eval()
    
    def _load_pretrained_weights(self):
        """加载预训练权重"""
        # 这里可以加载预训练的LPIPS权重
        # 如果没有预训练权重，使用随机初始化
        pass
    
    def forward(self, x, y):
        """
        计算LPIPS距离
        
        Args:
            x: 参考图像 [B, 3, H, W]
            y: 比较图像 [B, 3, H, W]
        
        Returns:
            lpips_distance: LPIPS距离 [B]
        """
        # 确保输入在 [0, 1] 范围内
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)
        
        # 提取特征
        x_features = self._extract_features(x)
        y_features = self._extract_features(y)
        
        # 计算LPIPS距离
        lpips_distance = 0.0
        
        for i, (x_feat, y_feat) in enumerate(zip(x_features, y_features)):
            # 归一化特征
            x_feat = F.normalize(x_feat, p=2, dim=1)
            y_feat = F.normalize(y_feat, p=2, dim=1)
            
            # 计算L2距离
            diff = (x_feat - y_feat) ** 2
            
            # 通过线性层
            diff = self.lin_layers[i](diff)
            
            # 空间平均池化
            lpips_distance += F.avg_pool2d(diff, kernel_size=diff.shape[2:]).squeeze()
        
        return lpips_distance
    
    def _extract_features(self, x):
        """提取多尺度特征"""
        features = []
        
        # VGG特征提取
        if self.net_type == 'vgg':
            for layer in self.net:
                x = layer(x)
                if isinstance(layer, nn.Conv2d) and x.shape[1] in self.channels:
                    features.append(x)
                    if len(features) == len(self.channels):
                        break
        
        # AlexNet特征提取
        elif self.net_type == 'alexnet':
            for layer in self.net:
                x = layer(x)
                if isinstance(layer, nn.Conv2d) and x.shape[1] in self.channels:
                    features.append(x)
                    if len(features) == len(self.channels):
                        break
        
        # SqueezeNet特征提取
        elif self.net_type == 'squeezenet':
            for layer in self.net:
                x = layer(x)
                if isinstance(layer, nn.Conv2d) and x.shape[1] in self.channels:
                    features.append(x)
                    if len(features) == len(self.channels):
                        break
        
        return features


def calculate_lpips(real_images, fake_images, net_type='vgg', device='cpu', batch_size=32):
    """
    计算LPIPS距离
    
    Args:
        real_images: 真实图像 [N, C, H, W]
        fake_images: 生成图像 [M, C, H, W]
        net_type: 骨干网络类型
        device: 计算设备
        batch_size: 批处理大小
    
    Returns:
        lpips_score: 平均LPIPS距离
        lpips_std: LPIPS距离标准差
    """
    # 确保图像数量一致
    min_len = min(len(real_images), len(fake_images))
    real_images = real_images[:min_len]
    fake_images = fake_images[:min_len]
    
    # 初始化LPIPS模型
    lpips_model = LPIPS(net_type=net_type).to(device)
    lpips_model.eval()
    
    lpips_distances = []
    
    with torch.no_grad():
        for i in range(0, min_len, batch_size):
            real_batch = real_images[i:i+batch_size].to(device)
            fake_batch = fake_images[i:i+batch_size].to(device)
            
            # 计算LPIPS距离
            distances = lpips_model(real_batch, fake_batch)
            lpips_distances.extend(distances.cpu().numpy())
    
    lpips_score = np.mean(lpips_distances)
    lpips_std = np.std(lpips_distances)
    
    return lpips_score, lpips_std


def calculate_class_lpips(real_images, fake_images, real_labels, fake_labels, num_classes,
                         net_type='vgg', device='cpu', batch_size=32):
    """
    计算每类LPIPS距离
    
    Args:
        real_images: 真实图像 [N, C, H, W]
        fake_images: 生成图像 [M, C, H, W]
        real_labels: 真实图像标签 [N]
        fake_labels: 生成图像标签 [M]
        num_classes: 类别数量
        net_type: 骨干网络类型
        device: 计算设备
        batch_size: 批处理大小
    
    Returns:
        class_lpips_scores: 各类别LPIPS距离
        class_lpips_stds: 各类别LPIPS标准差
        avg_lpips: 平均LPIPS距离
    """
    # 转换为numpy
    if isinstance(real_labels, torch.Tensor):
        real_labels = real_labels.detach().cpu().numpy()
    if isinstance(fake_labels, torch.Tensor):
        fake_labels = fake_labels.detach().cpu().numpy()
    
    class_lpips_scores = []
    class_lpips_stds = []
    
    for c in range(num_classes):
        # 获取当前类别的真实和生成图像
        real_mask = real_labels == c
        fake_mask = fake_labels == c
        
        real_class_images = real_images[real_mask]
        fake_class_images = fake_images[fake_mask]
        
        # 跳过没有样本的类别
        if len(real_class_images) == 0 or len(fake_class_images) == 0:
            class_lpips_scores.append(0.0)
            class_lpips_stds.append(0.0)
            continue
        
        # 计算当前类别的LPIPS
        lpips_score, lpips_std = calculate_lpips(
            real_class_images, fake_class_images, 
            net_type=net_type, device=device, batch_size=batch_size
        )
        
        class_lpips_scores.append(lpips_score)
        class_lpips_stds.append(lpips_std)
    
    avg_lpips = np.mean(class_lpips_scores)
    
    return class_lpips_scores, class_lpips_stds, avg_lpips


def calculate_lpips_with_features(real_features, fake_features, net_type='vgg'):
    """
    使用预提取的特征计算LPIPS（简化版本）
    
    Args:
        real_features: 真实图像特征
        fake_features: 生成图像特征
        net_type: 骨干网络类型
    
    Returns:
        lpips_distance: LPIPS距离
    """
    # 这里简化实现，直接计算特征之间的L2距离
    # 实际LPIPS需要更复杂的特征对齐和加权
    
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.detach().cpu().numpy()
    if isinstance(fake_features, torch.Tensor):
        fake_features = fake_features.detach().cpu().numpy()
    
    # 确保特征维度一致
    assert real_features.shape == fake_features.shape, "特征维度不一致"
    
    # 计算L2距离
    diff = real_features - fake_features
    lpips_distance = np.mean(np.sqrt(np.sum(diff ** 2, axis=1)))
    
    return lpips_distance


def calculate_multiscale_lpips(real_images, fake_images, scales=[1.0, 0.5, 0.25], 
                              net_type='vgg', device='cpu', batch_size=16):
    """
    计算多尺度LPIPS距离
    
    Args:
        real_images: 真实图像
        fake_images: 生成图像
        scales: 多尺度缩放因子
        net_type: 骨干网络类型
        device: 计算设备
        batch_size: 批处理大小
    
    Returns:
        multiscale_lpips: 多尺度LPIPS距离
    """
    multiscale_scores = []
    
    for scale in scales:
        if scale == 1.0:
            real_scaled = real_images
            fake_scaled = fake_images
        else:
            # 缩放图像
            real_scaled = F.interpolate(real_images, scale_factor=scale, mode='bilinear', align_corners=False)
            fake_scaled = F.interpolate(fake_images, scale_factor=scale, mode='bilinear', align_corners=False)
        
        # 计算当前尺度的LPIPS
        lpips_score, _ = calculate_lpips(real_scaled, fake_scaled, net_type, device, batch_size)
        multiscale_scores.append(lpips_score)
    
    # 加权平均
    weights = [1.0, 0.5, 0.25]  # 可以根据需要调整权重
    multiscale_lpips = np.average(multiscale_scores, weights=weights[:len(scales)])
    
    return multiscale_lpips