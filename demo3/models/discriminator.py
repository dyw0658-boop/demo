"""
ACGAN 判别器，包含多头自注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock
from .attention import AttentionBlock


class Discriminator(nn.Module):
    """
    ACGAN 判别器
    
    Args:
        ndf (int): 判别器特征图基数
        num_classes (int): 类别数量
        attn_heads (list): 注意力头数
        use_spectral_norm (bool): 是否使用谱归一化
    """
    
    def __init__(self, ndf=64, num_classes=10, attn_heads=[8], use_spectral_norm=True):
        super().__init__()
        self.ndf = ndf
        self.num_classes = num_classes
        
        # 初始卷积层: 3 -> ndf
        self.conv1 = nn.Conv2d(3, ndf, kernel_size=3, stride=1, padding=1)
        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
        
        # 下采样残差块序列
        # 32x32 -> 16x16
        self.res_block1 = ResidualBlock(ndf, ndf * 2, downsample=True, use_spectral_norm=use_spectral_norm)
        
        # 16x16 -> 8x8 (插入注意力层)
        self.res_block2 = ResidualBlock(ndf * 2, ndf * 4, downsample=True, use_spectral_norm=use_spectral_norm)
        self.attn1 = AttentionBlock(ndf * 4, num_heads=attn_heads[0], use_spectral_norm=use_spectral_norm)
        
        # 8x8 -> 4x4
        self.res_block3 = ResidualBlock(ndf * 4, ndf * 8, downsample=True, use_spectral_norm=use_spectral_norm)
        
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)
        
        # 有效性输出 (真实/假)
        self.validity_conv = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0)
        if use_spectral_norm:
            self.validity_conv = nn.utils.spectral_norm(self.validity_conv)
        
        # 类别分类输出
        self.class_conv = nn.Conv2d(ndf * 8, num_classes, kernel_size=4, stride=1, padding=0)
        if use_spectral_norm:
            self.class_conv = nn.utils.spectral_norm(self.class_conv)
        
        # 特征提取 (用于 FID/CMS 计算)
        self.feature_conv = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=1, stride=1, padding=0)
        if use_spectral_norm:
            self.feature_conv = nn.utils.spectral_norm(self.feature_conv)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像，shape [B, 3, 32, 32]
        
        Returns:
            validity: 有效性分数，shape [B, 1]
            class_logits: 类别 logits，shape [B, num_classes]
            features: 特征图，shape [B, ndf*8, 1, 1]
        """
        # 初始卷积
        x = self.conv1(x)  # [B, ndf, 32, 32]
        x = self.leaky_relu(x)
        
        # 32x32 -> 16x16
        x = self.res_block1(x)  # [B, ndf*2, 16, 16]
        x = self.leaky_relu(x)
        
        # 16x16 -> 8x8
        x = self.res_block2(x)  # [B, ndf*4, 8, 8]
        x = self.leaky_relu(x)
        
        # 注意力层 (8x8)
        x = self.attn1(x)  # [B, ndf*4, 8, 8]
        
        # 8x8 -> 4x4
        x = self.res_block3(x)  # [B, ndf*8, 4, 4]
        x = self.leaky_relu(x)
        
        # 有效性输出
        validity = self.validity_conv(x)  # [B, 1, 1, 1]
        validity = validity.view(validity.size(0), -1)  # [B, 1]
        
        # 类别分类输出
        class_logits = self.class_conv(x)  # [B, num_classes, 1, 1]
        class_logits = class_logits.view(class_logits.size(0), -1)  # [B, num_classes]
        
        # 特征提取 (倒数第二层特征)
        features = self.feature_conv(x)  # [B, ndf*8, 1, 1]
        features = features.reshape(features.size(0), -1)  # [B, ndf*8]
        
        return validity, class_logits, features
    
    def get_validity(self, x):
        """
        获取有效性分数
        
        Args:
            x: 输入图像，shape [B, 3, 32, 32]
        
        Returns:
            validity: 有效性分数，shape [B, 1]
        """
        validity, _, _ = self.forward(x)
        return validity
    
    def get_class_prob(self, x):
        """
        获取类别概率
        
        Args:
            x: 输入图像，shape [B, 3, 32, 32]
        
        Returns:
            class_prob: 类别概率，shape [B, num_classes]
        """
        _, class_logits, _ = self.forward(x)
        return F.softmax(class_logits, dim=1)
    
    def get_features(self, x):
        """
        获取特征向量 (用于 FID/CMS 计算)
        
        Args:
            x: 输入图像，shape [B, 3, 32, 32]
        
        Returns:
            features: 特征向量，shape [B, ndf*8]
        """
        _, _, features = self.forward(x)
        return features