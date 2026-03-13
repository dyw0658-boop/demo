"""
ACGAN 生成器，包含多头自注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock, ConditionalBatchNorm2d
from .attention import AttentionBlock


class Generator(nn.Module):
    """
    ACGAN 生成器
    
    Args:
        latent_dim (int): 噪声维度
        num_classes (int): 类别数量
        ngf (int): 生成器特征图基数
        attn_heads (list): 各分辨率下的注意力头数
        use_spectral_norm (bool): 是否使用谱归一化
    """
    
    def __init__(self, latent_dim=100, num_classes=10, ngf=64, attn_heads=[8, 4], use_spectral_norm=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.ngf = ngf
        self.attn_heads = attn_heads
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # 初始线性层: (latent_dim + latent_dim) -> ngf*8*4*4
        self.linear = nn.Linear(latent_dim * 2, ngf * 8 * 4 * 4)
        if use_spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)
        
        # 条件批归一化
        self.bn1 = ConditionalBatchNorm2d(ngf * 8, num_classes)
        
        # 上采样残差块序列
        # 4x4 -> 8x8
        self.res_block1 = ResidualBlock(ngf * 8, ngf * 8, upsample=True, use_spectral_norm=use_spectral_norm)
        self.bn2 = ConditionalBatchNorm2d(ngf * 8, num_classes)
        
        # 8x8 -> 16x16 (插入第一个注意力层)
        self.res_block2 = ResidualBlock(ngf * 8, ngf * 4, upsample=True, use_spectral_norm=use_spectral_norm)
        self.bn3 = ConditionalBatchNorm2d(ngf * 4, num_classes)
        self.attn1 = AttentionBlock(ngf * 4, num_heads=attn_heads[0], use_spectral_norm=use_spectral_norm)
        
        # 16x16 -> 32x32 (插入第二个注意力层)
        self.res_block3 = ResidualBlock(ngf * 4, ngf * 2, upsample=True, use_spectral_norm=use_spectral_norm)
        self.bn4 = ConditionalBatchNorm2d(ngf * 2, num_classes)
        self.attn2 = AttentionBlock(ngf * 2, num_heads=attn_heads[1], use_spectral_norm=use_spectral_norm)
        
        # 最终卷积层
        self.final_conv = nn.Conv2d(ngf * 2, 3, kernel_size=3, padding=1)
        
        # 激活函数
        self.activation = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, noise, labels):
        """
        前向传播
        
        Args:
            noise: 噪声张量，shape [B, latent_dim]
            labels: 类别标签，shape [B]
        
        Returns:
            output: 生成的图像，shape [B, 3, 32, 32]，值域 [-1, 1]
        """
        B = noise.size(0)
        
        # 标签嵌入
        label_emb = self.label_embedding(labels)  # [B, latent_dim]
        
        # 拼接噪声和标签嵌入
        concat_input = torch.cat([noise, label_emb], dim=1)  # [B, latent_dim * 2]
        
        # 线性投影
        x = self.linear(concat_input)  # [B, ngf*8*4*4]
        x = x.view(B, self.ngf * 8, 4, 4)  # [B, ngf*8, 4, 4]
        
        # 第一个条件批归一化和激活
        x = self.bn1(x, labels)
        x = self.activation(x)
        
        # 4x4 -> 8x8
        x = self.res_block1(x)
        x = self.bn2(x, labels)
        x = self.activation(x)
        
        # 8x8 -> 16x8
        x = self.res_block2(x)
        x = self.bn3(x, labels)
        x = self.activation(x)
        
        # 第一个注意力层 (16x16)
        x = self.attn1(x)
        
        # 16x16 -> 32x32
        x = self.res_block3(x)
        x = self.bn4(x, labels)
        x = self.activation(x)
        
        # 第二个注意力层 (32x32)
        x = self.attn2(x)
        
        # 最终卷积和激活
        x = self.final_conv(x)  # [B, 3, 32, 32]
        x = self.tanh(x)  # 值域 [-1, 1]
        
        return x
    
    def sample_noise(self, batch_size, device='cpu'):
        """
        生成随机噪声
        
        Args:
            batch_size (int): 批量大小
            device: 设备
        
        Returns:
            noise: 随机噪声，shape [batch_size, latent_dim]
        """
        return torch.randn(batch_size, self.latent_dim, device=device)
    
    def sample_labels(self, batch_size, device='cpu'):
        """
        生成随机标签
        
        Args:
            batch_size (int): 批量大小
            device: 设备
        
        Returns:
            labels: 随机标签，shape [batch_size]
        """
        return torch.randint(0, self.num_classes, (batch_size,), device=device)
    
    def generate(self, batch_size, device='cpu'):
        """
        生成图像
        
        Args:
            batch_size (int): 批量大小
            device: 设备
        
        Returns:
            images: 生成的图像，shape [batch_size, 3, 32, 32]
            labels: 对应的标签，shape [batch_size]
        """
        noise = self.sample_noise(batch_size, device)
        labels = self.sample_labels(batch_size, device)
        images = self.forward(noise, labels)
        return images, labels