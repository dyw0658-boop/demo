"""
残差块和基础构建块模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    残差块，支持上采样和下采样
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        upsample (bool): 是否上采样
        downsample (bool): 是否下采样
        use_spectral_norm (bool): 是否使用谱归一化
    """
    
    def __init__(self, in_channels, out_channels, upsample=False, downsample=False, use_spectral_norm=True):
        super().__init__()
        self.upsample = upsample
        self.downsample = downsample
        
        # 主路径
        layers = []
        
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(self._conv3x3(in_channels, out_channels, use_spectral_norm))
        elif downsample:
            layers.append(self._conv3x3(in_channels, out_channels, use_spectral_norm))
            layers.append(nn.AvgPool2d(2))
        else:
            layers.append(self._conv3x3(in_channels, out_channels, use_spectral_norm))
        
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=False))
        
        layers.append(self._conv3x3(out_channels, out_channels, use_spectral_norm))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.main = nn.Sequential(*layers)
        
        # 捷径路径
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or upsample or downsample:
            shortcut_layers = []
            if upsample:
                shortcut_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                shortcut_layers.append(self._conv1x1(in_channels, out_channels, use_spectral_norm))
            elif downsample:
                shortcut_layers.append(self._conv1x1(in_channels, out_channels, use_spectral_norm))
                shortcut_layers.append(nn.AvgPool2d(2))
            else:
                shortcut_layers.append(self._conv1x1(in_channels, out_channels, use_spectral_norm))
            
            shortcut_layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)
    
    def _conv3x3(self, in_channels, out_channels, use_spectral_norm):
        """3x3 卷积"""
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if use_spectral_norm:
            return nn.utils.spectral_norm(conv)
        return conv
    
    def _conv1x1(self, in_channels, out_channels, use_spectral_norm):
        """1x1 卷积"""
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        if use_spectral_norm:
            return nn.utils.spectral_norm(conv)
        return conv
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，shape [B, C, H, W]
        
        Returns:
            output: 输出张量，shape [B, C_out, H_out, W_out]
        """
        residual = self.shortcut(x)
        main = self.main(x)
        return F.relu(main + residual, inplace=False)


class ConditionalBatchNorm2d(nn.Module):
    """
    条件批归一化层
    
    Args:
        num_features (int): 特征数量
        num_classes (int): 类别数量
    """
    
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        
        # 为每个类别创建不同的仿射参数
        self.gamma = nn.Embedding(num_classes, num_features)
        self.beta = nn.Embedding(num_classes, num_features)
        
        # 初始化参数
        self.gamma.weight.data.normal_(1.0, 0.02)
        self.beta.weight.data.zero_()
    
    def forward(self, x, y):
        """
        Args:
            x: 输入张量，shape [B, C, H, W]
            y: 类别标签，shape [B]
        
        Returns:
            output: 条件归一化后的张量
        """
        # 批归一化
        out = self.bn(x)
        
        # 获取对应类别的仿射参数
        gamma = self.gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.beta(y).view(-1, self.num_features, 1, 1)
        
        # 应用仿射变换
        return gamma * out + beta