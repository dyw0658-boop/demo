"""
多头自注意力机制模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def reshape_for_attention(x, num_heads):
    """
    将输入张量重塑为多头注意力格式
    
    Args:
        x: 输入张量，shape [B, C, H, W]
        num_heads (int): 注意力头数
    
    Returns:
        reshaped: 重塑后的张量，shape [B*num_heads, C//num_heads, H*W]
    """
    B, C, H, W = x.shape
    
    # 确保通道数可以被头数整除
    assert C % num_heads == 0, f"通道数 {C} 必须能被头数 {num_heads} 整除"
    
    # 重塑: [B, C, H, W] -> [B, num_heads, C//num_heads, H*W]
    x = x.view(B, num_heads, C // num_heads, H * W)
    
    # 转置: [B, num_heads, C//num_heads, H*W] -> [B, H*W, num_heads, C//num_heads]
    x = x.permute(0, 3, 1, 2).contiguous()
    
    # 最终形状: [B*num_heads, H*W, C//num_heads]
    x = x.view(B * num_heads, H * W, C // num_heads)
    
    return x


def reshape_from_attention(x, original_shape, num_heads):
    """
    将多头注意力输出重塑回原始格式
    
    Args:
        x: 注意力输出，shape [B*num_heads, H*W, C//num_heads]
        original_shape: 原始形状 (B, C, H, W)
        num_heads (int): 注意力头数
    
    Returns:
        reshaped: 重塑后的张量，shape [B, C, H, W]
    """
    B, C, H, W = original_shape
    
    # 重塑: [B*num_heads, H*W, C//num_heads] -> [B, H*W, num_heads, C//num_heads]
    x = x.view(B, H * W, num_heads, C // num_heads)
    
    # 转置: [B, H*W, num_heads, C//num_heads] -> [B, num_heads, C//num_heads, H*W]
    x = x.permute(0, 2, 3, 1).contiguous()
    
    # 最终形状: [B, C, H, W]
    x = x.view(B, C, H, W)
    
    return x


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力层
    
    Args:
        channels (int): 输入通道数
        num_heads (int): 注意力头数
        use_spectral_norm (bool): 是否使用谱归一化
    """
    
    def __init__(self, channels, num_heads=8, use_spectral_norm=True):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "通道数必须能被头数整除"
        
        # 使用 1x1 卷积生成 Q, K, V
        def make_conv():
            conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            if use_spectral_norm:
                return nn.utils.spectral_norm(conv)
            return conv
        
        self.query_conv = make_conv()
        self.key_conv = make_conv()
        self.value_conv = make_conv()
        
        # 输出投影
        self.out_conv = make_conv()
        
        # 层归一化和缩放因子
        self.layer_norm = nn.LayerNorm(channels)
        self.scale = self.head_dim ** -0.5
        
        # 残差连接的 gamma 参数
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，shape [B, C, H, W]
        
        Returns:
            output: 注意力输出，shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x
        
        # 生成 Q, K, V
        q = self.query_conv(x)  # [B, C, H, W]
        k = self.key_conv(x)    # [B, C, H, W]
        v = self.value_conv(x)  # [B, C, H, W]
        
        # 重塑为多头格式
        q = reshape_for_attention(q, self.num_heads)  # [B*num_heads, H*W, head_dim]
        k = reshape_for_attention(k, self.num_heads)  # [B*num_heads, H*W, head_dim]
        v = reshape_for_attention(v, self.num_heads)  # [B*num_heads, H*W, head_dim]
        
        # 计算注意力权重: Q * K^T / sqrt(d_k)
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B*num_heads, H*W, H*W]
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力权重到 V
        attn_out = torch.bmm(attn_weights, v)  # [B*num_heads, H*W, head_dim]
        
        # 重塑回原始格式
        attn_out = reshape_from_attention(attn_out, (B, C, H, W), self.num_heads)  # [B, C, H, W]
        
        # 输出投影
        out = self.out_conv(attn_out)  # [B, C, H, W]
        
        # 层归一化
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.layer_norm(out)
        out = out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 残差连接
        out = self.gamma * out + residual
        
        return out


class AttentionBlock(nn.Module):
    """
    注意力块，包含多头自注意力和前馈网络
    
    Args:
        channels (int): 输入通道数
        num_heads (int): 注意力头数
        mlp_ratio (float): MLP 扩展比例
        use_spectral_norm (bool): 是否使用谱归一化
    """
    
    def __init__(self, channels, num_heads=8, mlp_ratio=4.0, use_spectral_norm=True):
        super().__init__()
        
        # 多头自注意力
        self.attn = MultiHeadSelfAttention(channels, num_heads, use_spectral_norm)
        
        # 前馈网络
        mlp_hidden_dim = int(channels * mlp_ratio)
        
        def make_linear(in_dim, out_dim):
            linear = nn.Linear(in_dim, out_dim)
            if use_spectral_norm:
                return nn.utils.spectral_norm(linear)
            return linear
        
        self.mlp = nn.Sequential(
            make_linear(channels, mlp_hidden_dim),
            nn.GELU(),
            make_linear(mlp_hidden_dim, channels),
        )
        
        # 第二个层归一化
        self.norm2 = nn.LayerNorm(channels)
        
        # MLP 的 gamma 参数
        self.gamma2 = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，shape [B, C, H, W]
        
        Returns:
            output: 注意力块输出，shape [B, C, H, W]
        """
        # 第一个残差连接 (注意力)
        x = self.attn(x)
        
        # 第二个残差连接 (MLP)
        residual = x
        
        # 应用 MLP
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        mlp_out = self.mlp(x)
        mlp_out = self.norm2(mlp_out)
        mlp_out = mlp_out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 残差连接
        x = self.gamma2 * mlp_out + residual
        
        return x