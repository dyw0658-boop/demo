"""
指标计算模块
"""

from .fid import calculate_fid
from .inception_score import InceptionScore
from .cms import calculate_cms
from .kl_divergence import (
    calculate_kl_divergence,
    calculate_js_divergence,
    calculate_class_kl_divergence,
    calculate_inception_kl_divergence
)
from .kid import (
    calculate_kid,
    calculate_class_kid,
    calculate_inception_kid,
    calculate_kid_with_inception_features
)
from .lpips import (
    LPIPS,
    calculate_lpips,
    calculate_class_lpips,
    calculate_lpips_with_features,
    calculate_multiscale_lpips
)


class MetricsCalculator:
    """
    统一指标计算器
    """
    
    def __init__(self, device='cpu', num_classes=10, use_inception=True, use_lpips=True):
        self.device = device
        self.num_classes = num_classes
        self.use_inception = use_inception
        self.use_lpips = use_lpips
        
        # 初始化Inception模型
        if use_inception:
            try:
                self.inception_model = InceptionScore(pretrained=False).to(device)
                print("✅ Inception模型初始化成功")
            except Exception as e:
                print(f"❌ Inception模型初始化失败: {e}")
                print("⚠️  将跳过基于Inception的指标计算")
                self.use_inception = False
                self.inception_model = None
        else:
            self.inception_model = None
        
        # 初始化LPIPS模型
        if use_lpips:
            try:
                self.lpips_model = LPIPS(net_type='vgg').to(device)
                print("✅ LPIPS模型初始化成功")
            except Exception as e:
                print(f"❌ LPIPS模型初始化失败: {e}")
                print("⚠️  将跳过LPIPS指标计算")
                self.use_lpips = False
                self.lpips_model = None
        else:
            self.lpips_model = None
    
    def calculate_all_metrics(self, real_images, fake_images, real_labels=None, fake_labels=None):
        """
        计算所有指标
        
        Args:
            real_images: 真实图像
            fake_images: 生成图像
            real_labels: 真实图像标签
            fake_labels: 生成图像标签
        
        Returns:
            metrics_dict: 指标字典
        """
        metrics_dict = {}
        
        # 确保图像数量一致
        min_len = min(len(real_images), len(fake_images))
        real_images = real_images[:min_len]
        fake_images = fake_images[:min_len]
        
        if real_labels is not None and fake_labels is not None:
            real_labels = real_labels[:min_len]
            fake_labels = fake_labels[:min_len]
        
        # 提取Inception特征
        with torch.no_grad():
            real_features = []
            fake_features = []
            
            batch_size = 64
            for i in range(0, min_len, batch_size):
                real_batch = real_images[i:i+batch_size].to(self.device)
                fake_batch = fake_images[i:i+batch_size].to(self.device)
                
                real_feat = self.inception_model(real_batch)
                fake_feat = self.inception_model(fake_batch)
                
                real_features.append(real_feat.cpu())
                fake_features.append(fake_feat.cpu())
            
            real_features = torch.cat(real_features, dim=0)
            fake_features = torch.cat(fake_features, dim=0)
        
        # 计算FID
        fid_score = calculate_fid(real_features, fake_features)
        metrics_dict['fid'] = fid_score
        
        # 计算KID
        kid_score, kid_std = calculate_kid(real_features, fake_features)
        metrics_dict['kid'] = kid_score
        metrics_dict['kid_std'] = kid_std
        
        # 计算KL散度
        kl_score = calculate_inception_kl_divergence(
            real_images, fake_images, self.inception_model, self.device
        )
        metrics_dict['kl_divergence'] = kl_score
        
        # 计算LPIPS
        lpips_score, lpips_std = calculate_lpips(
            real_images, fake_images, device=self.device
        )
        metrics_dict['lpips'] = lpips_score
        metrics_dict['lpips_std'] = lpips_std
        
        # 如果有标签，计算类间指标
        if real_labels is not None and fake_labels is not None:
            # 计算类间FID
            class_fid_scores, avg_fid = calculate_cms(
                real_features, fake_features, real_labels, fake_labels, self.num_classes
            )
            metrics_dict['class_fid'] = class_fid_scores
            metrics_dict['avg_class_fid'] = avg_fid
            
            # 计算类间KID
            class_kid_scores, class_kid_stds, avg_kid = calculate_class_kid(
                real_features, fake_features, real_labels, fake_labels, self.num_classes
            )
            metrics_dict['class_kid'] = class_kid_scores
            metrics_dict['class_kid_std'] = class_kid_stds
            metrics_dict['avg_class_kid'] = avg_kid
            
            # 计算类间KL散度
            class_kl_scores, avg_kl = calculate_class_kl_divergence(
                real_features, fake_features, real_labels, fake_labels, self.num_classes
            )
            metrics_dict['class_kl'] = class_kl_scores
            metrics_dict['avg_class_kl'] = avg_kl
            
            # 计算类间LPIPS
            class_lpips_scores, class_lpips_stds, avg_lpips = calculate_class_lpips(
                real_images, fake_images, real_labels, fake_labels, self.num_classes,
                device=self.device
            )
            metrics_dict['class_lpips'] = class_lpips_scores
            metrics_dict['class_lpips_std'] = class_lpips_stds
            metrics_dict['avg_class_lpips'] = avg_lpips
        
        return metrics_dict
    
    def calculate_batch_metrics(self, real_batch, fake_batch, real_labels=None, fake_labels=None):
        """
        计算批次指标（轻量级版本）
        
        Args:
            real_batch: 真实图像批次
            fake_batch: 生成图像批次
            real_labels: 真实图像标签
            fake_labels: 生成图像标签
        
        Returns:
            metrics_dict: 指标字典
        """
        metrics_dict = {}
        
        # 计算LPIPS（不需要特征提取）
        lpips_score, lpips_std = calculate_lpips(
            real_batch, fake_batch, device=self.device, batch_size=len(real_batch)
        )
        metrics_dict['lpips'] = lpips_score
        metrics_dict['lpips_std'] = lpips_std
        
        # 如果有标签，计算类间LPIPS
        if real_labels is not None and fake_labels is not None:
            class_lpips_scores, class_lpips_stds, avg_lpips = calculate_class_lpips(
                real_batch, fake_batch, real_labels, fake_labels, self.num_classes,
                device=self.device, batch_size=len(real_batch)
            )
            metrics_dict['class_lpips'] = class_lpips_scores
            metrics_dict['class_lpips_std'] = class_lpips_stds
            metrics_dict['avg_class_lpips'] = avg_lpips
        
        return metrics_dict