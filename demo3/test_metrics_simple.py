"""
简化版指标测试脚本 - 避免网络依赖
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_kl_divergence():
    """测试KL散度模块"""
    print("=== 测试KL散度模块 ===")
    
    from metrics.kl_divergence import calculate_kl_divergence, calculate_js_divergence
    
    # 创建测试数据
    p_dist = torch.softmax(torch.randn(100, 10), dim=1)
    q_dist = torch.softmax(torch.randn(100, 10), dim=1)
    
    # 计算KL散度
    kl_score = calculate_kl_divergence(p_dist, q_dist)
    print(f"KL散度: {kl_score:.6f}")
    
    # 计算JS散度
    js_score = calculate_js_divergence(p_dist, q_dist)
    print(f"JS散度: {js_score:.6f}")
    
    # 测试类间KL散度
    from metrics.kl_divergence import calculate_class_kl_divergence
    
    real_labels = torch.randint(0, 10, (100,))
    fake_labels = torch.randint(0, 10, (100,))
    
    class_kl_scores, avg_kl = calculate_class_kl_divergence(
        p_dist, q_dist, real_labels, fake_labels, 10
    )
    print(f"平均类间KL散度: {avg_kl:.6f}")
    print("✅ KL散度模块测试通过")


def test_kid():
    """测试KID模块"""
    print("\n=== 测试KID模块 ===")
    
    from metrics.kid import calculate_kid, calculate_class_kid
    
    # 创建测试特征
    real_features = torch.randn(500, 2048)
    fake_features = torch.randn(500, 2048)
    
    # 计算KID
    kid_score, kid_std = calculate_kid(real_features, fake_features, num_subsets=5, subset_size=100)
    print(f"KID分数: {kid_score:.6f} ± {kid_std:.6f}")
    
    # 测试类间KID
    real_labels = torch.randint(0, 10, (500,))
    fake_labels = torch.randint(0, 10, (500,))
    
    class_kid_scores, class_kid_stds, avg_kid = calculate_class_kid(
        real_features, fake_features, real_labels, fake_labels, 10,
        num_subsets=3, subset_size=50
    )
    print(f"平均类间KID: {avg_kid:.6f}")
    print("✅ KID模块测试通过")


def test_lpips_simple():
    """测试LPIPS模块（简化版）"""
    print("\n=== 测试LPIPS模块（简化版） ===")
    
    from metrics.lpips import calculate_lpips_with_features
    
    # 创建测试特征
    real_features = torch.randn(100, 512)
    fake_features = torch.randn(100, 512)
    
    # 计算简化版LPIPS
    lpips_distance = calculate_lpips_with_features(real_features, fake_features)
    print(f"简化版LPIPS距离: {lpips_distance:.6f}")
    
    print("✅ LPIPS简化版模块测试通过")


def test_metrics_calculator():
    """测试统一指标计算器"""
    print("\n=== 测试统一指标计算器 ===")
    
    from metrics import MetricsCalculator
    
    # 创建测试数据
    real_images = torch.randn(100, 3, 32, 32)
    fake_images = torch.randn(100, 3, 32, 32)
    real_labels = torch.randint(0, 10, (100,))
    fake_labels = torch.randint(0, 10, (100,))
    
    # 初始化指标计算器
    metrics_calculator = MetricsCalculator(device='cpu', num_classes=10)
    
    # 测试批次指标
    batch_metrics = metrics_calculator.calculate_batch_metrics(
        real_images, fake_images, real_labels, fake_labels
    )
    
    print("批次指标结果:")
    for metric_name, metric_value in batch_metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"  {metric_name}: {metric_value:.6f}")
        elif isinstance(metric_value, list):
            print(f"  {metric_name}: [{len(metric_value)}个类别]")
    
    print("✅ 统一指标计算器测试通过")


def test_fid_and_cms():
    """测试现有FID和CMS模块"""
    print("\n=== 测试现有FID和CMS模块 ===")
    
    from metrics.fid import calculate_fid
    from metrics.cms import calculate_cms
    
    # 创建测试特征
    real_features = torch.randn(500, 2048)
    fake_features = torch.randn(500, 2048)
    real_labels = torch.randint(0, 10, (500,))
    fake_labels = torch.randint(0, 10, (500,))
    
    # 计算FID
    fid_score = calculate_fid(real_features, fake_features)
    print(f"FID分数: {fid_score:.6f}")
    
    # 计算CMS
    cms_score, class_scores = calculate_cms(real_features, fake_features, real_labels, fake_labels, 10)
    print(f"CMS分数: {cms_score:.6f}")
    
    print("✅ FID和CMS模块测试通过")


def main():
    """主函数"""
    print("开始简化版指标测试...")
    
    try:
        # 测试KL散度模块
        test_kl_divergence()
        
        # 测试KID模块
        test_kid()
        
        # 测试LPIPS简化版
        test_lpips_simple()
        
        # 测试统一指标计算器（禁用网络依赖）
        print("\n=== 测试统一指标计算器（禁用网络依赖） ===")
        
        from metrics import MetricsCalculator
        
        # 创建测试数据
        real_images = torch.randn(100, 3, 32, 32)
        fake_images = torch.randn(100, 3, 32, 32)
        real_labels = torch.randint(0, 10, (100,))
        fake_labels = torch.randint(0, 10, (100,))
        
        # 初始化指标计算器（禁用网络依赖）
        metrics_calculator = MetricsCalculator(device='cpu', num_classes=10, use_inception=False, use_lpips=False)
        
        # 测试批次指标
        batch_metrics = metrics_calculator.calculate_batch_metrics(
            real_images, fake_images, real_labels, fake_labels
        )
        
        print("批次指标结果:")
        for metric_name, metric_value in batch_metrics.items():
            if isinstance(metric_value, (int, float)):
                print(f"  {metric_name}: {metric_value:.6f}")
            elif isinstance(metric_value, list):
                print(f"  {metric_name}: [{len(metric_value)}个类别]")
        
        print("✅ 统一指标计算器测试通过")
        
        # 测试现有FID和CMS模块
        test_fid_and_cms()
        
        print("\n🎉 所有指标模块测试通过！")
        print("新指标已成功集成到demo3项目中。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()