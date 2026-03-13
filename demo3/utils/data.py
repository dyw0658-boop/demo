"""
数据加载和预处理工具
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


def get_cifar10_dataset(data_root='./data', train=True, image_size=32, download=True):
    """
    获取 CIFAR-10 数据集对象（不包含数据加载器）
    
    Args:
        data_root: 数据根目录
        train: 是否获取训练集
        image_size: 图像大小
        download: 是否下载数据
    
    Returns:
        dataset: CIFAR-10 数据集对象
        num_classes: 类别数量
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    # 加载数据集
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=train, download=download, transform=transform
    )
    
    return dataset, 10


def get_cifar10_dataloader(data_root='./data', batch_size=128, num_workers=4, 
                          image_size=32, download=True):
    """
    获取 CIFAR-10 数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批量大小
        num_workers: 工作进程数
        image_size: 图像大小
        download: 是否下载数据
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_classes: 类别数量
    """
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    # 加载训练集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=download, transform=transform_train
    )
    
    # 加载测试集
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=download, transform=transform_test
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader, 10


def get_dataset_statistics(dataset):
    """
    获取数据集的统计信息
    
    Args:
        dataset: 数据集
    
    Returns:
        stats: 统计信息字典
    """
    # 计算均值和标准差
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    mean = 0.0
    std = 0.0
    num_samples = 0
    
    for images, _ in data_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += batch_samples
    
    mean /= num_samples
    std /= num_samples
    
    return {'mean': mean, 'std': std, 'num_samples': num_samples}


def denormalize_images(images, mean=0.5, std=0.5):
    """
    反标准化图像
    
    Args:
        images: 标准化后的图像，值域 [-1, 1]
        mean: 均值
        std: 标准差
    
    Returns:
        denormalized: 反标准化后的图像，值域 [0, 1]
    """
    # 反标准化: (x * std) + mean
    denormalized = images * std + mean
    
    # 裁剪到 [0, 1] 范围
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized


def normalize_images(images, mean=0.5, std=0.5):
    """
    标准化图像
    
    Args:
        images: 原始图像，值域 [0, 1]
        mean: 均值
        std: 标准差
    
    Returns:
        normalized: 标准化后的图像，值域 [-1, 1]
    """
    # 标准化: (x - mean) / std
    normalized = (images - mean) / std
    
    return normalized


def create_smoke_test_dataset(batch_size=32, image_size=32):
    """
    创建烟雾测试数据集
    
    Args:
        batch_size: 批量大小
        image_size: 图像大小
    
    Returns:
        smoke_loader: 烟雾测试数据加载器
    """
    # 创建随机数据
    num_samples = batch_size * 10  # 10 个批次
    
    # 随机图像
    images = torch.randn(num_samples, 3, image_size, image_size)
    images = torch.clamp(images, -1, 1)  # 限制到 [-1, 1]
    
    # 随机标签
    labels = torch.randint(0, 10, (num_samples,))
    
    # 创建数据集
    class SmokeDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]
    
    dataset = SmokeDataset(images, labels)
    
    # 创建数据加载器
    smoke_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return smoke_loader


def get_class_balanced_samples(dataset, num_samples_per_class=100):
    """
    获取类别平衡的样本
    
    Args:
        dataset: 数据集
        num_samples_per_class: 每类样本数
    
    Returns:
        balanced_samples: 平衡样本
        balanced_labels: 平衡标签
    """
    # 按类别分组
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # 采样
    sampled_indices = []
    for label, indices in class_indices.items():
        # 随机采样
        if len(indices) >= num_samples_per_class:
            sampled = np.random.choice(indices, num_samples_per_class, replace=False)
        else:
            sampled = np.random.choice(indices, num_samples_per_class, replace=True)
        
        sampled_indices.extend(sampled)
    
    # 创建子集
    balanced_dataset = torch.utils.data.Subset(dataset, sampled_indices)
    
    return balanced_dataset


def save_dataset_samples(dataset, num_samples=100, filename='dataset_samples.png'):
    """
    保存数据集样本
    
    Args:
        dataset: 数据集
        num_samples: 样本数量
        filename: 保存文件名
    """
    import matplotlib.pyplot as plt
    from .visualize import save_grid
    
    # 随机采样
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # 获取样本
    samples = []
    for idx in indices:
        image, _ = dataset[idx]
        samples.append(image)
    
    samples = torch.stack(samples)
    
    # 保存网格
    save_grid(samples, nrow=10, filename=filename)


if __name__ == "__main__":
    # 测试代码
    train_loader, test_loader, num_classes = get_cifar10_dataloader(batch_size=32)
    
    print(f"CIFAR-10 数据集加载成功")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"类别数量: {num_classes}")
    
    # 测试一个批次
    for images, labels in train_loader:
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"图像值域: [{images.min():.3f}, {images.max():.3f}]")
        break