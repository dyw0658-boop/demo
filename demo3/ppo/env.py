"""
PPO 环境定义，将生成器作为策略网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class ACGANEnvironment:
    """
    ACGAN PPO 环境
    
    Args:
        generator: 生成器模型
        discriminator: 判别器模型
        sigma (float): 动作噪声标准差
        device: 设备
    """
    
    def __init__(self, generator, discriminator, sigma=0.1, device='cpu',
                 w_adv=1.0, w_class=1.0, w_ssim=1.0, w_entropy=0.01):
        self.generator = generator
        self.discriminator = discriminator
        self.sigma = sigma
        self.device = device
        
        # 奖励权重
        self.w_adv = w_adv
        self.w_class = w_class
        self.w_ssim = w_ssim
        self.w_entropy = w_entropy
        
        # 冻结判别器
        for param in self.discriminator.parameters():
            param.requires_grad = False
        
        self.generator.eval()
        self.discriminator.eval()
    
    def reset(self, batch_size):
        """
        重置环境
        
        Args:
            batch_size (int): 批量大小
        
        Returns:
            states: 初始状态 (噪声和标签)，tuple (noise, labels)
        """
        # 生成随机噪声和标签
        noise = self.generator.sample_noise(batch_size, self.device)
        labels = self.generator.sample_labels(batch_size, self.device)
        
        # 如果使用GPU，设置非阻塞传输
        if self.device.type == 'cuda':
            noise = noise.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
        
        return (noise, labels)
    
    def step(self, states):
        """
        执行一步动作
        
        Args:
            states: 当前状态，tuple (noise, labels)
        
        Returns:
            actions: 动作 (生成的图像)
            rewards: 奖励
            dones: 是否结束
            next_states: 下一状态
            info: 额外信息
        """
        noise, labels = states
        
        # 生成图像作为均值
        with torch.no_grad():
            mu = self.generator(noise, labels)  # [B, 3, 32, 32]
        
        # 从高斯分布中采样动作
        action_dist = dist.Normal(mu, self.sigma)
        actions = action_dist.sample()  # [B, 3, 32, 32]
        
        # 计算动作的对数概率和熵
        log_probs = action_dist.log_prob(actions).sum(dim=[1, 2, 3])  # [B]
        entropy = action_dist.entropy().sum(dim=[1, 2, 3])  # [B]
        
        # 计算奖励
        rewards, reward_info = self._compute_rewards(actions, labels, entropy)
        
        # 环境总是继续 (非终止)
        dones = torch.zeros(actions.size(0), dtype=torch.bool, device=self.device)
        
        # 下一状态相同 (PPO 中状态不变)
        next_states = (noise, labels)
        
        # 额外信息
        info = {
            'log_probs': log_probs,
            'entropy': entropy,
            'mu': mu,
            **reward_info
        }
        
        return actions, rewards, dones, next_states, info
    
    def _compute_rewards(self, actions, labels, entropy):
        """
        计算奖励函数
        
        Args:
            actions: 动作 (生成的图像)，shape [B, 3, 32, 32]
            labels: 标签，shape [B]
            entropy: 动作熵，shape [B]
        
        Returns:
            rewards: 总奖励，shape [B]
            info: 奖励分量信息
        """
        with torch.no_grad():
            # 判别器输出
            validity, class_logits, features = self.discriminator(actions)
            class_probs = F.softmax(class_logits, dim=1)
            
            # 1. 对抗奖励 (使用判别器对假样本的输出)
            d_fake_prob = torch.sigmoid(validity).squeeze()  # [B]
            adv_reward = -torch.log(1 - d_fake_prob + 1e-8)  # [B]
            
            # 2. 分类奖励 (对应标签的概率)
            class_reward = torch.log(class_probs[torch.arange(class_probs.size(0)), labels] + 1e-8)  # [B]
            
            # 3. SSIM 奖励 (与同类真实样本对比)
            ssim_reward = self._compute_ssim_reward(actions, labels)  # [B]
            
            # 4. 熵奖励 (鼓励多样性)
            entropy_reward = entropy
            
            # 组合奖励 (权重在配置中设置)
            rewards = (
                self.w_adv * adv_reward +
                self.w_class * class_reward +
                self.w_ssim * ssim_reward +
                self.w_entropy * entropy_reward
            )
            
            info = {
                'adv_reward': adv_reward,
                'class_reward': class_reward,
                'ssim_reward': ssim_reward,
                'entropy_reward': entropy_reward,
                'd_fake_prob': d_fake_prob,
                'class_probs': class_probs
            }
            
            return rewards, info
    
    def _compute_ssim_reward(self, generated, labels):
        """
        计算 SSIM 奖励 - 使用真实数据作为参考
        
        Args:
            generated: 生成的图像，shape [B, 3, 32, 32]
            labels: 标签，shape [B]
        
        Returns:
            ssim_reward: SSIM 奖励，shape [B]
        """
        try:
            from pytorch_msssim import ssim
            
            # 将图像从 [-1, 1] 转换到 [0, 1]
            generated_01 = (generated + 1) / 2
            
            # 从真实数据集中采样同类样本作为参考
            reference = self._sample_real_images_by_labels(labels)
            
            # 计算 SSIM (每个生成样本与对应的真实样本对比)
            ssim_values = ssim(generated_01, reference, data_range=1.0, size_average=False)
            
            # SSIM 值在 [0, 1] 之间，越高表示质量越好
            return ssim_values
            
        except ImportError:
            # 如果 pytorch_msssim 不可用，返回零奖励
            print("警告: pytorch_msssim 未安装，SSIM 奖励设为0")
            return torch.zeros(generated.size(0), device=generated.device)
    
    def _sample_real_images_by_labels(self, labels):
        """
        根据标签从真实数据集中采样图像 - 学术研究严谨实现
        
        Args:
            labels: 标签，shape [B]
        
        Returns:
            real_images: 真实图像，shape [B, 3, 32, 32]
        """
        if not hasattr(self, '_real_dataset') or self._real_dataset is None:
            # 延迟加载真实数据集 (不是数据加载器，以确保可重复性)
            from utils.data import get_cifar10_dataset
            self._real_dataset, _ = get_cifar10_dataset(train=True)
            
            # 创建类别索引映射，提高采样效率
            self._class_indices = {}
            for class_idx in range(10):
                self._class_indices[class_idx] = torch.where(
                    torch.tensor(self._real_dataset.targets) == class_idx
                )[0]
        
        batch_size = len(labels)
        selected_real_images = []
        
        # 为每个标签选择对应的真实样本
        for label in labels:
            label = label.item() if isinstance(label, torch.Tensor) else label
            
            # 获取该类别的所有样本索引
            class_indices = self._class_indices.get(label, [])
            
            if len(class_indices) > 0:
                # 随机选择一个同类真实样本 (确保可重复性)
                random_idx = torch.randint(0, len(class_indices), (1,)).item()
                sample_idx = class_indices[random_idx]
                
                # 获取图像并标准化到 [-1, 1]
                image, _ = self._real_dataset[sample_idx]
                image = image.unsqueeze(0).to(self.device)
                
                selected_real_images.append(image)
            else:
                # 如果没有找到同类样本，使用零图像 (不应该发生)
                zero_image = torch.zeros(1, 3, 32, 32, device=self.device)
                selected_real_images.append(zero_image)
        
        selected_real_images = torch.cat(selected_real_images, dim=0)
        
        return selected_real_images
    
    def set_reward_weights(self, w_adv=1.0, w_class=0.1, w_ssim=0.5, w_entropy=0.01):
        """设置奖励权重"""
        self.w_adv = w_adv
        self.w_class = w_class
        self.w_ssim = w_ssim
        self.w_entropy = w_entropy


class ValueNetwork(nn.Module):
    """
    价值网络，用于 PPO
    
    Args:
        latent_dim (int): 噪声维度
        num_classes (int): 类别数量
        hidden_dim (int): 隐藏层维度
    """
    
    def __init__(self, latent_dim=100, num_classes=10, hidden_dim=256):
        super().__init__()
        
        # 输入: 噪声 + 标签 one-hot
        input_dim = latent_dim + num_classes
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, noise, labels):
        """
        前向传播
        
        Args:
            noise: 噪声，shape [B, latent_dim]
            labels: 标签，shape [B]
        
        Returns:
            values: 状态价值，shape [B, 1]
        """
        # 将标签转换为 one-hot
        labels_one_hot = F.one_hot(labels, num_classes=self.network[0].in_features - noise.size(1))
        
        # 拼接噪声和 one-hot 标签
        state = torch.cat([noise, labels_one_hot.float()], dim=1)
        
        # 通过价值网络
        values = self.network(state)
        
        return values