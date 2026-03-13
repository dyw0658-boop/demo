"""
PPO 训练器实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from .env import ACGANEnvironment, ValueNetwork
from .utils import compute_advantages, ppo_update


class PPOTrainer:
    """
    PPO 训练器
    
    Args:
        generator: 生成器模型
        discriminator: 判别器模型
        config: 配置字典
        device: 设备
    """
    
    def __init__(self, generator, discriminator, config, device='cpu'):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.device = device
        
        # 设置非阻塞传输标志
        self.non_blocking = device.type == 'cuda'
        
        # 创建环境
        self.env = ACGANEnvironment(
            generator=generator,
            discriminator=discriminator,
            sigma=config['ppo']['sigma'],
            device=device
        )
        
        # 设置奖励权重
        self.env.set_reward_weights(
            w_adv=config['rewards']['w_adv'],
            w_class=config['rewards']['w_class'],
            w_ssim=config['rewards']['w_ssim'],
            w_entropy=config['rewards']['w_entropy']
        )
        
        # 创建价值网络
        self.value_net = ValueNetwork(
            latent_dim=config['model']['latent_dim'],
            num_classes=config['model']['num_classes'],
            hidden_dim=256
        )
        
        # 如果使用GPU，设置数据并行
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.value_net = nn.DataParallel(self.value_net)
        
        self.value_net.to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=float(config['ppo']['actor_lr'])
        )
        
        self.critic_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=float(config['ppo']['critic_lr'])
        )
        
        # 训练参数
        self.n_steps = config['ppo']['n_steps']
        self.gamma = config['ppo']['gamma']
        self.lambda_gae = config['ppo']['lambda_gae']
        self.clip = config['ppo']['clip']
        self.ent_coef = config['ppo']['ent_coef']
        self.value_coef = config['ppo'].get('value_coef', 0.5)
        self.update_epochs = config['ppo']['update_epochs']
        self.batch_size = config['ppo']['batch_size']
        
        # 训练状态
        self.step_count = 0
        self.episode_rewards = []
    
    def collect_rollout(self):
        """
        收集 rollout 数据
        
        Returns:
            rollout_data: 包含 states, actions, rewards, dones, values, log_probs 的字典
        """
        # 重置环境
        states = self.env.reset(self.n_steps)
        
        # 初始化存储
        rollout_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': [],
            'entropy': []
        }
        
        # 收集数据
        for step in range(self.n_steps):
            # 获取当前状态价值
            noise, labels = states
            with torch.no_grad():
                values = self.value_net(noise, labels)
            
            # 执行一步
            actions, rewards, dones, next_states, info = self.env.step(states)
            
            # 存储数据
            rollout_data['states'].append(states)
            rollout_data['actions'].append(actions)
            rollout_data['rewards'].append(rewards)
            rollout_data['dones'].append(dones)
            rollout_data['values'].append(values)
            rollout_data['log_probs'].append(info['log_probs'])
            rollout_data['entropy'].append(info['entropy'])
            
            # 更新状态
            states = next_states
            
            # 记录奖励
            self.episode_rewards.append(rewards.mean().item())
        
        # 转换为张量
        for key in rollout_data:
            rollout_data[key] = torch.stack(rollout_data[key])
        
        return rollout_data
    
    def train_step(self):
        """执行一次 PPO 训练步骤"""
        # 收集 rollout 数据
        rollout_data = self.collect_rollout()
        
        # 计算优势
        advantages, returns = compute_advantages(
            rewards=rollout_data['rewards'],
            values=rollout_data['values'],
            dones=rollout_data['dones'],
            gamma=self.gamma,
            lambda_gae=self.lambda_gae
        )
        
        # 准备训练数据
        states = rollout_data['states']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['log_probs']
        old_values = rollout_data['values']
        
        # 扁平化数据
        B, T = states[0].size(0), states[0].size(1) if len(states[0].shape) > 1 else 1
        flat_states = [s.reshape(-1, *s.shape[2:]) for s in states]
        flat_actions = actions.reshape(-1, *actions.shape[2:])
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)
        flat_old_values = old_values.reshape(-1)
        
        # PPO 更新
        actor_loss, critic_loss, entropy_loss = ppo_update(
            generator=self.generator,
            value_net=self.value_net,
            states=flat_states,
            actions=flat_actions,
            advantages=flat_advantages,
            returns=flat_returns,
            old_log_probs=flat_old_log_probs,
            old_values=flat_old_values,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            clip=self.clip,
            ent_coef=self.ent_coef,
            value_coef=self.value_coef,
            update_epochs=self.update_epochs,
            batch_size=self.batch_size
        )
        
        # 更新步数
        self.step_count += 1
        
        # 返回训练统计
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss,
            'mean_reward': np.mean(self.episode_rewards[-self.n_steps:]),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item()
        }
    
    def train(self, n_updates, eval_callback=None):
        """
        训练 PPO
        
        Args:
            n_updates (int): 更新次数
            eval_callback: 评估回调函数
        """
        self.generator.train()
        
        pbar = tqdm(range(n_updates), desc="PPO Training")
        
        for update in pbar:
            # 训练一步
            stats = self.train_step()
            
            # 更新进度条
            pbar.set_postfix({
                'reward': f"{stats['mean_reward']:.3f}",
                'actor_loss': f"{stats['actor_loss']:.3f}",
                'critic_loss': f"{stats['critic_loss']:.3f}"
            })
            
            # 评估回调
            if eval_callback and (update + 1) % 10 == 0:
                eval_callback(self.generator, update + 1)
        
        # 训练完成
        self.generator.eval()
    
    def save_checkpoint(self, path):
        """保存检查点"""
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'step_count': self.step_count,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.step_count = checkpoint['step_count']