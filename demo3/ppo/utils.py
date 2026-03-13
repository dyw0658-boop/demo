"""
PPO 工具函数
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_advantages(rewards, values, dones, gamma=0.99, lambda_gae=0.95):
    """
    计算 GAE 优势
    
    Args:
        rewards: 奖励，shape [T, B]
        values: 价值估计，shape [T, B, 1]
        dones: 终止标志，shape [T, B]
        gamma: 折扣因子
        lambda_gae: GAE lambda 参数
    
    Returns:
        advantages: 优势，shape [T, B]
        returns: 回报，shape [T, B]
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    # 计算最后一个时间步的下一价值
    next_value = 0.0
    next_advantage = 0.0
    
    # 反向计算
    for t in reversed(range(T)):
        # 计算 TD 残差
        delta = rewards[t] + gamma * next_value * (~dones[t]).float() - values[t].squeeze()
        
        # 计算优势
        advantages[t] = delta + gamma * lambda_gae * next_advantage * (~dones[t]).float()
        
        # 计算回报
        returns[t] = advantages[t] + values[t].squeeze()
        
        # 更新下一价值
        next_value = values[t].squeeze()
        next_advantage = advantages[t]
    
    # 标准化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def ppo_update(generator, value_net, states, actions, advantages, returns, 
               old_log_probs, old_values, actor_optimizer, critic_optimizer,
               clip=0.2, ent_coef=0.01, value_coef=0.5, update_epochs=4, batch_size=64):
    """
    PPO 更新步骤
    
    Args:
        generator: 生成器模型
        value_net: 价值网络
        states: 状态列表 [(noise, labels)]
        actions: 动作，shape [N, 3, 32, 32]
        advantages: 优势，shape [N]
        returns: 回报，shape [N]
        old_log_probs: 旧对数概率，shape [N]
        old_values: 旧价值估计，shape [N]
        actor_optimizer: 演员优化器
        critic_optimizer: 评论家优化器
        clip: PPO clip 参数
        ent_coef: 熵系数
        value_coef: 价值损失系数
        update_epochs: 更新轮数
        batch_size: 批量大小
    
    Returns:
        actor_loss: 演员损失
        critic_loss: 评论家损失
        entropy_loss: 熵损失
    """
    N = actions.size(0)
    
    # 准备数据索引
    indices = torch.randperm(N)
    
    actor_losses = []
    critic_losses = []
    entropy_losses = []
    
    for epoch in range(update_epochs):
        for start in range(0, N, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            # 获取批次数据
            batch_states = [s[batch_indices] for s in states]
            batch_actions = actions[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_old_values = old_values[batch_indices]
            
            # 解包状态
            batch_noise, batch_labels = batch_states
            
            # 演员更新
            actor_optimizer.zero_grad()
            
            # 计算新策略的对数概率
            with torch.no_grad():
                mu = generator(batch_noise, batch_labels)
            
            # 创建动作分布
            action_dist = torch.distributions.Normal(mu, generator.env.sigma)
            batch_new_log_probs = action_dist.log_prob(batch_actions).sum(dim=[1, 2, 3])
            
            # 计算概率比
            ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
            
            # 计算裁剪的演员损失
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算熵损失
            entropy = action_dist.entropy().sum(dim=[1, 2, 3]).mean()
            entropy_loss = -ent_coef * entropy
            
            # 总演员损失
            total_actor_loss = actor_loss + entropy_loss
            total_actor_loss.backward()
            actor_optimizer.step()
            
            # 评论家更新
            critic_optimizer.zero_grad()
            
            # 计算新价值估计
            batch_new_values = value_net(batch_noise, batch_labels).squeeze()
            
            # 计算价值损失 (MSE)
            critic_loss = F.mse_loss(batch_new_values, batch_returns)
            
            # 可选: 添加价值裁剪
            value_loss_clipped = batch_old_values + torch.clamp(
                batch_new_values - batch_old_values, -clip, clip
            )
            critic_loss_clipped = F.mse_loss(value_loss_clipped, batch_returns)
            
            critic_loss = torch.max(critic_loss, critic_loss_clipped)
            
            # 总评论家损失
            total_critic_loss = value_coef * critic_loss
            total_critic_loss.backward()
            critic_optimizer.step()
            
            # 记录损失
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy.item())
    
    return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses)


def create_rollout_dataset(rollout_data):
    """
    创建 rollout 数据集
    
    Args:
        rollout_data: rollout 数据字典
    
    Returns:
        dataset: 可用于训练的数据集
    """
    # 扁平化数据
    states = rollout_data['states']
    actions = rollout_data['actions']
    rewards = rollout_data['rewards']
    dones = rollout_data['dones']
    values = rollout_data['values']
    log_probs = rollout_data['log_probs']
    
    # 计算优势
    advantages, returns = compute_advantages(rewards, values, dones)
    
    # 创建数据集
    dataset = {
        'states': states,
        'actions': actions,
        'advantages': advantages,
        'returns': returns,
        'old_log_probs': log_probs,
        'old_values': values
    }
    
    return dataset